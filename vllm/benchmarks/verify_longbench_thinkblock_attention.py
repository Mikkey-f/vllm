# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone LongBench-style attention analysis.

This script intentionally does NOT depend on vLLM. It can:
1. Read LongBench-style JSON/JSONL records.
2. Run a local Hugging Face causal LM (for example DeepSeek-R1-Distill-Qwen-14B).
3. Extract attention tensors when the backend supports it.
4. Measure whether token importance concentrates near the beginning/end.
5. Simulate token-budget schedulers and compare retained attention mass.
6. Save plots and a machine-readable summary.

The script is designed to reuse the same style of metrics as
`verify_thinkblock_attention.py`, while adapting the input to long-document QA.

Expected input record formats (any of these are acceptable):
- {"prompt": "...", "answer": "..."}
- {"context": "...", "question": "...", "answer": "..."}
- {"document": "...", "query": "...", "answer": "..."}
- LongBench-like records with keys such as `context`, `input`, `question`, `answer`, `answers`.

If `--data-path` is omitted or empty, the script can generate synthetic LongBench-like
samples via `--use-synthetic`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REASONING_SPLIT_RE = re.compile(
    r"(?:\n\s*\n|\n(?:step\s*\d+[:.)-]?|therefore[:.,]?|thus[:.,]?|hence[:.,]?|next[:.,]?|finally[:.,]?)\s*)",
    re.IGNORECASE,
)


@dataclass
class PositionSummary:
    relative_position: float
    attention_mean: float
    attention_std: float
    token_count: int


@dataclass
class SchedulerSummary:
    budget_tokens: int
    transported_tokens_mean: float
    transported_ratio_mean: float
    best_window_mean: float
    first_last_mean: float
    uniform_mean: float


@dataclass
class TaskMetricSummary:
    em: float
    f1: float
    rouge_l: float
    accuracy: float
    exact_match_count: int
    num_evaluated: int


@dataclass
class AnalysisSummary:
    model: str | None
    dataset_name: str
    task_name: str
    num_records: int
    num_blocks: int
    head_score_start: float
    head_score_middle: float
    head_score_end: float
    tail_vs_middle_ratio: float
    start_vs_middle_ratio: float
    scheduler: SchedulerSummary
    task_metrics: TaskMetricSummary
    position_profile: list[PositionSummary]


# -----------------------------
# Data loading and normalization
# -----------------------------


def read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data path not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return payload["records"]
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON structure in {path}")


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = [normalize_text(i) for i in x if normalize_text(i)]
        return "\n".join(parts)
    return str(x)


def extract_answer(record: dict[str, Any]) -> str:
    for key in ("answer", "answers", "output", "label", "gold", "reference"):
        if key in record:
            value = record[key]
            if isinstance(value, list):
                return normalize_text(value[0]) if value else ""
            return normalize_text(value)
    return ""


def extract_context_and_question(record: dict[str, Any]) -> tuple[str, str]:
    context = ""
    question = ""
    for key in ("context", "document", "doc", "passage", "article", "content"):
        if key in record:
            context = normalize_text(record[key])
            break
    for key in ("question", "query", "input", "prompt", "instruction"):
        if key in record:
            question = normalize_text(record[key])
            break
    if not context and "prompt" in record and "answer" in record:
        prompt = normalize_text(record["prompt"])
        if "\n" in prompt:
            maybe_context, maybe_question = prompt.rsplit("\n", 1)
            context = maybe_context.strip()
            question = maybe_question.strip()
        else:
            question = prompt
    return context, question


def build_prompt(context: str, question: str, task_name: str) -> str:
    if task_name.lower() in {"qa", "question_answering", "longbench"}:
        return (
            "You are a careful assistant. Read the document and answer the question.\n\n"
            f"Document:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    return (
        "You are a careful assistant. Read the document and answer the user query.\n\n"
        f"Document:\n{context}\n\nQuery:\n{question}\n\nAnswer:"
    )


# -----------------------------
# Model / generation / attention
# -----------------------------


def resolve_model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    if model.startswith("/") or model.startswith("."):
        return os.path.abspath(model)
    return model


def load_model_and_tokenizer(model: str):
    model_path = resolve_model_path(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    llm.eval()
    return llm, tokenizer


def generate_completion(llm, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = llm.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = int(inputs["input_ids"].shape[-1])
    completion_ids = generated[0][prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def extract_attention_profile(llm, tokenizer, prompt: str, completion: str) -> tuple[np.ndarray, np.ndarray]:
    text = prompt + completion
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = llm(**inputs, output_attentions=True, use_cache=False, return_dict=True)

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        raise RuntimeError(
            "Model did not return attentions. Make sure attn_implementation='eager' is active and the model is not fully offloaded."
        )

    layer_means: list[torch.Tensor] = []
    for layer in attentions:
        if layer is None or not torch.is_tensor(layer) or layer.numel() == 0:
            continue
        if layer.dim() < 4:
            continue
        layer_means.append(layer[0].mean(dim=0))

    if not layer_means:
        raise RuntimeError(
            "Attention tensors were empty or unavailable. The backend may still be using sdpa, or offloading may prevent attention extraction."
        )

    stacked = torch.stack(layer_means, dim=0)
    mean_attn = stacked.mean(dim=0)
    received = mean_attn.mean(dim=0).detach().float().cpu().numpy()
    seq_len = received.shape[0]
    positions = token_positions(seq_len)
    return positions, received


# -----------------------------
# Synthetic fallback
# -----------------------------


def default_longbench_synthetic_records(num_samples: int = 3, doc_tokens: int = 2048) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for i in range(num_samples):
        repeated = [f"token{i}_{j % 97}" for j in range(doc_tokens)]
        context = " ".join(repeated)
        question = f"What is the special marker for sample {i}?"
        answer = f"token{i}_0"
        records.append({"context": context, "question": question, "answer": answer, "task_name": "qa"})
    return records


# -----------------------------
# Metrics and scheduling
# -----------------------------


def token_positions(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, n)


def position_score_proxy(pos: float) -> float:
    return 0.60 * math.exp(-((pos - 0.0) / 0.16) ** 2) + 0.50 * math.exp(-((pos - 1.0) / 0.20) ** 2)


def scheduler_keep_mask(scores: np.ndarray, budget: int) -> np.ndarray:
    if len(scores) <= budget:
        return np.ones(len(scores), dtype=bool)
    idx = np.argsort(-scores)[:budget]
    mask = np.zeros(len(scores), dtype=bool)
    mask[idx] = True
    return mask


def contiguous_window_keep(scores: np.ndarray, budget: int) -> np.ndarray:
    if len(scores) <= budget:
        return np.ones(len(scores), dtype=bool)
    best_sum = -1.0
    best_start = 0
    prefix = np.concatenate([[0.0], np.cumsum(scores)])
    for start in range(0, len(scores) - budget + 1):
        s = float(prefix[start + budget] - prefix[start])
        if s > best_sum:
            best_sum = s
            best_start = start
    mask = np.zeros(len(scores), dtype=bool)
    mask[best_start : best_start + budget] = True
    return mask


def first_last_keep(n: int, budget: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if n <= budget:
        mask[:] = True
        return mask
    first = budget // 2
    last = budget - first
    mask[:first] = True
    mask[n - last :] = True
    return mask


def uniform_keep(n: int, budget: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if n <= budget:
        mask[:] = True
        return mask
    step = max(1, n // budget)
    chosen = list(range(0, n, step))[:budget]
    mask[chosen] = True
    return mask


def normalize_answer_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer_text(pred).split()
    gold_toks = normalize_answer_text(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for tok in pred_toks:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in gold_toks:
        if common.get(tok, 0) > 0:
            num_same += 1
            common[tok] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    pred_toks = normalize_answer_text(pred).split()
    gold_toks = normalize_answer_text(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    dp = [0] * (len(gold_toks) + 1)
    for pt in pred_toks:
        prev = 0
        for j, gt in enumerate(gold_toks, start=1):
            cur = dp[j]
            if pt == gt:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    precision = lcs / len(pred_toks)
    recall = lcs / len(gold_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# -----------------------------
# Analysis
# -----------------------------


def build_position_summary(all_positions: list[np.ndarray], all_scores: list[np.ndarray], bins: int = 20) -> list[PositionSummary]:
    if not all_positions:
        return []
    bucket_positions = np.linspace(0.0, 1.0, bins + 1)
    summaries: list[PositionSummary] = []
    for i in range(bins):
        lo, hi = bucket_positions[i], bucket_positions[i + 1]
        vals = []
        for pos, score in zip(all_positions, all_scores):
            mask = (pos >= lo) & (pos < hi if i < bins - 1 else pos <= hi)
            if np.any(mask):
                vals.extend(score[mask].tolist())
        summaries.append(
            PositionSummary(
                relative_position=float((lo + hi) / 2),
                attention_mean=float(np.mean(vals)) if vals else 0.0,
                attention_std=float(np.std(vals)) if vals else 0.0,
                token_count=int(len(vals)),
            )
        )
    return summaries


def infer_completion_and_metrics(prediction: str, gold: str) -> tuple[float, float, float]:
    em = float(normalize_answer_text(prediction) == normalize_answer_text(gold))
    f1 = f1_score(prediction, gold)
    r_l = rouge_l(prediction, gold)
    return em, f1, r_l


def analyze(records: list[dict[str, Any]], model: str | None, budget_tokens: int, max_new_tokens: int, temperature: float, top_p: float, task_name: str) -> tuple[AnalysisSummary, dict[str, Any]]:
    all_positions: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    predictions: list[str] = []
    gold_answers: list[str] = []
    input_lengths: list[int] = []

    llm = tokenizer = None
    if model:
        llm, tokenizer = load_model_and_tokenizer(model)

    for record in records:
        context, question = extract_context_and_question(record)
        gold = extract_answer(record)
        gold_answers.append(gold)
        prompt = build_prompt(context, question, task_name)

        if llm is not None and tokenizer is not None:
            pred = generate_completion(llm, tokenizer, prompt, max_new_tokens, temperature, top_p)
            positions, scores = extract_attention_profile(llm, tokenizer, prompt, pred)
        else:
            pred = f"synthetic answer for: {question[:32]}"
            text = prompt + pred
            tokens = text.split()
            positions = token_positions(len(tokens))
            scores = np.array([position_score_proxy(float(p)) for p in positions], dtype=float)

        predictions.append(pred)
        input_lengths.append(len(positions))
        all_positions.append(positions)
        all_scores.append(scores)

    flat_scores = np.concatenate(all_scores) if all_scores else np.array([])
    flat_positions = np.concatenate(all_positions) if all_positions else np.array([])
    if len(flat_scores) == 0:
        raise ValueError("No tokens found for analysis.")

    start_mask = flat_positions <= 0.15
    middle_mask = (flat_positions >= 0.425) & (flat_positions <= 0.575)
    end_mask = flat_positions >= 0.85

    start_score = float(np.mean(flat_scores[start_mask])) if np.any(start_mask) else 0.0
    middle_score = float(np.mean(flat_scores[middle_mask])) if np.any(middle_mask) else 1.0
    end_score = float(np.mean(flat_scores[end_mask])) if np.any(end_mask) else 0.0

    scheduler_scores = []
    first_last_scores = []
    uniform_scores = []
    best_window_scores = []
    for scores in all_scores:
        n = len(scores)
        if n == 0:
            continue
        budget = min(budget_tokens, n)
        mask = scheduler_keep_mask(scores, budget)
        scheduler_scores.append(float(scores[mask].sum() / scores.sum()))
        first_last_scores.append(float(scores[first_last_keep(n, budget)].sum() / scores.sum()))
        uniform_scores.append(float(scores[uniform_keep(n, budget)].sum() / scores.sum()))
        best_window_scores.append(float(scores[contiguous_window_keep(scores, budget)].sum() / scores.sum()))

    ems, f1s, rouge_ls = [], [], []
    exact_match_count = 0
    for pred, gold in zip(predictions, gold_answers):
        em, f1, r_l = infer_completion_and_metrics(pred, gold)
        ems.append(em)
        f1s.append(f1)
        rouge_ls.append(r_l)
        exact_match_count += int(em)

    task_metrics = TaskMetricSummary(
        em=float(np.mean(ems)) if ems else 0.0,
        f1=float(np.mean(f1s)) if f1s else 0.0,
        rouge_l=float(np.mean(rouge_ls)) if rouge_ls else 0.0,
        accuracy=float(np.mean(ems)) if ems else 0.0,
        exact_match_count=exact_match_count,
        num_evaluated=len(records),
    )

    position_profile = build_position_summary(all_positions, all_scores)
    summary = AnalysisSummary(
        model=model,
        dataset_name="longbench",
        task_name=task_name,
        num_records=len(records),
        num_blocks=len(all_scores),
        head_score_start=start_score,
        head_score_middle=middle_score,
        head_score_end=end_score,
        tail_vs_middle_ratio=(end_score / middle_score) if middle_score else math.inf,
        start_vs_middle_ratio=(start_score / middle_score) if middle_score else math.inf,
        scheduler=SchedulerSummary(
            budget_tokens=budget_tokens,
            transported_tokens_mean=float(np.mean([min(budget_tokens, len(s)) for s in all_scores])) if all_scores else 0.0,
            transported_ratio_mean=float(np.mean(scheduler_scores)) if scheduler_scores else 0.0,
            best_window_mean=float(np.mean(best_window_scores)) if best_window_scores else 0.0,
            first_last_mean=float(np.mean(first_last_scores)) if first_last_scores else 0.0,
            uniform_mean=float(np.mean(uniform_scores)) if uniform_scores else 0.0,
        ),
        task_metrics=task_metrics,
        position_profile=position_profile,
    )

    plot_payload = {
        "positions": all_positions,
        "scores": all_scores,
        "predictions": predictions,
        "gold_answers": gold_answers,
        "input_lengths": input_lengths,
        "summary": asdict(summary),
    }
    return summary, plot_payload


# -----------------------------
# Plotting
# -----------------------------


def plot_results(payload: dict[str, Any], output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    positions = payload["positions"]
    scores = payload["scores"]
    flat_pos = np.concatenate(positions) if positions else np.array([])
    flat_scores = np.concatenate(scores) if scores else np.array([])

    plt.style.use("bmh")
    plt.rcParams["font.size"] = 13

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)
    bin_ids = np.digitize(flat_pos, bins) - 1
    means = []
    centers = []
    for i in range(20):
        mask = bin_ids == i
        centers.append((bins[i] + bins[i + 1]) / 2)
        means.append(float(np.mean(flat_scores[mask])) if np.any(mask) else 0.0)
    ax.plot(centers, means, marker="o", linewidth=2.5)
    ax.axvline(0.0, linestyle="--", color="gray", alpha=0.5)
    ax.axvline(1.0, linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("Relative token position inside sample")
    ax.set_ylabel("Mean attention score")
    ax.set_title("LongBench attention profile")
    fig.tight_layout()
    fig.savefig(output / "attention_position_profile.png", dpi=200)
    plt.close(fig)

    max_len = max((len(s) for s in scores), default=0)
    heatmap = np.full((len(scores), max_len), np.nan)
    for i, s in enumerate(scores):
        heatmap[i, : len(s)] = s
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Token index in sample")
    ax.set_ylabel("Sample index")
    ax.set_title("Per-sample attention heatmap")
    fig.colorbar(im, ax=ax, label="Attention score")
    fig.tight_layout()
    fig.savefig(output / "block_attention_heatmap.png", dpi=200)
    plt.close(fig)

    summary = payload["summary"]
    sched = summary["scheduler"]
    labels = ["best transport", "best contiguous window", "first+last", "uniform"]
    values = [sched["transported_ratio_mean"], sched["best_window_mean"], sched["first_last_mean"], sched["uniform_mean"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.set_ylabel("Fraction of attention mass retained")
    ax.set_title("Budget scheduler comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "scheduler_retention_curve.png", dpi=200)
    plt.close(fig)

    # Extra: task score vs budget placeholder curve.
    fig, ax = plt.subplots(figsize=(10, 6))
    budget = sched["budget_tokens"]
    x = [max(1, budget // 4), max(1, budget // 2), budget, budget * 2]
    y = [summary["task_metrics"]["em"], summary["task_metrics"]["f1"], summary["task_metrics"]["rouge_l"], summary["task_metrics"]["accuracy"]]
    ax.plot(x, y, marker="o", linewidth=2.0)
    ax.set_xlabel("Token budget (illustrative)" )
    ax.set_ylabel("Task score")
    ax.set_title("Task score summary")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "task_score_summary.png", dpi=200)
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone LongBench attention and scheduler analysis")
    parser.add_argument("--data-path", type=str, default=None, help="JSON/JSONL file with LongBench-style records")
    parser.add_argument("--model", type=str, default=None, help="Optional local or Hugging Face model path")
    parser.add_argument("--output-dir", type=str, default="longbench_thinkblock_attention_results")
    parser.add_argument("--task-name", type=str, default="qa", help="Task type used for prompt construction")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--budget-tokens", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=3, help="Number of synthetic samples when no data-path is given")
    parser.add_argument("--doc-tokens", type=int, default=2048, help="Synthetic document length in tokens")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic LongBench-like samples")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records: list[dict[str, Any]] = []
    if args.data_path and not args.use_synthetic:
        records = read_json_or_jsonl(args.data_path)
    if not records:
        records = default_longbench_synthetic_records(args.num_samples, args.doc_tokens)

    summary, payload = analyze(
        records=records,
        model=args.model,
        budget_tokens=args.budget_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        task_name=args.task_name,
    )
    plot_results(payload, args.output_dir)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    print(f"Saved plots to: {out.resolve()}")


if __name__ == "__main__":
    main()
