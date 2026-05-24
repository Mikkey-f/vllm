# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify whether thinkblock attention concentrates near the start and end.

This script is intentionally standalone so it can be run without depending on
internal vLLM modules.

What it does:
1. Loads prompts/completions from a JSON file or generates them from a model.
2. Extracts token-level attention statistics from the model when available.
3. Aggregates attention by relative position inside each thinkblock.
4. Simulates a small token-budget scheduler that preferentially keeps the most
   informative tokens under a fixed transport budget.
5. Saves plots showing whether the beginning and ending positions receive the
   highest scores.

Input JSON format:
{
  "records": [
    {"prompt": "...", "completion": "..."},
    ...
  ]
}

Output directory will contain:
- attention_position_profile.png
- block_attention_heatmap.png
- scheduler_retention_curve.png
- summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
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
class AnalysisSummary:
    model: str | None
    num_records: int
    num_blocks: int
    head_score_start: float
    head_score_middle: float
    head_score_end: float
    tail_vs_middle_ratio: float
    start_vs_middle_ratio: float
    scheduler: SchedulerSummary
    position_profile: list[PositionSummary]


def split_thinkblocks(text: str) -> list[str]:
    blocks = [b.strip() for b in _REASONING_SPLIT_RE.split(text) if b and b.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


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
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    llm.eval()
    return llm, tokenizer


def read_records(input_json: str | None) -> list[dict[str, str]]:
    if not input_json:
        return []
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    return payload.get("records", [])


def generate_records(model: str, prompts: list[str], max_tokens: int, temperature: float, top_p: float) -> list[dict[str, str]]:
    llm, tokenizer = load_model_and_tokenizer(model)
    records: list[dict[str, str]] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = llm.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[-1])
        completion_ids = generated[0][prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        records.append({"prompt": prompt, "completion": completion_text})
    return records


def token_positions(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, n)


def position_score_proxy(pos: float) -> float:
    """Proxy for the hypothesized U-shape: high at beginning and end."""
    return 0.55 * math.exp(-((pos - 0.0) / 0.18) ** 2) + 0.55 * math.exp(-((pos - 1.0) / 0.18) ** 2)


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


def extract_attention_profile(llm, tokenizer, prompt: str, completion: str) -> tuple[np.ndarray, np.ndarray]:
    text = prompt + completion
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = llm(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attentions.")

    # attentions: tuple[num_layers] with shape [batch, heads, seq, seq]
    stacked = torch.stack([layer[0].mean(dim=0) for layer in attentions], dim=0)  # [layers, seq, seq]
    mean_attn = stacked.mean(dim=0)  # [seq, seq]
    received = mean_attn.mean(dim=0).detach().float().cpu().numpy()  # token-level received attention
    seq_len = received.shape[0]
    positions = token_positions(seq_len)
    return positions, received


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


def analyze(records: list[dict[str, str]], model: str | None, budget_tokens: int) -> tuple[AnalysisSummary, dict[str, Any]]:
    all_positions: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    per_block_stats = []

    llm = tokenizer = None
    if model:
        llm, tokenizer = load_model_and_tokenizer(model)

    for record in records:
        completion = record["completion"]
        blocks = split_thinkblocks(completion)
        for block in blocks:
            if llm is not None and tokenizer is not None:
                positions, scores = extract_attention_profile(llm, tokenizer, record["prompt"], block)
            else:
                tokens = block.split()
                positions = token_positions(len(tokens))
                scores = np.array([position_score_proxy(float(p)) for p in positions], dtype=float)
            all_positions.append(positions)
            all_scores.append(scores)
            per_block_stats.append((len(scores), float(scores[0]) if len(scores) else 0.0, float(scores[len(scores) // 2]) if len(scores) else 0.0, float(scores[-1]) if len(scores) else 0.0))

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
    for positions, scores in zip(all_positions, all_scores):
        n = len(scores)
        if n == 0:
            continue
        mask = scheduler_keep_mask(scores, min(budget_tokens, n))
        scheduler_scores.append(float(scores[mask].sum() / scores.sum()))
        first_last_scores.append(float(scores[first_last_keep(n, min(budget_tokens, n))].sum() / scores.sum()))
        uniform_scores.append(float(scores[uniform_keep(n, min(budget_tokens, n))].sum() / scores.sum()))
        best_window_scores.append(float(scores[contiguous_window_keep(scores, min(budget_tokens, n))].sum() / scores.sum()))

    position_profile = build_position_summary(all_positions, all_scores)
    summary = AnalysisSummary(
        model=model,
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
        position_profile=position_profile,
    )

    plot_payload = {
        "positions": all_positions,
        "scores": all_scores,
        "summary": asdict(summary),
    }
    return summary, plot_payload


def plot_results(payload: dict[str, Any], output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    positions = payload["positions"]
    scores = payload["scores"]
    flat_pos = np.concatenate(positions) if positions else np.array([])
    flat_scores = np.concatenate(scores) if scores else np.array([])

    plt.style.use("bmh")
    plt.rcParams["font.size"] = 13

    # 1) position profile
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
    ax.set_xlabel("Relative token position inside thinkblock")
    ax.set_ylabel("Mean attention score")
    ax.set_title("Thinkblock attention profile")
    fig.tight_layout()
    fig.savefig(output / "attention_position_profile.png", dpi=200)
    plt.close(fig)

    # 2) heatmap by block position
    max_len = max((len(s) for s in scores), default=0)
    heatmap = np.full((len(scores), max_len), np.nan)
    for i, s in enumerate(scores):
        heatmap[i, : len(s)] = s
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Token index in thinkblock")
    ax.set_ylabel("Block index")
    ax.set_title("Per-block attention heatmap")
    fig.colorbar(im, ax=ax, label="Attention score")
    fig.tight_layout()
    fig.savefig(output / "block_attention_heatmap.png", dpi=200)
    plt.close(fig)

    # 3) scheduler comparison
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


def default_prompts() -> list[str]:
    return [
        "You are a careful reasoning assistant. Solve the task step by step, then give the final answer at the end. What is 17 * 24?",
        "You are a careful reasoning assistant. Explain briefly how to derive the area of a circle, then compute it for radius 7.",
        "You are a careful reasoning assistant. If a train travels 180 km in 3 hours, what is its speed?",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify thinkblock attention concentration at start/end positions")
    parser.add_argument("--input-json", type=str, default=None, help="JSON file containing records with prompt/completion")
    parser.add_argument("--model", type=str, default=None, help="Optional local or Hugging Face model path for real attention extraction")
    parser.add_argument("--output-dir", type=str, default="thinkblock_attention_results")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--budget-tokens", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.input_json)
    if not records:
        prompts = default_prompts()[: args.num_prompts]
        if args.model:
            records = generate_records(args.model, prompts, args.max_tokens, args.temperature, args.top_p)
        else:
            # Fallback mode: use synthetic completions to validate the analysis pipeline.
            records = [
                {"prompt": p, "completion": "First, identify the relevant quantities. Then compute the intermediate value. Finally, summarize the answer clearly."}
                for p in prompts
            ]

    summary, payload = analyze(records, args.model, args.budget_tokens)
    plot_results(payload, args.output_dir)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    print(f"Saved plots to: {out.resolve()}")


if __name__ == "__main__":
    main()
