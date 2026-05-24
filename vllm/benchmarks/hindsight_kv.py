# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone HindsightKV benchmark.

Three-phase workflow:
  A. Generate once and persist outputs.
  B. Re-analyze the same outputs under multiple baseline policies.
  C. Emit lightweight runtime/analysis overhead statistics for future serving work.

This script intentionally does not import the local vLLM source tree.
It uses Hugging Face Transformers directly so the benchmark can run even when
editable installation of the repository is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass

from tqdm import tqdm
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BaselineName = Literal["hindsight", "random", "recency", "first_last", "uniform_topk"]
ModeName = Literal["generate", "analyze", "compare", "full"]

_REASONING_SPLIT_RE = re.compile(
    r"(?:\n\s*\n|\n(?:step\s*\d+[:.)-]?|therefore[:.,]?|thus[:.,]?|hence[:.,]?|next[:.,]?|finally[:.,]?)\s*)",
    re.IGNORECASE,
)


@dataclass
class BlockAnalysis:
    block_index: int
    token_count: int
    retained_count: int
    retention_ratio: float
    score_mean: float
    score_max: float


@dataclass
class RequestAnalysis:
    request_id: int
    prompt: str
    prompt_len: int
    completion_len: int
    block_count: int
    retained_tokens: int
    total_reasoning_tokens: int
    compression_ratio: float
    block_analyses: list[BlockAnalysis]


@dataclass
class GenerationSummary:
    model: str
    num_requests: int
    elapsed_sec: float
    tokens_generated: int
    avg_completion_len: float
    avg_block_count: float
    generated_outputs: list[str]


@dataclass
class BaselineSummary:
    baseline: str
    avg_retention_ratio: float
    avg_compression_ratio: float
    median_compression_ratio: float
    p95_compression_ratio: float


@dataclass
class CompareSummary:
    baselines: list[BaselineSummary]


@dataclass
class PhaseCStats:
    generation_elapsed_sec: float
    analysis_elapsed_sec: float
    compare_elapsed_sec: float
    generation_throughput_tok_s: float
    num_requests: int
    num_outputs: int


@dataclass
class FullResult:
    model: str
    mode: str
    generation: GenerationSummary
    analyses: dict[str, list[RequestAnalysis]]
    compare: CompareSummary | None
    phase_c: PhaseCStats


def split_reasoning_blocks(text: str) -> list[str]:
    blocks = [b.strip() for b in _REASONING_SPLIT_RE.split(text) if b and b.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def token_score(token: str, position: int, block_len: int) -> float:
    lower = token.lower()
    score = 1.0 + (position / max(1, block_len - 1))
    if any(ch.isdigit() for ch in token):
        score += 0.8
    if any(ch in token for ch in ("=", ":", ";", "(", ")", ",")):
        score += 0.3
    if len(token) >= 8:
        score += 0.15
    if lower in {"the", "and", "of", "to", "a", "in", "is", "for", "we", "i"}:
        score -= 0.6
    return score


def analyze_block(block_index: int, block_text: str, max_keep: int, baseline: BaselineName) -> BlockAnalysis:
    tokens = block_text.split()
    if not tokens:
        return BlockAnalysis(block_index, 0, 0, 0.0, 0.0, 0.0)

    retained_count = min(max_keep, len(tokens))

    if baseline == "hindsight":
        scores = [token_score(tok, idx, len(tokens)) for idx, tok in enumerate(tokens)]
    elif baseline == "random":
        scores = [random.random() for _ in tokens]
    elif baseline == "recency":
        scores = [float(idx >= len(tokens) - retained_count) for idx in range(len(tokens))]
    elif baseline == "first_last":
        scores = [0.0 for _ in tokens]
        first_keep = min(retained_count // 2, len(tokens))
        last_keep = min(retained_count - first_keep, len(tokens) - first_keep)
        for idx in range(first_keep):
            scores[idx] = 1.0
        for idx in range(len(tokens) - last_keep, len(tokens)):
            scores[idx] = 1.0
    elif baseline == "uniform_topk":
        scores = [1.0 + (idx / max(1, len(tokens) - 1)) * 0.01 for idx in range(len(tokens))]
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    return BlockAnalysis(
        block_index=block_index,
        token_count=len(tokens),
        retained_count=retained_count,
        retention_ratio=retained_count / len(tokens),
        score_mean=statistics.fmean(scores) if scores else 0.0,
        score_max=max(scores) if scores else 0.0,
    )


def analyze_completion(
    request_id: int,
    prompt: str,
    completion_text: str,
    block_token_budget: int,
    baseline: BaselineName,
) -> RequestAnalysis:
    blocks = split_reasoning_blocks(completion_text)
    block_analyses = [
        analyze_block(i, block, block_token_budget, baseline)
        for i, block in enumerate(blocks)
    ]
    total_reasoning_tokens = sum(b.token_count for b in block_analyses)
    retained_tokens = sum(b.retained_count for b in block_analyses)
    compression_ratio = total_reasoning_tokens / retained_tokens if retained_tokens else math.inf
    return RequestAnalysis(
        request_id=request_id,
        prompt=prompt,
        prompt_len=len(prompt.split()),
        completion_len=len(completion_text.split()),
        block_count=len(blocks),
        retained_tokens=retained_tokens,
        total_reasoning_tokens=total_reasoning_tokens,
        compression_ratio=compression_ratio,
        block_analyses=block_analyses,
    )


def build_prompts() -> list[str]:
    return [
        "You are a careful reasoning assistant. Solve the task step by step, then give the final answer at the end. What is 17 * 24?",
        "You are a careful reasoning assistant. Explain briefly how to derive the area of a circle, then compute it for radius 7.",
        "You are a careful reasoning assistant. If a train travels 180 km in 3 hours, what is its speed?",
    ]


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


def generate_outputs(
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_requests: int,
) -> tuple[list[dict[str, str]], GenerationSummary, float]:
    llm, tokenizer = load_model_and_tokenizer(model)
    base_prompts = build_prompts()
    prompts = [base_prompts[i % len(base_prompts)] for i in range(num_requests)]

    records: list[dict[str, str]] = []
    outputs: list[str] = []
    start = time.perf_counter()

    for prompt in tqdm(prompts, desc="Generating", unit="req"):
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
        outputs.append(completion_text)
        records.append({"prompt": prompt, "completion": completion_text})

    elapsed = time.perf_counter() - start
    tokens_generated = sum(len(item["completion"].split()) for item in records)
    avg_completion_len = statistics.fmean(len(item["completion"].split()) for item in records) if records else 0.0
    avg_block_count = statistics.fmean(len(split_reasoning_blocks(item["completion"])) for item in records) if records else 0.0
    summary = GenerationSummary(
        model=model,
        num_requests=num_requests,
        elapsed_sec=elapsed,
        tokens_generated=tokens_generated,
        avg_completion_len=avg_completion_len,
        avg_block_count=avg_block_count,
        generated_outputs=outputs,
    )
    return records, summary, elapsed


def analyze_records(
    records: list[dict[str, str]],
    block_token_budget: int,
    baselines: list[BaselineName],
) -> tuple[dict[str, list[RequestAnalysis]], CompareSummary, float]:
    analyses: dict[str, list[RequestAnalysis]] = {}
    summaries: list[BaselineSummary] = []
    start = time.perf_counter()

    for baseline in tqdm(baselines, desc="Analyzing baselines", unit="baseline"):
        request_results = [
            analyze_completion(
                request_id=idx,
                prompt=record["prompt"],
                completion_text=record["completion"],
                block_token_budget=block_token_budget,
                baseline=baseline,
            )
            for idx, record in enumerate(records)
        ]
        analyses[baseline] = request_results
        summaries.append(summarize_baseline(baseline, request_results))

    elapsed = time.perf_counter() - start
    return analyses, CompareSummary(baselines=summaries), elapsed


def summarize_baseline(baseline: BaselineName, request_results: list[RequestAnalysis]) -> BaselineSummary:
    retention_ratios = [
        (r.retained_tokens / r.total_reasoning_tokens) if r.total_reasoning_tokens else 0.0
        for r in request_results
    ]
    compression_ratios = [
        r.compression_ratio for r in request_results if math.isfinite(r.compression_ratio)
    ]
    p95 = (
        sorted(compression_ratios)[
            max(0, min(len(compression_ratios) - 1, int(0.95 * len(compression_ratios)) - 1))
        ]
        if compression_ratios
        else math.inf
    )
    return BaselineSummary(
        baseline=baseline,
        avg_retention_ratio=statistics.fmean(retention_ratios) if retention_ratios else 0.0,
        avg_compression_ratio=statistics.fmean(compression_ratios) if compression_ratios else math.inf,
        median_compression_ratio=statistics.median(compression_ratios) if compression_ratios else math.inf,
        p95_compression_ratio=p95,
    )


def print_generation_summary(summary: GenerationSummary) -> None:
    print(f"Model: {summary.model}")
    print(f"Requests: {summary.num_requests}")
    print(f"Elapsed: {summary.elapsed_sec:.2f}s")
    print(f"Generated tokens: {summary.tokens_generated}")
    print(f"Avg completion length: {summary.avg_completion_len:.2f}")
    print(f"Avg reasoning blocks: {summary.avg_block_count:.2f}")


def print_compare_summary(compare: CompareSummary) -> None:
    for item in compare.baselines:
        print(f"Baseline: {item.baseline}")
        print(f"  Avg retention ratio: {item.avg_retention_ratio:.3f}")
        print(f"  Avg compression ratio: {item.avg_compression_ratio:.3f}")
        print(f"  Median compression ratio: {item.median_compression_ratio:.3f}")
        print(f"  P95 compression ratio: {item.p95_compression_ratio:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HindsightKV benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--block-token-budget", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=3)
    parser.add_argument("--mode", type=str, default="full", choices=["generate", "analyze", "compare", "full"])
    parser.add_argument("--baseline", type=str, default="hindsight", choices=["hindsight", "random", "recency", "first_last", "uniform_topk"])
    parser.add_argument("--compare-baselines", type=str, default="hindsight,random,recency,first_last,uniform_topk")
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = [b.strip() for b in args.compare_baselines.split(",") if b.strip()]
    baselines = [b for b in baselines if b in {"hindsight", "random", "recency", "first_last", "uniform_topk"}]

    generation_summary: GenerationSummary | None = None
    records: list[dict[str, str]]
    generation_elapsed = 0.0

    if args.mode in {"generate", "full"}:
        records, generation_summary, generation_elapsed = generate_outputs(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_requests=args.num_requests,
        )
        print_generation_summary(generation_summary)
        print(f"Generation mode: {args.mode}")
        if args.output_json:
            payload = {
                "model": args.model,
                "mode": args.mode,
                "generation": asdict(generation_summary),
                "records": records,
            }
            Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        if not args.input_json:
            raise ValueError("--input-json is required for analyze/compare modes")
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        records = payload["records"]
        if "generation" in payload:
            generation_summary = GenerationSummary(**payload["generation"])

    analyses: dict[str, list[RequestAnalysis]] = {}
    compare_summary: CompareSummary | None = None
    analysis_elapsed = 0.0
    compare_elapsed = 0.0

    if args.mode in {"analyze", "compare", "full"}:
        if args.mode == "analyze":
            selected = [args.baseline]
        else:
            selected = baselines
        analyses, compare_summary, compare_elapsed = analyze_records(
            records=records,
            block_token_budget=args.block_token_budget,
            baselines=selected,  # type: ignore[arg-type]
        )
        analysis_elapsed = compare_elapsed
        print_compare_summary(compare_summary)

    if args.mode == "full":
        generation_elapsed = generation_summary.elapsed_sec if generation_summary else 0.0

    phase_c = PhaseCStats(
        generation_elapsed_sec=generation_elapsed,
        analysis_elapsed_sec=analysis_elapsed,
        compare_elapsed_sec=compare_elapsed,
        generation_throughput_tok_s=(generation_summary.tokens_generated / generation_summary.elapsed_sec) if generation_summary and generation_summary.elapsed_sec > 0 else 0.0,
        num_requests=len(records),
        num_outputs=len(records),
    )

    result = FullResult(
        model=args.model,
        mode=args.mode,
        generation=generation_summary or GenerationSummary(
            model=args.model,
            num_requests=len(records),
            elapsed_sec=generation_elapsed,
            tokens_generated=sum(len(r["completion"].split()) for r in records),
            avg_completion_len=statistics.fmean(len(r["completion"].split()) for r in records) if records else 0.0,
            avg_block_count=statistics.fmean(len(split_reasoning_blocks(r["completion"])) for r in records) if records else 0.0,
            generated_outputs=[r["completion"] for r in records],
        ),
        analyses=analyses,
        compare=compare_summary,
        phase_c=phase_c,
    )

    if args.output_json and args.mode != "generate":
        Path(args.output_json).write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
