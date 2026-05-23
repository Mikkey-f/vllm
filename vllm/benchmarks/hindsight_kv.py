# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone HindsightKV v1.0 benchmark.

This version does not import the local vLLM source tree. It uses Hugging Face
Transformers directly so it can run even when the cloned vLLM checkout cannot be
built in editable mode.

It also supports baseline retention policies so the same generated completions
can be re-analyzed under multiple heuristics.
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
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    prompt_len: int
    completion_len: int
    block_count: int
    retained_tokens: int
    total_reasoning_tokens: int
    compression_ratio: float
    block_analyses: list[BlockAnalysis]


@dataclass
class BenchmarkSummary:
    model: str
    baseline: str
    num_requests: int
    elapsed_sec: float
    tokens_generated: int
    avg_completion_len: float
    avg_block_count: float
    avg_retention_ratio: float
    avg_compression_ratio: float
    median_compression_ratio: float
    p95_compression_ratio: float
    request_results: list[RequestAnalysis]


@dataclass
class BaselineSummary:
    baseline: str
    avg_retention_ratio: float
    avg_compression_ratio: float
    median_compression_ratio: float
    p95_compression_ratio: float


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


def analyze_block(block_index: int, block_text: str, max_keep: int, baseline: str) -> BlockAnalysis:
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
    prompt_len: int,
    completion_text: str,
    block_token_budget: int,
    baseline: str,
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
        prompt_len=prompt_len,
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


def run_generation(
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_requests: int,
) -> tuple[list[str], float, int, float, float, int]:
    llm, tokenizer = load_model_and_tokenizer(model)
    base_prompts = build_prompts()
    prompts = [base_prompts[i % len(base_prompts)] for i in range(num_requests)]

    outputs: list[str] = []
    start = time.perf_counter()

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
        outputs.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

    elapsed = time.perf_counter() - start
    tokens_generated = sum(len(o.split()) for o in outputs)
    avg_completion_len = statistics.fmean(len(o.split()) for o in outputs) if outputs else 0.0
    return outputs, elapsed, tokens_generated, avg_completion_len, 0.0, num_requests


def summarize_results(baseline: str, request_results: list[RequestAnalysis]) -> BaselineSummary:
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


def run_benchmark(
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    block_token_budget: int,
    num_requests: int,
    output_json: str | None,
    baseline: str,
) -> BenchmarkSummary:
    llm, tokenizer = load_model_and_tokenizer(model)
    base_prompts = build_prompts()
    prompts = [base_prompts[i % len(base_prompts)] for i in range(num_requests)]

    request_results: list[RequestAnalysis] = []
    outputs: list[str] = []
    start = time.perf_counter()

    for idx, prompt in enumerate(prompts):
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
        request_results.append(
            analyze_completion(
                request_id=idx,
                prompt_len=prompt_len,
                completion_text=completion_text,
                block_token_budget=block_token_budget,
                baseline=baseline,
            )
        )

    elapsed = time.perf_counter() - start
    tokens_generated = sum(r.completion_len for r in request_results)
    avg_completion_len = statistics.fmean(r.completion_len for r in request_results) if request_results else 0.0
    avg_block_count = statistics.fmean(r.block_count for r in request_results) if request_results else 0.0
    avg_retention_ratio = statistics.fmean(
        (r.retained_tokens / r.total_reasoning_tokens) if r.total_reasoning_tokens else 0.0
        for r in request_results
    ) if request_results else 0.0
    compression_ratios = [r.compression_ratio for r in request_results if math.isfinite(r.compression_ratio)]
    avg_compression_ratio = statistics.fmean(compression_ratios) if compression_ratios else math.inf
    median_compression_ratio = statistics.median(compression_ratios) if compression_ratios else math.inf
    p95_compression_ratio = (
        sorted(compression_ratios)[
            max(0, min(len(compression_ratios) - 1, int(0.95 * len(compression_ratios)) - 1))
        ]
        if compression_ratios
        else math.inf
    )

    summary = BenchmarkSummary(
        model=model,
        baseline=baseline,
        num_requests=num_requests,
        elapsed_sec=elapsed,
        tokens_generated=tokens_generated,
        avg_completion_len=avg_completion_len,
        avg_block_count=avg_block_count,
        avg_retention_ratio=avg_retention_ratio,
        avg_compression_ratio=avg_compression_ratio,
        median_compression_ratio=median_compression_ratio,
        p95_compression_ratio=p95_compression_ratio,
        request_results=request_results,
    )

    print(f"Model: {model}")
    print(f"Baseline: {baseline}")
    print(f"Requests: {num_requests}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Generated tokens: {tokens_generated}")
    print(f"Avg completion length: {avg_completion_len:.2f}")
    print(f"Avg reasoning blocks: {avg_block_count:.2f}")
    print(f"Avg retention ratio: {avg_retention_ratio:.3f}")
    print(f"Avg compression ratio: {avg_compression_ratio:.3f}")
    print(f"Median compression ratio: {median_compression_ratio:.3f}")
    print(f"P95 compression ratio: {p95_compression_ratio:.3f}")

    if output_json:
        payload: dict[str, Any] = asdict(summary)
        payload["request_results"] = [asdict(r) for r in request_results]
        payload["generated_outputs"] = outputs
        payload["baseline_summary"] = asdict(summarize_results(baseline, request_results))
        Path(output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HindsightKV v1.0 benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--block-token-budget", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=3)
    parser.add_argument("--baseline", type=str, default="hindsight", choices=["hindsight", "random", "recency", "first_last", "uniform_topk"])
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        block_token_budget=args.block_token_budget,
        num_requests=args.num_requests,
        output_json=args.output_json,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()
