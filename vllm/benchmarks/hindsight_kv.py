# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prototype benchmark for HindsightKV-style reasoning-block compression."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt


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


def analyze_block(block_index: int, block_text: str, max_keep: int) -> BlockAnalysis:
    tokens = block_text.split()
    scores = [token_score(tok, idx, len(tokens)) for idx, tok in enumerate(tokens)]
    retained_count = min(max_keep, len(tokens))
    return BlockAnalysis(
        block_index=block_index,
        token_count=len(tokens),
        retained_count=retained_count,
        retention_ratio=(retained_count / len(tokens)) if tokens else 0.0,
        score_mean=statistics.fmean(scores) if scores else 0.0,
        score_max=max(scores) if scores else 0.0,
    )


def analyze_completion(request_id: int, prompt_len: int, completion_text: str, block_token_budget: int) -> RequestAnalysis:
    blocks = split_reasoning_blocks(completion_text)
    block_analyses = [analyze_block(i, block, block_token_budget) for i, block in enumerate(blocks)]
    total_reasoning_tokens = sum(b.token_count for b in block_analyses)
    retained_tokens = sum(b.retained_count for b in block_analyses)
    compression_ratio = (total_reasoning_tokens / retained_tokens) if retained_tokens else math.inf
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


def run_benchmark(model: str, max_tokens: int, temperature: float, top_p: float, block_token_budget: int, num_requests: int, output_json: str | None) -> BenchmarkSummary:
    llm = LLM(model=model)
    base_prompts = build_prompts()
    prompts = [TextPrompt(prompt=base_prompts[i % len(base_prompts)]) for i in range(num_requests)]
    sampling_params = [SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, ignore_eos=True, detokenize=True) for _ in range(num_requests)]

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.perf_counter() - start

    request_results: list[RequestAnalysis] = []
    for idx, output in enumerate(outputs):
        text = output.outputs[0].text if output.outputs else ""
        request_results.append(analyze_completion(idx, len(output.prompt_token_ids or []), text, block_token_budget))

    tokens_generated = sum(r.completion_len for r in request_results)
    avg_completion_len = statistics.fmean(r.completion_len for r in request_results)
    avg_block_count = statistics.fmean(r.block_count for r in request_results)
    avg_retention_ratio = statistics.fmean((r.retained_tokens / r.total_reasoning_tokens) if r.total_reasoning_tokens else 0.0 for r in request_results)
    compression_ratios = [r.compression_ratio for r in request_results if math.isfinite(r.compression_ratio)]
    avg_compression_ratio = statistics.fmean(compression_ratios) if compression_ratios else math.inf
    median_compression_ratio = statistics.median(compression_ratios) if compression_ratios else math.inf
    p95_compression_ratio = sorted(compression_ratios)[max(0, min(len(compression_ratios) - 1, int(0.95 * len(compression_ratios)) - 1))] if compression_ratios else math.inf

    summary = BenchmarkSummary(model=model, num_requests=num_requests, elapsed_sec=elapsed, tokens_generated=tokens_generated, avg_completion_len=avg_completion_len, avg_block_count=avg_block_count, avg_retention_ratio=avg_retention_ratio, avg_compression_ratio=avg_compression_ratio, median_compression_ratio=median_compression_ratio, p95_compression_ratio=p95_compression_ratio, request_results=request_results)

    print(f"Model: {model}")
    print(f"Requests: {num_requests}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Generated tokens: {tokens_generated}")
    print(f"Avg completion length: {avg_completion_len:.2f}")
    print(f"Avg reasoning blocks: {avg_block_count:.2f}")
    print(f"Avg retention ratio: {avg_retention_ratio:.3f}")
    print(f"Avg compression ratio: {avg_compression_ratio:.3f}")

    if output_json:
        payload = asdict(summary)
        payload["request_results"] = [asdict(r) for r in request_results]
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
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(args.model, args.max_tokens, args.temperature, args.top_p, args.block_token_budget, args.num_requests, args.output_json)


if __name__ == "__main__":
    main()
