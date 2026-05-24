# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick standalone verifier for the hypothesis:
"the tail block of a reasoning trace carries the highest information density".

This script does NOT depend on vLLM runtime. It uses Hugging Face Transformers
only, generates completions from a local model path, splits each completion into
reasoning blocks, and compares the final block against earlier blocks using a
few simple, heuristic information-density proxies.

The goal is a fast sanity check for the idea, not a proof.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_SPLIT_RE = re.compile(
    r"(?:\n\s*\n|\n(?:step\s*\d+[:.)-]?|therefore[:.,]?|thus[:.,]?|hence[:.,]?|next[:.,]?|finally[:.,]?)\s*)",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "is", "for", "we", "i", "it",
    "this", "that", "on", "with", "as", "are", "be", "by", "or", "an",
    "at", "from", "into", "then", "so", "but", "if", "not", "can", "will",
}


@dataclass
class BlockStats:
    block_index: int
    token_count: int
    content_word_count: int
    numeric_count: int
    symbol_count: int
    unique_token_count: int
    density: float


@dataclass
class SampleStats:
    sample_id: int
    prompt: str
    completion_len: int
    block_count: int
    tail_block_index: int
    tail_block_density: float
    best_block_index: int
    best_block_density: float
    tail_is_best: bool
    block_stats: list[BlockStats]


@dataclass
class Summary:
    model: str
    num_samples: int
    avg_blocks: float
    avg_completion_len: float
    tail_best_rate: float
    tail_rank_avg: float
    tail_density_avg: float
    best_density_avg: float
    elapsed_sec: float


def split_blocks(text: str) -> list[str]:
    blocks = [b.strip() for b in _SPLIT_RE.split(text) if b and b.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def block_density(block: str) -> BlockStats:
    tokens = [t for t in re.findall(r"\w+|[^\w\s]", block) if t.strip()]
    if not tokens:
        return BlockStats(0, 0, 0, 0, 0, 0.0)

    words = [t for t in tokens if re.match(r"\w+", t)]
    content_words = [w for w in words if w.lower() not in _STOPWORDS]
    numeric_count = sum(any(ch.isdigit() for ch in t) for t in tokens)
    symbol_count = sum(bool(re.match(r"[^\w\s]", t)) for t in tokens)
    unique_token_count = len(set(t.lower() for t in words))

    density = (
        1.0 * len(content_words)
        + 0.8 * numeric_count
        + 0.4 * symbol_count
        + 0.3 * unique_token_count
    ) / max(1, len(tokens))

    return BlockStats(
        block_index=0,
        token_count=len(tokens),
        content_word_count=len(content_words),
        numeric_count=numeric_count,
        symbol_count=symbol_count,
        unique_token_count=unique_token_count,
        density=density,
    )


def build_prompts() -> list[str]:
    return [
        "You are a careful reasoning assistant. Solve the task step by step, then give the final answer at the end. What is 17 * 24?",
        "You are a careful reasoning assistant. Explain briefly how to derive the area of a circle, then compute it for radius 7.",
        "You are a careful reasoning assistant. If a train travels 180 km in 3 hours, what is its speed?",
        "You are a careful reasoning assistant. Compare 12, 15, and 18 and explain which is largest.",
        "You are a careful reasoning assistant. If x + 3 = 11, what is x? Show the steps.",
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


def generate_completion(llm, tokenizer, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
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
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def analyze_completion(sample_id: int, prompt: str, completion: str) -> SampleStats:
    blocks = split_blocks(completion)
    stats: list[BlockStats] = []
    for idx, block in enumerate(blocks):
        bs = block_density(block)
        bs.block_index = idx
        stats.append(bs)

    densities = [b.density for b in stats]
    tail_idx = len(stats) - 1 if stats else 0
    tail_density = densities[tail_idx] if densities else 0.0
    best_idx = max(range(len(densities)), key=lambda i: densities[i]) if densities else 0
    best_density = densities[best_idx] if densities else 0.0
    tail_rank = sorted(densities, reverse=True).index(tail_density) + 1 if densities else 0

    return SampleStats(
        sample_id=sample_id,
        prompt=prompt,
        completion_len=len(completion.split()),
        block_count=len(blocks),
        tail_block_index=tail_idx,
        tail_block_density=tail_density,
        best_block_index=best_idx,
        best_block_density=best_density,
        tail_is_best=(tail_idx == best_idx),
        block_stats=stats,
    )


def summarize(samples: list[SampleStats], elapsed_sec: float, model: str) -> Summary:
    if not samples:
        return Summary(model, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, elapsed_sec)

    avg_blocks = statistics.fmean(s.block_count for s in samples)
    avg_completion_len = statistics.fmean(s.completion_len for s in samples)
    tail_best_rate = sum(1 for s in samples if s.tail_is_best) / len(samples)
    tail_rank_avg = statistics.fmean(
        (sorted((b.density for b in s.block_stats), reverse=True).index(s.tail_block_density) + 1)
        for s in samples
        if s.block_stats
    )
    tail_density_avg = statistics.fmean(s.tail_block_density for s in samples)
    best_density_avg = statistics.fmean(s.best_block_density for s in samples)
    return Summary(
        model=model,
        num_samples=len(samples),
        avg_blocks=avg_blocks,
        avg_completion_len=avg_completion_len,
        tail_best_rate=tail_best_rate,
        tail_rank_avg=tail_rank_avg,
        tail_density_avg=tail_density_avg,
        best_density_avg=best_density_avg,
        elapsed_sec=elapsed_sec,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify whether tail reasoning blocks are denser")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm, tokenizer = load_model_and_tokenizer(args.model)
    prompts = build_prompts()
    samples: list[SampleStats] = []
    start = time.perf_counter()

    for i, prompt in enumerate(tqdm(prompts[: args.num_samples], desc="Generating", unit="sample")):
        completion = generate_completion(
            llm=llm,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        samples.append(analyze_completion(i, prompt, completion))

    elapsed = time.perf_counter() - start
    summary = summarize(samples, elapsed, args.model)

    print(f"Model: {summary.model}")
    print(f"Samples: {summary.num_samples}")
    print(f"Avg blocks: {summary.avg_blocks:.2f}")
    print(f"Avg completion len: {summary.avg_completion_len:.2f}")
    print(f"Tail-best rate: {summary.tail_best_rate:.3f}")
    print(f"Avg tail density: {summary.tail_density_avg:.3f}")
    print(f"Avg best density: {summary.best_density_avg:.3f}")
    print(f"Avg tail rank (1=best): {summary.tail_rank_avg:.2f}")
    print(f"Elapsed: {summary.elapsed_sec:.2f}s")

    if args.output_json:
        payload: dict[str, Any] = {
            "summary": asdict(summary),
            "samples": [asdict(s) for s in samples],
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
