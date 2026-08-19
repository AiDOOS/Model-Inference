"""
Module 02 - Approach 1: No batching (the baseline).

Reuses module 00's approach: every request is processed FULLY - its whole
prefill, then its whole decode loop, all the way to its own EOS or the
shared MAX_NEW_TOKENS cap - before the next request even starts. One
sequence occupies the CPU at a time. This is the "one cashier serving one
customer completely before calling the next" model from the README.

There's no scheduler here worth calling one: process request 0, then
request 1, then request 2, ... in arrival order. Whatever gets submitted
first finishes first, and everyone behind it just waits for their turn -
regardless of whether their own request is short or long.
"""

import argparse
import time

import torch

from cpu_utilization import CpuUtilizationSampler
from workload import (
    MAX_NEW_TOKENS,
    build_input_ids,
    build_workload,
    eos_ids_for,
    load_model,
)


def run_one_request(model, tokenizer, eos_ids, req, max_new_tokens, wall_start):
    input_ids = build_input_ids(tokenizer, req["prompt"])
    prompt_tokens = input_ids.shape[1]

    t0 = time.perf_counter()

    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
    past_key_values = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
    generated = [next_token.item()]

    cur_token = next_token
    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos_ids:
            break
        with torch.inference_mode():
            out = model(cur_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        cur_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated.append(cur_token.item())

    own_compute_s = time.perf_counter() - t0
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "request_id": req["request_id"],
        "category": req["category"],
        "prompt_tokens": prompt_tokens,
        "output_tokens": len(generated),
        # end-to-end: time since the WHOLE workload was submitted, not just
        # this request's own turn - everyone strictly ahead of it in the
        # queue is part of what it actually waited through.
        "latency_s": time.perf_counter() - wall_start,
        "own_compute_s": own_compute_s,
        "text": text,
    }


def run_no_batching(max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    tokenizer, model = load_model()
    eos_ids = eos_ids_for(tokenizer, model)
    workload = build_workload()

    sampler = CpuUtilizationSampler(interval_s=0.5)
    sampler.start()
    wall_start = time.perf_counter()

    results = []
    for req in workload:
        results.append(run_one_request(model, tokenizer, eos_ids, req, max_new_tokens, wall_start))

    total_time_s = time.perf_counter() - wall_start
    cpu_samples = sampler.stop()

    total_output_tokens = sum(r["output_tokens"] for r in results)

    return {
        "approach": "no_batching",
        "total_time_s": total_time_s,
        "throughput_tok_s": total_output_tokens / total_time_s,
        "avg_cpu_percent": CpuUtilizationSampler.average(cpu_samples),
        "cpu_samples": cpu_samples,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    print("[no_batching] loading model + workload ...", flush=True)
    summary = run_no_batching(args.max_new_tokens)

    print(f"[no_batching] total_time_s = {summary['total_time_s']:.2f}s")
    print(f"[no_batching] throughput_tok_s = {summary['throughput_tok_s']:.2f}")
    print(f"[no_batching] avg_cpu_percent = {summary['avg_cpu_percent']:.1f}%")

    short_latencies = [r["latency_s"] for r in summary["results"] if r["category"] == "short"]
    long_latencies = [r["latency_s"] for r in summary["results"] if r["category"] == "long"]
    print(f"[no_batching] short-request avg latency = {sum(short_latencies)/len(short_latencies):.2f}s")
    print(f"[no_batching] long-request avg latency  = {sum(long_latencies)/len(long_latencies):.2f}s")

    for r in summary["results"]:
        print(f"  [{r['request_id']:2d}] {r['category']:5s} "
              f"prompt_tokens={r['prompt_tokens']:3d} output_tokens={r['output_tokens']:3d} "
              f"latency={r['latency_s']:.3f}s")


if __name__ == "__main__":
    main()
