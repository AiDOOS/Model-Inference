"""
Module 02 - Benchmark: no batching vs. static batching vs. continuous
batching (toy) vs. vLLM, on the exact same fixed 16-request workload.

Runs all four approaches back to back (each loads its own model instance
fresh, to keep timings clean and independent), prints one comparison
table, and saves everything - including the CPU utilization time-series
for each approach - to results/batching_<timestamp>.json.

The single number this module exists to produce is the short-request
latency row: the same short prompt, submitted alongside long ones, under
four different scheduling policies. If continuous batching and vLLM don't
both look dramatically better than static batching there, something's
wrong with the run, not the theory.
"""

import argparse
import json
import os
import statistics
import time

from continuous_batching_toy import run_continuous_batching_toy
from no_batching import run_no_batching
from static_batching import run_static_batching
from vllm_batching import check_vllm_available, run_vllm_batching
from workload import BATCH_SIZE, MAX_NEW_TOKENS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def category_latency(results, category):
    values = [r["latency_s"] for r in results if r["category"] == category]
    return statistics.mean(values) if values else float("nan")


def print_table(rows):
    headers = [
        "Approach", "Total(s)", "Tput(tok/s)", "AvgCPU(%)",
        "ShortLat(s)", "LongLat(s)",
    ]
    col_w = [24, 10, 12, 10, 12, 11]
    fmt = "  ".join(f"{{:>{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-" * (sum(col_w) + 2 * (len(col_w) - 1)))
    for row in rows:
        print(fmt.format(
            row["approach"],
            f"{row['total_time_s']:.2f}" if row["total_time_s"] is not None else "N/A",
            f"{row['throughput_tok_s']:.2f}" if row["throughput_tok_s"] is not None else "N/A",
            f"{row['avg_cpu_percent']:.1f}" if row.get("avg_cpu_percent") is not None else "N/A",
            f"{row['short_latency_s']:.2f}" if row["short_latency_s"] is not None else "N/A",
            f"{row['long_latency_s']:.2f}" if row["long_latency_s"] is not None else "N/A",
        ))
    print()
    print("ShortLat/LongLat = avg end-to-end client latency for that category, from")
    print("t=0 (when all 16 requests were submitted) to when that request's full result")
    print("came back. static_batching's ShortLat should be BAD - a short request there")
    print("waits for the whole batch, including its 2 long neighbors, to finish.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    rows = []
    raw_summaries = {}

    print("[benchmark] === no_batching ===", flush=True)
    nb = run_no_batching(args.max_new_tokens)
    raw_summaries["no_batching"] = nb
    rows.append({
        "approach": "No batching",
        "total_time_s": nb["total_time_s"],
        "throughput_tok_s": nb["throughput_tok_s"],
        "avg_cpu_percent": nb["avg_cpu_percent"],
        "short_latency_s": category_latency(nb["results"], "short"),
        "long_latency_s": category_latency(nb["results"], "long"),
    })

    print("[benchmark] === static_batching ===", flush=True)
    sb = run_static_batching(args.max_new_tokens, args.batch_size)
    raw_summaries["static_batching"] = sb
    rows.append({
        "approach": "Static batching",
        "total_time_s": sb["total_time_s"],
        "throughput_tok_s": sb["throughput_tok_s"],
        "avg_cpu_percent": sb["avg_cpu_percent"],
        "short_latency_s": category_latency(sb["results"], "short"),
        "long_latency_s": category_latency(sb["results"], "long"),
    })

    print("[benchmark] === continuous_batching_toy ===", flush=True)
    cb = run_continuous_batching_toy(args.max_new_tokens, args.batch_size)
    raw_summaries["continuous_batching_toy"] = cb
    rows.append({
        "approach": "Continuous (toy)",
        "total_time_s": cb["total_time_s"],
        "throughput_tok_s": cb["throughput_tok_s"],
        "avg_cpu_percent": cb["avg_cpu_percent"],
        "short_latency_s": category_latency(cb["results"], "short"),
        "long_latency_s": category_latency(cb["results"], "long"),
    })

    print("[benchmark] === vllm_batching ===", flush=True)
    vllm_available, vllm_reason = check_vllm_available()
    if vllm_available:
        vb = run_vllm_batching(args.max_new_tokens)
        raw_summaries["vllm_batching"] = vb
        rows.append({
            "approach": "vLLM",
            "total_time_s": vb["total_time_s"],
            "throughput_tok_s": vb["throughput_tok_s"],
            "avg_cpu_percent": None,
            "short_latency_s": category_latency(vb["results"], "short"),
            "long_latency_s": category_latency(vb["results"], "long"),
        })
    else:
        print(f"[benchmark] vLLM SKIPPED: {vllm_reason}")
        raw_summaries["vllm_batching"] = {"skipped": True, "reason": vllm_reason}
        rows.append({
            "approach": "vLLM",
            "total_time_s": None,
            "throughput_tok_s": None,
            "avg_cpu_percent": None,
            "short_latency_s": None,
            "long_latency_s": None,
        })

    print()
    print_table(rows)
    if not vllm_available:
        print()
        print(f"vLLM row is N/A: {vllm_reason}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"batching_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "table": rows,
            "raw": raw_summaries,
        }, f, indent=2)
    print(f"\n[benchmark] full results + CPU utilization time-series saved to {out_path}")


if __name__ == "__main__":
    main()
