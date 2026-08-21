"""
Module 03 - Benchmark: naive allocation vs. manual paged attention vs.
vLLM, all three admitting from the exact same fixed 24-request workload
against the exact same fixed MEMORY_BUDGET_BYTES.

The single number this module exists to produce is "max concurrent
sequences fit" - if manual paging and vLLM don't both fit dramatically
more concurrent sequences than naive allocation under the SAME memory
budget, something's wrong with the run, not the theory.
"""

import argparse
import json
import os
import time

from kv_cache import MAX_NEW_TOKENS, MEMORY_BUDGET_BYTES
from manual_paged_attention import run_manual_paged_attention
from naive_allocation import run_naive_allocation
from vllm_paged import check_vllm_available, run_vllm_paged

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def print_table(rows):
    headers = ["Approach", "MaxConcurFit", "Waste/Frag(%)", "Total(s)", "Tput(tok/s)"]
    col_w = [24, 12, 14, 10, 12]
    fmt = "  ".join(f"{{:>{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-" * (sum(col_w) + 2 * (len(col_w) - 1)))
    for row in rows:
        print(fmt.format(
            row["approach"],
            row["max_concurrent_fit"] if row["max_concurrent_fit"] is not None else "N/A",
            f"{row['waste_percent']:.1f}" if row["waste_percent"] is not None else "N/A",
            f"{row['total_time_s']:.2f}" if row["total_time_s"] is not None else "N/A",
            f"{row['throughput_tok_s']:.2f}" if row["throughput_tok_s"] is not None else "N/A",
        ))
    print()
    print("MaxConcurFit = highest number of sequences ever concurrently admitted under the")
    print("SAME fixed memory budget. Waste/Frag(%) = naive: share of every reserved token-slot")
    print("that was never actually used; paged/vLLM: internal fragmentation only (the much")
    print("smaller, expected kind of waste - see README Gotcha).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    rows = []
    raw_summaries = {}

    print(f"[benchmark] memory budget = {MEMORY_BUDGET_BYTES / 1024 / 1024:.0f} MiB "
          f"(same for every approach below)")

    print("[benchmark] === naive_allocation ===", flush=True)
    naive = run_naive_allocation(args.max_new_tokens)
    raw_summaries["naive_allocation"] = naive
    rows.append({
        "approach": "Naive allocation",
        "max_concurrent_fit": naive["peak_concurrent"],
        "waste_percent": naive["waste_percent"],
        "total_time_s": naive["total_time_s"],
        "throughput_tok_s": naive["throughput_tok_s"],
    })

    print("[benchmark] === manual_paged_attention ===", flush=True)
    paged = run_manual_paged_attention(args.max_new_tokens)
    raw_summaries["manual_paged_attention"] = paged
    rows.append({
        "approach": "Manual paged",
        "max_concurrent_fit": paged["peak_concurrent"],
        "waste_percent": paged["internal_fragmentation_percent"],
        "total_time_s": paged["total_time_s"],
        "throughput_tok_s": paged["throughput_tok_s"],
    })

    print("[benchmark] === vllm_paged ===", flush=True)
    vllm_available, vllm_reason = check_vllm_available()
    if vllm_available:
        vb = run_vllm_paged(args.max_new_tokens)
        raw_summaries["vllm_paged"] = vb
        rows.append({
            "approach": "vLLM",
            "max_concurrent_fit": vb.get("total_blocks"),
            "waste_percent": None,
            "total_time_s": vb["total_time_s"],
            "throughput_tok_s": vb["throughput_tok_s"],
        })
    else:
        print(f"[benchmark] vLLM SKIPPED: {vllm_reason}")
        raw_summaries["vllm_paged"] = {"skipped": True, "reason": vllm_reason}
        rows.append({
            "approach": "vLLM",
            "max_concurrent_fit": None,
            "waste_percent": None,
            "total_time_s": None,
            "throughput_tok_s": None,
        })

    print()
    print_table(rows)
    if not vllm_available:
        print()
        print(f"vLLM row is N/A: {vllm_reason}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"paged_attention_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "memory_budget_bytes": MEMORY_BUDGET_BYTES,
            "max_new_tokens": args.max_new_tokens,
            "table": rows,
            "raw": raw_summaries,
        }, f, indent=2)
    print(f"\n[benchmark] full results saved to {out_path}")


if __name__ == "__main__":
    main()
