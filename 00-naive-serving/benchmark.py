"""
Module 00 - Load test for the naive server.

Fires N concurrent requests at the server at each concurrency level and
aggregates the phase-level timing server.py returns per request (prefill,
per-token decode, queue wait) alongside whole-batch wall-clock throughput.

What this is expected to show:
  - avg prefill time and avg decode time PER TOKEN (both server-side compute,
    measured while holding the model lock) stay roughly flat across
    concurrency levels - each request still gets the CPU to itself while it
    runs, the lock guarantees that.
  - avg queue wait and avg end-to-end latency grow roughly linearly with
    concurrency - requests pile up waiting for the one model.
  - throughput (total output tokens / batch wall time) stays roughly flat
    too, instead of scaling up with concurrency. That flatness IS the
    collapse: adding concurrent users buys you nothing but worse latency,
    because nothing is happening in parallel on the model itself.
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEFAULT_PROMPT = (
    "Explain, in a few sentences, why the sky appears blue during the day "
    "and orange or red at sunset. Keep the explanation accessible to a "
    "curious 10-year-old."
)


def one_request(base_url: str, prompt: str, max_new_tokens: int) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/generate",
        json={"prompt": prompt, "max_new_tokens": max_new_tokens},
        timeout=600,
    )
    client_wall_time_s = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    data["client_wall_time_s"] = client_wall_time_s
    return data


def run_level(base_url: str, concurrency: int, prompt: str, max_new_tokens: int):
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_request, base_url, prompt, max_new_tokens)
            for _ in range(concurrency)
        ]
        results = [fut.result() for fut in as_completed(futures)]
    batch_wall_time_s = time.perf_counter() - batch_start
    return results, batch_wall_time_s


def summarize(concurrency: int, results: list, batch_wall_time_s: float) -> dict:
    prefill_times = [r["prefill_time_s"] for r in results]
    all_decode_steps = [t for r in results for t in r["decode_times_s"]]
    queue_waits = [r["queue_wait_s"] for r in results]
    e2e_times = [r["client_wall_time_s"] for r in results]
    total_output_tokens = sum(r["output_tokens"] for r in results)

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "batch_wall_time_s": batch_wall_time_s,
        "throughput_tok_s": total_output_tokens / batch_wall_time_s,
        "avg_prefill_s": statistics.mean(prefill_times),
        "avg_decode_per_token_s": (
            statistics.mean(all_decode_steps) if all_decode_steps else float("nan")
        ),
        "avg_queue_wait_s": statistics.mean(queue_waits),
        "avg_e2e_latency_s": statistics.mean(e2e_times),
        "total_output_tokens": total_output_tokens,
    }


def print_table(rows: list) -> None:
    headers = [
        "Conc", "Reqs", "Wall(s)", "Tput(tok/s)",
        "AvgPrefill(s)", "AvgDecode/tok(s)", "AvgQueueWait(s)", "AvgE2E(s)",
    ]
    col_w = [6, 5, 8, 12, 14, 17, 16, 10]
    fmt = "  ".join(f"{{:>{w}}}" for w in col_w)

    print(fmt.format(*headers))
    print("-" * (sum(col_w) + 2 * (len(col_w) - 1)))
    for row in rows:
        print(fmt.format(
            row["concurrency"],
            row["requests"],
            f"{row['batch_wall_time_s']:.2f}",
            f"{row['throughput_tok_s']:.2f}",
            f"{row['avg_prefill_s']:.4f}",
            f"{row['avg_decode_per_token_s']:.4f}",
            f"{row['avg_queue_wait_s']:.4f}",
            f"{row['avg_e2e_latency_s']:.2f}",
        ))
    print()
    print("Prefill/Decode = server-side compute time, held while it owns the model lock.")
    print("QueueWait/E2E  = client-visible cost of everyone else queueing ahead of you.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    print("[benchmark] warming up (first call pays model/thread-pool warm-up cost) ...")
    one_request(args.base_url, args.prompt, args.max_new_tokens)

    rows = []
    for level in args.levels:
        print(f"[benchmark] concurrency={level} ...")
        results, batch_wall_time_s = run_level(
            args.base_url, level, args.prompt, args.max_new_tokens
        )
        rows.append(summarize(level, results, batch_wall_time_s))

    print()
    print_table(rows)


if __name__ == "__main__":
    main()
