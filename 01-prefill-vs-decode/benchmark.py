"""
Module 01 - Benchmark: prefill scales with prompt length, decode doesn't
(much).

Runs the manual HF split (with_cache and no_cache, from
manual_prefill_decode.py) on a short (~50 token) and a long (~500 token)
prompt, plus vLLM (from vllm_prefill_decode.py) as a reference point if it's
importable in this environment. Same max_new_tokens for every run, so the
only thing that changes between the short and long rows is prompt length.

What this is expected to show:
  - prefill_time_s grows a lot from short -> long prompt, for BOTH manual
    variants (prefill's cost is "one pass over every prompt token at once",
    so it scales with prompt length, full stop).
  - with_cache decode/token stays roughly flat from short -> long prompt
    (each step is still just "one new token forward pass, attend over the
    cached KV" - the cache does more work as context grows, but it's cheap
    relative to prefill's O(n) full-sequence pass).
  - no_cache decode/token is dramatically worse on the long prompt, and
    grows noticeably step-to-step within a single run (each step re-runs
    the ENTIRE sequence so far, so cost tracks total sequence length, not
    just "one token").
"""

import argparse
import statistics

from manual_prefill_decode import (
    build_input_ids,
    eos_ids_for,
    load_model,
    run_no_cache,
    run_with_cache,
)
from vllm_prefill_decode import check_vllm_available, run_vllm

SHORT_PROMPT = (
    "Explain, in a few sentences, why the sky appears blue during the day "
    "and orange or red at sunset. Keep the explanation accessible to a "
    "curious 10-year-old."
)

LONG_PROMPT = (
    "You are helping someone understand how large-scale distributed systems "
    "stay consistent while remaining available under network partitions. "
    "Start from first principles. A distributed system is a collection of "
    "independent computers that appears to its users as a single coherent "
    "system. The moment you split state across more than one machine, you "
    "introduce the possibility that those machines disagree about the "
    "current state, if only for a brief window, because messages between "
    "them take a nonzero amount of time and can be delayed, reordered, or "
    "lost outright. The CAP theorem formalizes a consequence of this: when "
    "a network partition occurs, a system must choose between remaining "
    "consistent (every read sees the latest write) and remaining available "
    "(every request gets a response, even if it might be stale). It cannot "
    "guarantee both during the partition. Many real systems sidestep the "
    "binary framing by picking different tradeoffs for different "
    "operations, or by weakening consistency to something like "
    "eventual consistency, where replicas are allowed to diverge "
    "temporarily but are guaranteed to converge once communication is "
    "restored and no new writes arrive. Consensus protocols such as Paxos "
    "and Raft exist to let a cluster of machines agree on a single sequence "
    "of operations despite some machines failing or messages being "
    "delayed, by requiring a majority quorum to agree before a value is "
    "considered committed. This majority requirement is what allows the "
    "system to keep making progress even if a minority of nodes are down "
    "or unreachable, while guaranteeing that any two majorities must "
    "overlap in at least one node, which is what prevents the cluster from "
    "committing two different, conflicting values for the same slot. "
    "Replication strategies build on top of these primitives: leader-based "
    "replication routes all writes through a single elected leader who "
    "orders them and streams the resulting log to followers, while "
    "leaderless replication allows writes to any replica and relies on "
    "read-repair and quorum reads to reconcile divergence after the fact. "
    "Each approach trades off write latency, read latency, and the "
    "complexity of handling conflicting concurrent writes differently. "
    "Understanding which tradeoff a given system has made - and why - is "
    "usually the fastest way to predict how it will behave under real "
    "failure conditions, rather than memorizing the name of the protocol "
    "it uses. Now, in a few clear paragraphs, summarize the core tension "
    "these systems are all navigating, and why no design fully escapes it."
)


def summarize(label, prompt_tokens, result):
    decode = result["decode_times_s"]
    if decode:
        mean_s = statistics.mean(decode)
        first_s = decode[0]
        last_s = decode[-1]
    else:
        mean_s = first_s = last_s = float("nan")
    return {
        "label": label,
        "prompt_tokens": prompt_tokens,
        "prefill_s": result["prefill_time_s"],
        "decode_steps": len(decode),
        "decode_mean_s": mean_s,
        "decode_first_s": first_s,
        "decode_last_s": last_s,
    }


def print_table(rows, vllm_rows, vllm_available, vllm_reason):
    headers = [
        "PromptLen", "Approach", "Prefill(s)", "Steps",
        "AvgDecode/tok(s)", "FirstDecode(s)", "LastDecode(s)",
    ]
    col_w = [10, 14, 11, 6, 17, 15, 14]
    fmt = "  ".join(f"{{:>{w}}}" for w in col_w)

    print(fmt.format(*headers))
    print("-" * (sum(col_w) + 2 * (len(col_w) - 1)))
    for row in rows:
        print(fmt.format(
            row["prompt_tokens"],
            row["approach"],
            f"{row['prefill_s']:.4f}",
            row["decode_steps"],
            f"{row['decode_mean_s']:.4f}",
            f"{row['decode_first_s']:.4f}",
            f"{row['decode_last_s']:.4f}",
        ))
    print()

    print("vLLM reference:")
    if not vllm_available:
        print(f"  N/A - {vllm_reason}")
    else:
        for label, r in vllm_rows:
            ttft = f"{r['vllm_ttft_s']:.4f}s" if r["vllm_ttft_s"] is not None else "n/a"
            print(
                f"  {label}: prompt_tokens={r['prompt_tokens']}  "
                f"total={r['vllm_total_time_s']:.4f}s  ttft={ttft}"
            )
    print()
    print("Prefill scales with prompt length; with_cache decode/token roughly doesn't.")
    print("no_cache decode/token is the odd one: it re-runs the full sequence every")
    print("step, so it grows with prompt length AND with how many tokens are already")
    print("generated - proof that skipping the KV cache turns decode back into a")
    print("chain of prefills.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    print("[benchmark] loading model for manual HF runs ...", flush=True)
    tokenizer, model = load_model()
    eos_ids = eos_ids_for(tokenizer, model)

    prompts = [("short (~65 tok)", SHORT_PROMPT), ("long (~480 tok)", LONG_PROMPT)]

    rows = []
    for label, prompt in prompts:
        input_ids = build_input_ids(tokenizer, prompt)
        prompt_tokens = input_ids.shape[1]
        print(f"[benchmark] {label}: prompt_tokens={prompt_tokens}")

        print(f"[benchmark]   with_cache ...", flush=True)
        wc = run_with_cache(model, input_ids, args.max_new_tokens, eos_ids)
        row = summarize(label, prompt_tokens, wc)
        row["approach"] = "with_cache"
        rows.append(row)

        print(f"[benchmark]   no_cache ...", flush=True)
        nc = run_no_cache(model, input_ids, args.max_new_tokens, eos_ids)
        row = summarize(label, prompt_tokens, nc)
        row["approach"] = "no_cache"
        rows.append(row)

    vllm_available, vllm_reason = check_vllm_available()
    vllm_rows = []
    if vllm_available:
        for label, prompt in prompts:
            print(f"[benchmark] vLLM on {label} ...", flush=True)
            vllm_rows.append((label, run_vllm(prompt, args.max_new_tokens)))
    else:
        print(f"[benchmark] vLLM SKIPPED: {vllm_reason}")

    print()
    print_table(rows, vllm_rows, vllm_available, vllm_reason)


if __name__ == "__main__":
    main()
