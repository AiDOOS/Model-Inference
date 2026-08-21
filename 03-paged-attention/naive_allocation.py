"""
Module 03 - Approach 1: Naive KV-cache allocation (the baseline).

This is the "reserving an entire hotel floor for every guest, just in case
they invite 50 friends" approach from the README. The instant a request is
admitted, it reserves a CONTIGUOUS buffer sized for the worst case it could
ever need - MAX_SEQ_LEN_NAIVE tokens of KV cache - even though real prompts
in this workload need as little as 41 tokens and none of them exceed 242
(182-token prompt + 60 decode tokens). That reservation is held, unshared,
for the sequence's entire lifetime, and only released in full when it
finishes - whether it used 242 of its 256 reserved token-slots or 49.

This is exactly what a naive server has to do if it wants a single
contiguous cache buffer per request (the simplest thing that works, and
genuinely how early LLM-serving code worked): it doesn't know in advance
how long a generation will run, so it can't allocate less than the worst
case without risking a buffer overrun mid-generation.

The scheduler shape below (evict finished -> admit from queue -> run one
batched decode step) is the same admit/evict loop as module 02's
continuous_batching_toy.py. The ONLY thing that's different is what gates
admission: there, a fixed SLOT COUNT; here, a MEMORY BUDGET.
"""

import argparse
import time
from typing import Dict, List, Optional

import torch

from kv_cache import (
    MAX_ITERATIONS,
    MAX_NEW_TOKENS,
    MAX_SEQ_LEN_NAIVE,
    MEMORY_BUDGET_BYTES,
    build_input_ids,
    build_workload,
    eos_ids_for,
    load_model,
    measure_kv_bytes_per_token,
    run_batched_decode_step,
)


class NaiveMemoryPool:
    """Tracks how many bytes are currently reserved out of a fixed budget,
    where EVERY reservation is the same fixed worst-case size regardless of
    what the sequence actually turns out to need.

    There's no notion of "partial" reservation and no notion of blocks -
    a sequence either gets its full worst-case slice or it doesn't get
    admitted at all. That's the entire naive policy in one sentence."""

    def __init__(self, budget_bytes: int, max_seq_len: int, bytes_per_token: float):
        self.budget_bytes = budget_bytes
        self.per_seq_reservation_bytes = max_seq_len * bytes_per_token
        self.reserved_bytes = 0.0
        self.peak_reserved_bytes = 0.0
        self.peak_concurrent = 0

    def can_admit(self) -> bool:
        return self.reserved_bytes + self.per_seq_reservation_bytes <= self.budget_bytes

    def reserve(self) -> None:
        """Admission: reserve the FULL worst-case slice, sight unseen.
        This is the wasteful moment - we don't yet know if this sequence
        will use 40 of its reserved tokens or 240, and it doesn't matter:
        the reservation is the same size either way."""
        self.reserved_bytes += self.per_seq_reservation_bytes
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)

    def release(self) -> None:
        """Eviction: free the FULL worst-case slice back, even though the
        sequence almost certainly used only a fraction of it. Whatever
        fraction went unused was unusable by anyone ELSE for this
        sequence's entire lifetime too - that's the naive waste, and it's
        paid continuously, not just once at the end."""
        self.reserved_bytes -= self.per_seq_reservation_bytes

    def note_concurrency(self, active_count: int) -> None:
        self.peak_concurrent = max(self.peak_concurrent, active_count)


def run_naive_allocation(
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq_len: int = MAX_SEQ_LEN_NAIVE,
    budget_bytes: int = MEMORY_BUDGET_BYTES,
) -> Dict:
    tokenizer, model = load_model()
    eos_ids = eos_ids_for(tokenizer, model)
    num_layers = model.config.num_hidden_layers
    bytes_per_token = measure_kv_bytes_per_token(tokenizer, model)

    pool = NaiveMemoryPool(budget_bytes, max_seq_len, bytes_per_token)
    queue = list(build_workload())
    active: List[Dict] = []
    results = []
    admitted_over_time = []  # (iteration, active_count) - for the JSON record

    wall_start = time.perf_counter()

    for iteration in range(MAX_ITERATIONS):
        # --- EVICT: any sequence that just finished gives its FULL
        # worst-case reservation back, regardless of how much it used. ---
        still_active = []
        for slot in active:
            if slot["finished"]:
                pool.release()
                text = tokenizer.decode(slot["generated"], skip_special_tokens=True)
                actual_len = slot["prompt_tokens"] + len(slot["generated"])
                results.append({
                    "request_id": slot["request_id"],
                    "category": slot["category"],
                    "prompt_tokens": slot["prompt_tokens"],
                    "output_tokens": len(slot["generated"]),
                    "actual_len_tokens": actual_len,
                    "reserved_len_tokens": max_seq_len,
                    "waste_tokens": max_seq_len - actual_len,
                    "latency_s": time.perf_counter() - wall_start,
                    "text": text,
                })
            else:
                still_active.append(slot)
        active = still_active

        # --- ADMIT: pull from the queue while the FULL worst-case
        # reservation still fits in the remaining budget. A request that
        # only needs 41 tokens is gated by the same 256-token check as one
        # that will need all 242 - naive allocation cannot tell them apart
        # at admission time, by construction. ---
        while queue and pool.can_admit():
            req = queue.pop(0)
            input_ids = build_input_ids(tokenizer, req["prompt"])
            prompt_tokens = input_ids.shape[1]
            pool.reserve()
            with torch.inference_mode():
                out = model(input_ids, use_cache=True)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).view(1, 1)
            generated = [next_token.item()]
            active.append({
                "request_id": req["request_id"],
                "category": req["category"],
                "prompt_tokens": prompt_tokens,
                "generated": generated,
                "cache": out.past_key_values,
                "cur_len": prompt_tokens,
                "next_token": next_token,
                "finished": generated[-1] in eos_ids or len(generated) >= max_new_tokens,
            })

        pool.note_concurrency(len(active))
        admitted_over_time.append({"iteration": iteration, "active_count": len(active),
                                    "reserved_bytes": pool.reserved_bytes})

        if not active and not queue:
            break
        if not active:
            continue

        new_next_tokens, trimmed_caches, real_lens = run_batched_decode_step(model, active, num_layers)
        for j, slot in enumerate(active):
            slot["cache"] = trimmed_caches[j]
            slot["cur_len"] = real_lens[j]
            token_id = new_next_tokens[j].item()
            slot["generated"].append(token_id)
            slot["next_token"] = new_next_tokens[j].view(1, 1)
            if token_id in eos_ids or len(slot["generated"]) >= max_new_tokens:
                slot["finished"] = True
    else:
        raise RuntimeError(
            f"naive_allocation hit MAX_ITERATIONS={MAX_ITERATIONS} without draining the "
            "queue - likely a scheduler bug, not real workload size."
        )

    total_time_s = time.perf_counter() - wall_start
    total_output_tokens = sum(r["output_tokens"] for r in results)
    total_reserved_tokens = sum(r["reserved_len_tokens"] for r in results)
    total_used_tokens = sum(r["actual_len_tokens"] for r in results)

    return {
        "approach": "naive_allocation",
        "budget_bytes": budget_bytes,
        "bytes_per_token": bytes_per_token,
        "max_seq_len_reservation": max_seq_len,
        "peak_concurrent": pool.peak_concurrent,
        "peak_reserved_bytes": pool.peak_reserved_bytes,
        "waste_percent": 100.0 * (1 - total_used_tokens / total_reserved_tokens),
        "total_time_s": total_time_s,
        "throughput_tok_s": total_output_tokens / total_time_s,
        "admitted_over_time": admitted_over_time,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    print("[naive_allocation] loading model + workload ...", flush=True)
    summary = run_naive_allocation(args.max_new_tokens)

    print(f"[naive_allocation] peak_concurrent = {summary['peak_concurrent']}")
    print(f"[naive_allocation] waste_percent = {summary['waste_percent']:.1f}%")
    print(f"[naive_allocation] total_time_s = {summary['total_time_s']:.2f}s")
    print(f"[naive_allocation] throughput_tok_s = {summary['throughput_tok_s']:.2f}")

    for r in sorted(summary["results"], key=lambda r: r["request_id"]):
        print(f"  [{r['request_id']:2d}] {r['category']:5s} "
              f"actual={r['actual_len_tokens']:3d} reserved={r['reserved_len_tokens']:3d} "
              f"waste={r['waste_tokens']:3d} latency={r['latency_s']:.3f}s")


if __name__ == "__main__":
    main()
