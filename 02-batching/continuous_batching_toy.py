"""
Module 02 - Approach 3: Continuous batching (toy, manual scheduler).

This is the "ride-share van" from the README: people hop on and off at
every stop, nobody waits for the slowest passenger. Concretely: a fixed
number of SLOTS (MAX_SLOTS = workload.BATCH_SIZE, same number as static
batching's batch size - so the only thing that changes between the two
approaches is the SCHEDULING POLICY, not how much concurrency is allowed).
Every iteration:

  1. EVICT any slot whose sequence just finished (hit EOS or the shared
     token cap) - free it up immediately, don't wait for its neighbors.
  2. ADMIT a new waiting request into any slot that's now free - this is
     the whole point: a short request queued behind a long one does NOT
     sit idle until the long one's entire batch drains. It gets a seat as
     soon as ANY seat opens up.
  3. Run ONE real batched decode step across whatever's currently in the
     slots - which may now be a mix of sequences that have been running
     for 40 steps and one that was just admitted a moment ago.

HONEST SIMPLIFICATION (see README "Gotcha" for the full version): a newly
admitted request pays for its own prefill as a separate, individual
forward pass at admission time (step 2), not fused into the same matmul as
everyone else's decode step. Real continuous-batching engines (and vLLM)
can mix prefill and decode work for DIFFERENT sequences into the SAME
iteration's compute via chunked/fused prefill. This toy scheduler keeps
prefill and decode as two distinct kinds of forward passes to stay
correct and readable - it demonstrates the ADMISSION/EVICTION policy (the
scheduling idea), not the fused-kernel engineering vLLM does on top of it.

WHY THE DECODE STEP HAS TO RE-PAD EVERY ITERATION: each active slot has
its own KV cache, and those caches are different lengths (slots joined at
different iterations). A single batched matmul needs one shared shape, so
every iteration: left-pad each slot's cache up to the current max active
length, run ONE batched forward pass, then immediately split the result
back into individual per-slot caches (trimmed of the padding) before the
next iteration - because by then the active set may have changed again.
This is a real, documented technique (sometimes called iteration-level /
Orca-style scheduling) - it's what continuous batching looked like BEFORE
PagedAttention. vLLM's actual memory management (module 05) is what makes
this cheap at scale instead of re-padding by hand every step.
"""

import argparse
import time
from typing import Dict, List, Optional

import torch
from transformers.cache_utils import DynamicCache

from cpu_utilization import CpuUtilizationSampler
from workload import BATCH_SIZE, MAX_NEW_TOKENS, build_input_ids, build_workload, eos_ids_for, load_model

MAX_SLOTS = BATCH_SIZE  # same concurrency cap as static batching - policy is the only variable
MAX_ITERATIONS = 2000    # safety cap against an infinite loop from a scheduler bug, not a real limit


def _pad_cache_left(cache: DynamicCache, cur_len: int, max_len: int, num_layers: int) -> DynamicCache:
    """Single-sequence cache -> single-sequence cache, left-padded with
    zeros from cur_len up to max_len. Padding is masked out by
    attention_mask later, so its actual (zero) values never affect logits."""
    pad_amount = max_len - cur_len
    padded = DynamicCache()
    for i in range(num_layers):
        layer = cache.layers[i]
        key, value = layer.keys, layer.values
        if pad_amount > 0:
            key = torch.nn.functional.pad(key, (0, 0, pad_amount, 0))
            value = torch.nn.functional.pad(value, (0, 0, pad_amount, 0))
        padded.update(key, value, i)
    return padded


def _stack_caches(padded_caches: List[DynamicCache], num_layers: int) -> DynamicCache:
    """N single-sequence caches (all now the same length) -> one batched cache."""
    batched = DynamicCache()
    for i in range(num_layers):
        keys = torch.cat([c.layers[i].keys for c in padded_caches], dim=0)
        values = torch.cat([c.layers[i].values for c in padded_caches], dim=0)
        batched.update(keys, values, i)
    return batched


def _split_and_trim(batched_cache: DynamicCache, real_lens: List[int], num_layers: int) -> List[DynamicCache]:
    """Batched cache -> N single-sequence caches, each trimmed back down to
    its own real length (undoing this iteration's left-padding) so next
    iteration starts from exactly the right size again, not the padded one."""
    result = []
    for j, real_len in enumerate(real_lens):
        single = DynamicCache()
        for i in range(num_layers):
            layer = batched_cache.layers[i]
            key = layer.keys[j:j + 1, :, -real_len:, :].contiguous()
            value = layer.values[j:j + 1, :, -real_len:, :].contiguous()
            single.update(key, value, i)
        result.append(single)
    return result


def run_continuous_batching_toy(
    max_new_tokens: int = MAX_NEW_TOKENS, max_slots: int = MAX_SLOTS
) -> Dict:
    tokenizer, model = load_model()
    eos_ids = eos_ids_for(tokenizer, model)
    num_layers = model.config.num_hidden_layers
    workload = build_workload()
    queue = list(workload)  # popped from the front as slots free up

    slots: List[Optional[Dict]] = [None] * max_slots
    results = []

    sampler = CpuUtilizationSampler(interval_s=0.5)
    sampler.start()
    wall_start = time.perf_counter()

    for _iteration in range(MAX_ITERATIONS):
        # --- 1. EVICT: free any slot whose sequence just finished. ---
        # No waiting for siblings - the instant a sequence is done, its
        # seat is available for someone else next step.
        for idx in range(max_slots):
            slot = slots[idx]
            if slot is not None and slot["finished"]:
                text = tokenizer.decode(slot["generated"], skip_special_tokens=True)
                results.append({
                    "request_id": slot["request_id"],
                    "category": slot["category"],
                    "prompt_tokens": slot["prompt_tokens"],
                    "output_tokens": len(slot["generated"]),
                    "latency_s": time.perf_counter() - wall_start,
                    "text": text,
                })
                slots[idx] = None

        # --- 2. ADMIT: fill every free slot with the next waiting request. ---
        # This is the core difference from static batching: admission
        # happens per-slot, per-iteration, not "wait until every slot in
        # the batch is simultaneously free." A newcomer pays its own
        # individual prefill cost right here (see module docstring for why
        # this is done as a separate forward pass) and is folded into the
        # shared decode step immediately after, in this same iteration.
        for idx in range(max_slots):
            if slots[idx] is None and queue:
                req = queue.pop(0)
                input_ids = build_input_ids(tokenizer, req["prompt"])
                prompt_tokens = input_ids.shape[1]
                with torch.inference_mode():
                    out = model(input_ids, use_cache=True)
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1).view(1, 1)
                generated = [next_token.item()]
                slots[idx] = {
                    "request_id": req["request_id"],
                    "category": req["category"],
                    "prompt_tokens": prompt_tokens,
                    "generated": generated,
                    "cache": out.past_key_values,
                    "cur_len": prompt_tokens,
                    "next_token": next_token,
                    "finished": generated[-1] in eos_ids or len(generated) >= max_new_tokens,
                }

        active_indices = [i for i in range(max_slots) if slots[i] is not None]
        if not active_indices and not queue:
            break  # every request admitted, decoded, and evicted - done
        if not active_indices:
            continue  # shouldn't happen (queue non-empty implies a free slot got filled above), but safe

        # A slot admitted THIS iteration may already be "finished" (e.g. a
        # one-token answer) - it'll ride through one harmless batched step
        # below and get evicted at the top of the NEXT iteration. Simpler
        # than special-casing it out of the batch mid-iteration.

        # --- 3. BATCHED DECODE STEP across whatever's active right now. ---
        cur_lens = [slots[i]["cur_len"] for i in active_indices]
        max_len = max(cur_lens)

        padded_caches = [
            _pad_cache_left(slots[i]["cache"], slots[i]["cur_len"], max_len, num_layers)
            for i in active_indices
        ]
        batched_cache = _stack_caches(padded_caches, num_layers)

        next_tokens_batch = torch.cat([slots[i]["next_token"] for i in active_indices], dim=0)
        attention_mask = torch.zeros((len(active_indices), max_len + 1), dtype=torch.long)
        position_ids = torch.zeros((len(active_indices), 1), dtype=torch.long)
        for j, i in enumerate(active_indices):
            cur_len = slots[i]["cur_len"]
            attention_mask[j, max_len - cur_len:] = 1  # real history (right-aligned) + the new token column
            position_ids[j, 0] = cur_len  # new token's own position in ITS sequence, unaffected by padding

        with torch.inference_mode():
            out = model(
                next_tokens_batch, past_key_values=batched_cache,
                attention_mask=attention_mask, position_ids=position_ids, use_cache=True,
            )
        new_next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)

        real_lens = [cur_len + 1 for cur_len in cur_lens]
        trimmed_caches = _split_and_trim(out.past_key_values, real_lens, num_layers)

        for j, i in enumerate(active_indices):
            slot = slots[i]
            slot["cache"] = trimmed_caches[j]
            slot["cur_len"] = real_lens[j]
            token_id = new_next_tokens[j].item()
            slot["generated"].append(token_id)
            slot["next_token"] = new_next_tokens[j].view(1, 1)
            if token_id in eos_ids or len(slot["generated"]) >= max_new_tokens:
                slot["finished"] = True
    else:
        raise RuntimeError(
            f"continuous_batching_toy hit MAX_ITERATIONS={MAX_ITERATIONS} without draining the "
            "queue - likely a scheduler bug (a slot stuck never finishing), not real workload size."
        )

    total_time_s = time.perf_counter() - wall_start
    cpu_samples = sampler.stop()
    total_output_tokens = sum(r["output_tokens"] for r in results)

    return {
        "approach": "continuous_batching_toy",
        "max_slots": max_slots,
        "total_time_s": total_time_s,
        "throughput_tok_s": total_output_tokens / total_time_s,
        "avg_cpu_percent": CpuUtilizationSampler.average(cpu_samples),
        "cpu_samples": cpu_samples,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-slots", type=int, default=MAX_SLOTS)
    args = parser.parse_args()

    print("[continuous_batching_toy] loading model + workload ...", flush=True)
    summary = run_continuous_batching_toy(args.max_new_tokens, args.max_slots)

    print(f"[continuous_batching_toy] total_time_s = {summary['total_time_s']:.2f}s")
    print(f"[continuous_batching_toy] throughput_tok_s = {summary['throughput_tok_s']:.2f}")
    print(f"[continuous_batching_toy] avg_cpu_percent = {summary['avg_cpu_percent']:.1f}%")

    short_latencies = [r["latency_s"] for r in summary["results"] if r["category"] == "short"]
    long_latencies = [r["latency_s"] for r in summary["results"] if r["category"] == "long"]
    print(f"[continuous_batching_toy] short-request avg latency = {sum(short_latencies)/len(short_latencies):.2f}s")
    print(f"[continuous_batching_toy] long-request avg latency  = {sum(long_latencies)/len(long_latencies):.2f}s")

    for r in sorted(summary["results"], key=lambda r: r["request_id"]):
        print(f"  [{r['request_id']:2d}] {r['category']:5s} "
              f"prompt_tokens={r['prompt_tokens']:3d} output_tokens={r['output_tokens']:3d} "
              f"latency={r['latency_s']:.3f}s")


if __name__ == "__main__":
    main()
