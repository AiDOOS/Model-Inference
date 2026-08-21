"""
Module 03 - Shared model loading, workload, and KV-cache memory accounting.

naive_allocation.py, manual_paged_attention.py, and vllm_paged.py all import
from here so every approach admits/evicts against the exact same fixed
memory budget, the exact same 24-request workload, and the exact same
real, MEASURED (not formula-guessed) KV-cache byte size per token. Only the
ALLOCATION POLICY differs between the three scripts - everything else about
the comparison is held fixed on purpose.

Workload is the same short/long prompt mix module 02 used (verified there:
short prompts really do hit EOS early - 8 to 28 output tokens - while long
prompts really do run to the shared token cap), just scaled up from 16 to
24 requests (6 blocks of 1 long + 3 short instead of 4) so there's enough
concurrency headroom in the workload for paging's advantage to actually
show up as a measured peak, not just a formula.
"""

from typing import Dict, List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 60          # shared generation budget/cap for every request (same as module 02)

# --- The ONE memory budget every approach is measured against. ---
# 24 MiB was picked, after measuring this model's real KV-cache size below,
# specifically so naive allocation's worst-case reservation fits an exact,
# round 4 concurrent sequences - small enough to force real queueing on a
# workload of 24 requests, which is the whole point: prove paging fits
# more by measuring it, not by asserting it.
MEMORY_BUDGET_BYTES = 24 * 1024 * 1024   # 24 MiB

# Naive allocation's worst-case reservation, in TOKENS of context, per
# sequence. Must cover the longest possible sequence in the workload
# (longest prompt 182 tokens + MAX_NEW_TOKENS 60 = 242) with real margin,
# the same way a real server has to size its per-slot buffer for whatever
# max_model_len it advertises, not for what any given request turns out to
# need. 256 is that cap - and not coincidentally, MEMORY_BUDGET_BYTES was
# chosen so BUDGET / (256 tokens * bytes_per_token) comes out to exactly 4.
MAX_SEQ_LEN_NAIVE = 256

# Manual paged attention's block size, in tokens. 16 is the same order of
# magnitude vLLM itself defaults to (16 tokens/block in most vLLM configs).
BLOCK_SIZE_TOKENS = 16

# Fraction of the total block pool that admission must always leave free,
# never touching it to admit a new sequence - see manual_paged_attention.py
# for why this exists (a real deadlock, not just a slow queue, shows up
# without it): vLLM has this exact knob, called `watermark`, for the same
# reason. Ours is far larger than vLLM's real default (~0.01) because this
# module's whole pool is only 64 blocks - a toy scale where a handful of
# blocks is a meaningful fraction, unlike a real deployment's thousands.
PAGED_WATERMARK_FRACTION = 0.10

MAX_ITERATIONS = 4000  # safety cap against an infinite loop from a scheduler bug, not a real limit

SHORT_PROMPTS = [
    "What is the capital of France? Answer in one short sentence.",
    "Is the sun a star? Answer in one short sentence.",
    "What color do you get mixing blue and yellow? One short sentence.",
]

LONG_PROMPTS = [
    "Write a detailed, multi-paragraph explanation of how photosynthesis "
    "works, covering the light-dependent and light-independent reactions, "
    "the specific role of chlorophyll and other pigments in capturing "
    "light energy, the electron transport chain that produces ATP and "
    "NADPH, how the Calvin cycle uses those to fix carbon dioxide into "
    "glucose, and why this process matters for essentially all life on "
    "Earth, not just plants, since it's the base of nearly every food "
    "chain and the source of atmospheric oxygen. Discuss also how factors "
    "like light intensity, temperature, and carbon dioxide concentration "
    "affect the rate of photosynthesis. Cover at least six distinct points "
    "in depth, with concrete detail at each step rather than a "
    "surface-level summary, as if explaining this to a biology student "
    "preparing for an exam.",
    "Write a detailed, multi-paragraph explanation of how the water cycle "
    "works, covering evaporation from oceans, lakes, and soil, "
    "transpiration from plants, condensation into clouds, precipitation "
    "as rain, snow, or hail, and collection back into rivers, lakes, and "
    "groundwater aquifers. Explain why each stage matters for climate "
    "regulation and for fresh water availability to humans and "
    "ecosystems, how the cycle interacts with ocean currents, and how "
    "human activity such as deforestation, groundwater extraction, and "
    "climate change can disrupt the cycle at each stage. Cover at least "
    "six distinct points in depth, with concrete detail rather than a "
    "surface-level summary, as if explaining this to a geography student "
    "preparing for an exam.",
    "Write a detailed, multi-paragraph explanation of how vaccines train "
    "the immune system, covering what antigens are, how the innate "
    "immune system first responds, how the adaptive immune system's B "
    "cells and T cells recognize specific antigens, the difference "
    "between the initial primary immune response and the much faster "
    "memory response on later exposure to the real pathogen, why some "
    "vaccines need multiple doses or periodic boosters to establish "
    "strong memory, and how different vaccine technologies (inactivated, "
    "live-attenuated, mRNA) each achieve this differently. Cover at least "
    "six distinct points in depth, with concrete detail at each step "
    "rather than a surface-level summary, as if explaining this to a "
    "biology student preparing for an exam.",
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def eos_ids_for(tokenizer, model) -> Set[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif isinstance(gen_eos, (list, tuple)):
        ids.update(gen_eos)
    return ids


def build_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(chat_text, return_tensors="pt").input_ids


def build_workload() -> List[Dict]:
    """24 requests: 6 blocks of (1 long, 3 short), request_id 0..23 in
    arrival order, all arriving simultaneously (same convention as module
    02) so every approach's admission controller - not request timing - is
    what's actually being measured."""
    requests = []
    request_id = 0
    for block in range(6):
        requests.append({
            "request_id": request_id,
            "prompt": LONG_PROMPTS[block % len(LONG_PROMPTS)],
            "category": "long",
        })
        request_id += 1
        for i in range(3):
            requests.append({
                "request_id": request_id,
                "prompt": SHORT_PROMPTS[(block * 3 + i) % len(SHORT_PROMPTS)],
                "category": "short",
            })
            request_id += 1
    return requests


def measure_kv_bytes_per_token(tokenizer, model) -> float:
    """The real per-token KV-cache footprint for THIS model, measured by
    actually running one forward pass and inspecting the resulting cache
    tensors - not computed from a num_layers*num_kv_heads*head_dim formula.
    Formulas can be wrong (e.g. for GQA models the "heads" that matter for
    cache size are num_key_value_heads, not num_attention_heads); the
    actual tensor shapes PyTorch produced can't be.

    Sums K + V across every layer, for a real short prompt, then divides by
    that prompt's token count to get bytes/token. This is the single number
    every approach's memory-budget math is built on."""
    input_ids = build_input_ids(tokenizer, "Hello, how are you today?")
    seq_len = input_ids.shape[1]
    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
    num_layers = model.config.num_hidden_layers
    total_bytes = 0
    for i in range(num_layers):
        layer = out.past_key_values.layers[i]
        total_bytes += layer.keys.numel() * layer.keys.element_size()
        total_bytes += layer.values.numel() * layer.values.element_size()
    return total_bytes / seq_len


# --- Shared batched-decode plumbing (same technique as module 02's
# continuous_batching_toy.py: every active sequence has its own KV cache,
# of its own length, and a single batched matmul needs one shared shape -
# so every iteration, pad/stack/run/split). Both naive_allocation.py and
# manual_paged_attention.py reuse this UNCHANGED; the only thing that
# differs between those two scripts is the memory-accounting policy that
# decides WHO gets a seat in this batch, not how the batch itself runs. ---

def pad_cache_left(cache: DynamicCache, cur_len: int, max_len: int, num_layers: int) -> DynamicCache:
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


def stack_caches(padded_caches: List[DynamicCache], num_layers: int) -> DynamicCache:
    batched = DynamicCache()
    for i in range(num_layers):
        keys = torch.cat([c.layers[i].keys for c in padded_caches], dim=0)
        values = torch.cat([c.layers[i].values for c in padded_caches], dim=0)
        batched.update(keys, values, i)
    return batched


def split_and_trim(batched_cache: DynamicCache, real_lens: List[int], num_layers: int) -> List[DynamicCache]:
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


def run_batched_decode_step(model, active_slots: List[Dict], num_layers: int):
    """One batched decode step across `active_slots` (each a dict with at
    least "cache", "cur_len", "next_token"). Returns (new_next_tokens,
    trimmed_caches, real_lens) - caller is responsible for writing these
    back into its own slot representation, since naive and paged slots
    otherwise carry different bookkeeping (memory reservation vs. block
    table)."""
    cur_lens = [s["cur_len"] for s in active_slots]
    max_len = max(cur_lens)

    padded_caches = [
        pad_cache_left(s["cache"], s["cur_len"], max_len, num_layers)
        for s in active_slots
    ]
    batched_cache = stack_caches(padded_caches, num_layers)

    next_tokens_batch = torch.cat([s["next_token"] for s in active_slots], dim=0)
    attention_mask = torch.zeros((len(active_slots), max_len + 1), dtype=torch.long)
    position_ids = torch.zeros((len(active_slots), 1), dtype=torch.long)
    for j, cur_len in enumerate(cur_lens):
        attention_mask[j, max_len - cur_len:] = 1
        position_ids[j, 0] = cur_len

    with torch.inference_mode():
        out = model(
            next_tokens_batch, past_key_values=batched_cache,
            attention_mask=attention_mask, position_ids=position_ids, use_cache=True,
        )
    new_next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)

    real_lens = [cur_len + 1 for cur_len in cur_lens]
    trimmed_caches = split_and_trim(out.past_key_values, real_lens, num_layers)
    return new_next_tokens, trimmed_caches, real_lens


if __name__ == "__main__":
    tokenizer, model = load_model()
    bpt = measure_kv_bytes_per_token(tokenizer, model)
    print(f"measured bytes/token = {bpt:.1f}  ({bpt/1024:.2f} KB/token)")
    print(f"naive per-seq reservation = {MAX_SEQ_LEN_NAIVE} tok x {bpt/1024:.2f} KB = "
          f"{MAX_SEQ_LEN_NAIVE*bpt/1024/1024:.2f} MB")
    print(f"naive max concurrent (budget/reservation) = "
          f"{MEMORY_BUDGET_BYTES // (MAX_SEQ_LEN_NAIVE*bpt):.0f}")
    print(f"paged total blocks (budget/block_bytes) = "
          f"{MEMORY_BUDGET_BYTES // (BLOCK_SIZE_TOKENS*bpt):.0f}")
    for req in build_workload():
        ids = build_input_ids(tokenizer, req["prompt"])
        print(f"[{req['request_id']:2d}] {req['category']:5s}  prompt_tokens={ids.shape[1]}")
