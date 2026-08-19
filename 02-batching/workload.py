"""
Module 02 - Shared model loading + the ONE fixed workload every approach runs.

All four serving approaches (no_batching, static_batching,
continuous_batching_toy, vllm_batching) and benchmark.py import
build_workload() from here so they're all measuring the exact same 16
requests, in the exact same order, every time. That's what makes the
four-way comparison fair - only the SERVING STRATEGY changes between runs,
nothing about the requests themselves does.

The workload is a deliberate MIX, not a random sample:
  - SHORT requests: short prompts, worded to elicit a brief answer. Real
    models actually DO stop early on these (verified: 8-28 output tokens
    before EOS, well under the shared cap) - this isn't a fabricated
    "pretend this one is short" label, it's a real, reproducible behavior
    difference.
  - LONG requests: long, detailed prompts that ask for a thorough,
    multi-paragraph answer. These consistently run to the shared
    MAX_NEW_TOKENS cap without hitting EOS.
  - The 16 requests are arranged as 4 blocks of (1 long + 3 short), so
    every batch of BATCH_SIZE=8 (static batching's batch, and continuous
    batching's max concurrent slots) contains 2 long + 6 short requests -
    a short request is ALWAYS sitting in the same batch as a long one,
    which is exactly the scenario that exposes head-of-line blocking.
"""

from typing import Dict, List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 60   # shared generation budget/cap for every request
BATCH_SIZE = 8        # static batch size AND continuous scheduler's max concurrent slots

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
    """16 requests: 4 blocks of (1 long, 3 short), request_id 0..15 in
    arrival order. Every approach receives this exact same list."""
    requests = []
    request_id = 0
    for block in range(4):
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


if __name__ == "__main__":
    tokenizer, _ = load_model()
    for req in build_workload():
        ids = build_input_ids(tokenizer, req["prompt"])
        print(f"[{req['request_id']:2d}] {req['category']:5s}  prompt_tokens={ids.shape[1]}")
