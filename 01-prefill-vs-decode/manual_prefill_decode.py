"""
Module 01 - Prefill vs. decode, isolated.

Module 00 already split prefill and decode by hand instead of calling
generate(), but it only ever showed ONE version of decode: the one that
reuses past_key_values. This module isolates the two phases further and
adds the comparison that actually proves they're different *kinds* of
computation, not just two timers around the same loop:

  run_no_cache(...)   - "bypass generate(), split prefill/decode explicitly",
                         but decode does NOT reuse any cache. Every step
                         re-runs the full forward pass over the entire
                         sequence generated so far. Each step is really its
                         own little prefill over a sequence that grows by
                         one token every time.
  run_with_cache(...) - same split, but decode reuses past_key_values.
                         Each step only ever forward-passes ONE new token;
                         all prior context is already summarized in the KV
                         cache.

Both share the same PREFILL call - model(input_ids, use_cache=...) once,
over the full prompt, timed as a single number - because prefill's cost
comes from processing the whole prompt in one parallel pass, not from
whether the KV cache is going to be kept afterward.

Expected shape of the results (see README for why):
  - prefill_time_s is one big number, roughly equal whether or not the
    cache is kept afterward.
  - with_cache decode_times_s are many SMALL, roughly FLAT numbers -
    memory-bound, one new token per step, cost independent of position.
  - no_cache decode_times_s are many numbers that GROW step over step -
    compute-bound, each step reprocesses a sequence one token longer than
    the last, so cost scales with position.
"""

import argparse
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_PROMPT = (
    "Explain, in a few sentences, why the sky appears blue during the day "
    "and orange or red at sunset. Keep the explanation accessible to a "
    "curious 10-year-old."
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def eos_ids_for(tokenizer, model) -> set:
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


def run_with_cache(
    model, input_ids: torch.Tensor, max_new_tokens: int, eos_ids: set
) -> Dict:
    """Prefill once, then decode by feeding one new token + the cached
    past_key_values each step. This is what a real inference server does."""
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
    prefill_time_s = time.perf_counter() - t0

    past_key_values = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
    generated = [next_token.item()]

    decode_times_s: List[float] = []
    cur_token = next_token
    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos_ids:
            break
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(cur_token, past_key_values=past_key_values, use_cache=True)
        decode_times_s.append(time.perf_counter() - t0)
        past_key_values = out.past_key_values
        cur_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated.append(cur_token.item())

    return {
        "prefill_time_s": prefill_time_s,
        "decode_times_s": decode_times_s,
        "generated": generated,
    }


def run_no_cache(
    model, input_ids: torch.Tensor, max_new_tokens: int, eos_ids: set
) -> Dict:
    """Prefill once (cache discarded), then decode by re-running the full
    forward pass over the ENTIRE sequence so far, every step. No
    past_key_values are ever reused. Each decode step is, computationally,
    just another prefill - over a sequence that's one token longer than the
    step before it."""
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids, use_cache=False)
    prefill_time_s = time.perf_counter() - t0

    next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
    generated = [next_token.item()]
    full_seq = torch.cat([input_ids, next_token], dim=-1)

    decode_times_s: List[float] = []
    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos_ids:
            break
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(full_seq, use_cache=False)
        decode_times_s.append(time.perf_counter() - t0)
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated.append(next_token.item())
        full_seq = torch.cat([full_seq, next_token], dim=-1)

    return {
        "prefill_time_s": prefill_time_s,
        "decode_times_s": decode_times_s,
        "generated": generated,
    }


def _stats(values: List[float]) -> str:
    if not values:
        return "n/a (0 decode steps taken - EOS on first token)"
    n = len(values)
    mean = sum(values) / n
    return (
        f"n={n}  mean={mean:.4f}s  first={values[0]:.4f}s  last={values[-1]:.4f}s  "
        f"min={min(values):.4f}s  max={max(values):.4f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    print(f"[manual] loading {MODEL_NAME} ...", flush=True)
    t0 = time.perf_counter()
    tokenizer, model = load_model()
    print(f"[manual] model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    eos_ids = eos_ids_for(tokenizer, model)
    input_ids = build_input_ids(tokenizer, args.prompt)
    prompt_tokens = input_ids.shape[1]
    print(f"[manual] prompt_tokens={prompt_tokens}  max_new_tokens={args.max_new_tokens}")

    print("\n[manual] running WITH cache (past_key_values reused) ...")
    with_cache = run_with_cache(model, input_ids, args.max_new_tokens, eos_ids)

    print("[manual] running WITHOUT cache (full sequence re-run every step) ...")
    no_cache = run_no_cache(model, input_ids, args.max_new_tokens, eos_ids)

    print()
    print(f"prefill_time_s (with_cache) : {with_cache['prefill_time_s']:.4f}s")
    print(f"prefill_time_s (no_cache)   : {no_cache['prefill_time_s']:.4f}s")
    print(f"  -> prefill cost is ~equal either way: it's one parallel pass over")
    print(f"     the whole prompt, independent of whether the cache is kept.")
    print()
    print(f"decode_times_s (with_cache) : {_stats(with_cache['decode_times_s'])}")
    print(f"decode_times_s (no_cache)   : {_stats(no_cache['decode_times_s'])}")
    print()
    if with_cache["decode_times_s"] and no_cache["decode_times_s"]:
        wc_mean = sum(with_cache["decode_times_s"]) / len(with_cache["decode_times_s"])
        nc_first = no_cache["decode_times_s"][0]
        nc_last = no_cache["decode_times_s"][-1]
        print(f"  -> with_cache decode/step stays flat (~{wc_mean:.4f}s regardless of position).")
        print(f"  -> no_cache decode/step GROWS with position: step 1 = {nc_first:.4f}s, "
              f"step {len(no_cache['decode_times_s'])} = {nc_last:.4f}s.")
        print(f"     Every no_cache step reprocesses the whole sequence so far, so cost")
        print(f"     scales with how many tokens have been generated - decode without a")
        print(f"     cache degrades toward prefill-like, quadratic-total cost.")
    print()
    print(f"prefill_time_s vs single decode step: prefill is one big parallel pass over")
    print(f"{prompt_tokens} tokens; a with_cache decode step processes exactly 1 token.")


if __name__ == "__main__":
    main()
