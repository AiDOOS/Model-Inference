"""
Module 00 - Naive LLM serving.

Loads Qwen2.5-0.5B-Instruct once at startup and exposes POST /generate.
Two things are deliberately naive here, on purpose, so later modules have
something concrete to fix:

1. Every request is served with plain, uninstrumented-by-the-library
   inference: no continuous batching, no paged attention, no scheduler.
   Prefill and decode are done manually (model(...) calls, not
   model.generate()) so their costs can be timed separately instead of
   hidden inside one black-box .generate() call.

2. A single global lock (_model_lock) wraps each ENTIRE request - prefill
   AND its whole decode loop. Only one request touches the model at a
   time; everything else queues. That queueing is the naive bottleneck
   benchmark.py measures. A real scheduler would interleave many
   requests' decode steps together (continuous batching); this server
   does the opposite on purpose.
"""

import argparse
import threading
import time
from typing import List

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

app = FastAPI(title="00-naive-serving")

print(f"[server] loading {MODEL_NAME} ...", flush=True)
_load_start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()
print(f"[server] model loaded in {time.perf_counter() - _load_start:.2f}s", flush=True)

_eos_ids = set()
if tokenizer.eos_token_id is not None:
    _eos_ids.add(tokenizer.eos_token_id)
_gen_eos = getattr(model.generation_config, "eos_token_id", None)
if isinstance(_gen_eos, int):
    _eos_ids.add(_gen_eos)
elif isinstance(_gen_eos, (list, tuple)):
    _eos_ids.update(_gen_eos)

# Naive on purpose: one model, one lock. See module docstring above.
_model_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64


class GenerateResponse(BaseModel):
    prefill_time_s: float
    decode_times_s: List[float]
    prompt_tokens: int
    output_tokens: int
    text: str
    queue_wait_s: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    # FastAPI runs sync `def` endpoints in a worker thread pool, so many
    # of these can be in flight concurrently. They all then queue on the
    # single model lock below - that wait is real queueing time, captured
    # as queue_wait_s.
    queue_start = time.perf_counter()
    with _model_lock:
        queue_wait_s = time.perf_counter() - queue_start
        return _run(req.prompt, req.max_new_tokens, queue_wait_s)


def _run(prompt: str, max_new_tokens: int, queue_wait_s: float) -> GenerateResponse:
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(chat_text, return_tensors="pt").input_ids
    prompt_tokens = input_ids.shape[1]

    # ---- PREFILL ----
    # One forward pass over the *entire* prompt at once. Cost scales with
    # prompt length, not with how many tokens we're about to generate.
    # This single number is effectively time-to-first-token.
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
    prefill_time_s = time.perf_counter() - t0

    past_key_values = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
    generated = [next_token.item()]

    # ---- DECODE ----
    # Strictly sequential: one new token in, one new token out, reusing the
    # KV cache from every step before it. Token 0 above was a free
    # byproduct of the prefill pass (its logits predict the next token), so
    # decode_times_s holds one entry per *additional* sequential step
    # actually taken - i.e. len(decode_times_s) == output_tokens - 1.
    decode_times_s: List[float] = []
    cur_token = next_token
    for _ in range(max_new_tokens - 1):
        if generated[-1] in _eos_ids:
            break
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(cur_token, past_key_values=past_key_values, use_cache=True)
        decode_times_s.append(time.perf_counter() - t0)
        past_key_values = out.past_key_values
        cur_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated.append(cur_token.item())

    text = tokenizer.decode(generated, skip_special_tokens=True)

    return GenerateResponse(
        prefill_time_s=prefill_time_s,
        decode_times_s=decode_times_s,
        prompt_tokens=prompt_tokens,
        output_tokens=len(generated),
        text=text,
        queue_wait_s=queue_wait_s,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # workers=1: there is exactly ONE model instance in memory, on purpose.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
