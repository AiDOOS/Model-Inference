# Results

## Smoke test (2026-08-19)

Not the full benchmark - a reduced run used only to verify server.py and
benchmark.py work end-to-end. Levels 1/2/4, 8 max new tokens, on the CPU
described in the module README.

```
python benchmark.py --levels 1 2 4 --max-new-tokens 8
```

```
  Conc   Reqs   Wall(s)   Tput(tok/s)   AvgPrefill(s)   AvgDecode/tok(s)   AvgQueueWait(s)   AvgE2E(s)
------------------------------------------------------------------------------------------------------
     1      1      2.39          3.34          1.0570             0.1870            0.0000        2.39
     2      2      4.76          3.36          1.0175             0.1914            1.1954        3.59
     4      4      9.63          3.32          1.0192             0.1954            3.5706        5.99

Prefill/Decode = server-side compute time, held while it owns the model lock.
QueueWait/E2E  = client-visible cost of everyone else queueing ahead of you.
```

**Reading it:**

- `AvgPrefill` and `AvgDecode/tok` are flat across concurrency (~1.02s,
  ~0.19s) - whoever holds the model lock always has the CPU to itself, so
  its own compute cost doesn't change.
- `AvgQueueWait` and `AvgE2E` grow with concurrency (0s -> 1.20s -> 3.57s
  queue wait; 2.39s -> 3.59s -> 5.99s end-to-end) - requests piling up
  behind each other.
- `Tput(tok/s)` stays flat (~3.3) instead of scaling up with concurrency -
  proof that adding concurrent load buys nothing here but worse latency.

## Full benchmark

Not yet run. The default sweep (`python benchmark.py`, levels 1/4/8/16/32,
64 tokens) takes an estimated 15-20 minutes on this CPU, since every request
is served strictly one at a time by design. Run it and append results here
when ready.
