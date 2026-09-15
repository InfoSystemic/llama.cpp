# Branches in this fork

Measured on a Lenovo SR950: 4x Xeon Gold 6242 (Cascade Lake, AVX-512 VNNI, no AMX), 755 GB DDR4, 381 GB/s
aggregate measured. Models: Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, GLM-5.3-Flash.

Full methodology and the measurements behind these: https://github.com/InfoSystemic/duck-duck-llama

## Ready to propose upstream

| branch | change | status |
|---|---|---|
| `unary-ops-parallel` | `ggml_get_n_tasks()` gives 17 of 22 unary ops `n_tasks = 1` while `GELU`/`SILU` in the same switch get `n_threads`. All 22 dispatch through the same `apply_unary_op`, which splits work with `get_thread_range`, so there is no implementation difference behind it — `SILU` is threaded and `SIGMOID` is not. A 4-line deletion. | **bit-exact**, measured +3.4% on Qwen3.8-Flash-Next and +2.1% on GLM-5.3-Flash, greedy output byte-identical |
| `kv-seq-bounded-scan` | `seq_rm`/`seq_cp`/`seq_add`/`seq_div` walk every cell, but each loop body tests `cells.pos_in()`, empty cells hold `pos == -1`, and all four callers clamp `p0 >= 0` — so nothing at or beyond `used_max_p1()` can match. Bounds the scans there. Net −4 lines. | **behaviour unchanged**; matters for speculative decode at long context, where every rejected draft token triggers `seq_rm` over a cache sized by the allocated context rather than the occupied part |

Both are bit-exact and neither touches selection, so neither can have the failure mode below.

## Retained as a record, NOT to be merged

| branch | why it is here |
|---|---|
| `topk-linear-selection` | **Refuted in practice.** Histogram selection makes `ggml_compute_forward_top_k_f32` 4-7% cheaper per cycle and the *model* 7-17% slower, because masked (`-INFINITY`) scores make the tie tail arbitrary and the model then attends to different positions. It is **conformant to `test_top_k`**, which explicitly permits different indices for tied values — so an implementation can satisfy the op's documented contract and still degrade a model. The commit carries the full A/B and the selection timings, which are themselves useful: `nth_element`, the intuitive replacement, is 1.66x *slower* than `partial_sort` at 262,144 while winning below ~100K. |

## The lesson worth carrying

The top-k branch passed a microbenchmark at five context sizes with zero differing indices, passed a
lineage-verified build, and passed a control arm — and was still wrong, because the benchmark fed
normal-distributed scores while real indexer scores are heavily masked. Validate selection changes end-to-end on a
model, not against the op test alone.
