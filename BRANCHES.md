# Branches in this fork

Measured on a Lenovo SR950: 4x Xeon Gold 6242 (Cascade Lake, AVX-512 VNNI, no AMX), 755 GB DDR4, 381 GB/s
aggregate measured. Models: Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, GLM-5.3-Flash.

Full methodology and the measurements behind these: https://github.com/InfoSystemic/duck-duck-llama

## Upstream submission is gated on a human, by upstream's rules

`ggml-org/llama.cpp` AGENTS.md lists under "Prohibited AI Usage (results in immediate PR closure)":
AI-written PR descriptions, **commit messages**, or reviewer responses; and **automated commits or PR
submissions** — "may result in contributor ban". CONTRIBUTING.md adds that undisclosed AI use "may result in
your account being permanently banned from contributing".

**Private forks are explicitly exempt**, so everything in this fork is fine as it stands. Only the hop to
upstream is gated. The commit messages currently on these branches were drafted with AI assistance and must
be rewritten by the submitting human, along with the PR description and the required AI-usage disclosure.

`unary-ops-parallel` is rebased onto current upstream master and applies cleanly; the four lines it deletes
are still present there. A full briefing — every claim with the file and line to verify it, the measurements
with their sources, and the reviewer questions to expect — is in the serving tree at
`fleet-0912-ctx/UPSTREAM-HANDOFF.md`.

## Candidate, measured but not yet a branch: GET_ROWS is single-threaded

`ggml_get_n_tasks()` pins `GET_ROWS` and `SET_ROWS` to one thread:

```c
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                n_tasks = 1;
            } break;
```

The stated reason is GPU-offload launch cost, which does not apply to a CPU-only build, and
`ggml_compute_forward_get_rows` **already splits by rows** (`dr = (nr + nth - 1)/nth`) in every type variant.
Same defect shape as `unary-ops-parallel`: the kernel honours ith/nth, the scheduler refuses to supply them.

Measured on Qwen3.8-Flash-Next, where a sparse-attention indexer gathers the whole key cache every token:
`GET_ROWS` moves 8.2 MB per layer in 2.93 ms = **2.80 GB/s, 0.74% of this machine's 381 GB/s**, at **91 ns per
gathered row** -- one DRAM latency with no memory-level parallelism.

Threading it above a row threshold, four arms, greedy output byte-identical in all of them:

| arm | 190 ctx | 28,880 ctx | acceptance | greedy hash |
|---|---:|---:|---:|---|
| production | 24.04 | 16.64 | 76% | `13b7ea22` |
| patched lib, flag off | 24.06 | 16.28 | 76% | `13b7ea22` |
| **threaded above 256 rows** | 24.33 | **18.44** | 76% | `13b7ea22` |

**+12.0%** against the control mean on a 2.2% noise floor, bit-exact. The gain is a 1.80x speedup on the
gather, not the 8-60x a naive thread-count argument predicts, because the gather is DRAM-latency bound and
sixty threads each stalling on their own random 256-byte read do not recover that linearly.

**Not yet a branch, deliberately.** A fourth arm at a 4096-row threshold never completed its long-context
point (killed at 50 min against ~6 for the others, no error in the log), and a higher threshold behaving
worse than both a lower threshold and no threading has no obvious mechanism. Until that is reproduced or
dismissed, the threshold to propose is undetermined. Re-run queued.

## Ready to propose upstream (pending the above)

| branch | change | status |
|---|---|---|
| `unary-ops-parallel` | `ggml_get_n_tasks()` gives 17 of 22 unary ops `n_tasks = 1` while `GELU`/`SILU` in the same switch get `n_threads`. All 17 are `unary_op<op_x>` in `unary-ops.cpp`, reaching `apply_unary_op`, which already splits work with `get_thread_range`. `XIELU` sits on that exact path (`unary_op_functor`) and *is* given `n_threads`, so the row split is proven there; the 17 differ from it only in this switch. A 4-line deletion. (`GELU` and `SILU` have their own implementations in `ops.cpp` and are not evidence about this path — an earlier draft of this note claimed all 22 shared one implementation, which is false.) | **bit-exact**, measured +3.4% on Qwen3.8-Flash-Next and +2.1% on GLM-5.3-Flash, greedy output byte-identical |
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
