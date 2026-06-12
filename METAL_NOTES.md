# Metal GPU campaign (Apple Silicon, M3 Pro)

Port of the GPU front-end to Metal for Apple-Silicon Macs, mirroring the
OpenCL campaign in `GPU_NOTES.md` on the `nuc15-hardware` branch. This
branch combines it with the NEON CPU port (`PERF_NOTES.md`). Measured on an M3 Pro (6P+6E CPU,
18-core GPU, unified memory) over the 133-image `vide_images2` corpus
(3088×2064, tagStandard52h13, `-x 1.0`); per-commit benchmarks live in
`results.tsv` (hyperfine via `bench.sh`), output equivalence is gated by
`check.sh` against `benchmark_results/dets-baseline-macos.tsv`.

## What runs where

| stage | where | notes |
| --- | --- | --- |
| threshold + RLE | GPU | tile 4×4 min/max, 3×3 clamped blur, threshold, per-row RLE into run tables |
| unionfind (CCL) | GPU | run-indexed union-find reproducing `connect_runs_to_prev`, min-root CAS union, label paint |
| make clusters | GPU | boundary emission in legacy (y, x, conn) order, stable radix sort by cluster key, packed `pt_list` arena |
| fit quads | CPU | reads the arena-backed `pt_list`s directly (`do_quad_task` skips the free when `td->metal` is set) |
| decode + refine | CPU | unchanged |

Output is **identical** to the CPU pipeline on the corpus: 4583/4583
detections, ids and hamming exact, max coordinate delta 0.0 px. The GPU
path is opt-in via `APRILTAG_METAL=1` and falls back to the CPU pipeline
when no Metal device is available.

## Headline numbers (results.tsv has the full history)

| config | wall (corpus) | detector ms/img | user CPU s |
| --- | --- | --- | --- |
| CPU-only, 4 threads | 6.19 s | 30.3 | 16.8 |
| CPU-only, 12 threads | 4.45 s | 17.0 | 21.7 |
| Metal pipelined, 4 threads | 4.20 s | 14.5 | 9.0 |
| Metal pipelined, 12 threads | 3.23 s | 7.9 | 11.2 |

1.38× faster wall and 2.15× faster detector than the best CPU config at
12 threads, with ~50% of the CPU time — the freed CPU headroom being the
actual point, as on the NUC15.

## Architecture

One serial compute command buffer encodes the whole front-end
(`frontend_commit`); sequential dispatches in a serial encoder are
ordered, including the indirect dispatch arguments the scan kernels
write, so there are no intermediate host round-trips.
`apriltag_detector_detect_prepare` commits the chain for the upcoming
frame without waiting; `apriltag_metal_clusters` then only waits for the
in-flight work. With the demo's one-image lookahead the GPU runs under
the next JPEG load and the steady-state front-end wait is ~0–3 ms.

The five CPU-visible buffers (image staging, params, state, cluster
arena + offsets) are double-buffered per pipeline slot and the pending
queue is mutex-guarded, so prepare may be issued from another thread
while detect runs. GPU-internal scratch is shared; Metal's hazard
tracking serializes successive command buffers over it.

Clusters land in one GPU arena as packed `pt_list` records — the zarray
handed to `fit_quads` holds pointers into the arena, valid until the
next detect on the same slot.

## Findings (measured)

- **Serial offload breaks even; overlap wins.** Serial Metal front-end
  at 12 threads: 26.3 ms/image vs 17.0 CPU-only. Pipelined via
  `detect_prepare`: 7.9 ms. Same lesson as the NUC15.
- **Power coupling, amplified.** With 12 CPU threads busy, the same GPU
  chain stretches from ~7 ms to 14–22 ms (shared package power). It is
  invisible in the demo because the GPU overlaps the single-threaded
  JPEG decode, not the parallel back-half.
- **Sustained saturation throttles the package.** ~30 s of continuous
  CPU+GPU load trips a throttle that stretches GPU time ~2.4× and wall
  runs from 3.3 s to 6.7–15 s; recovery takes seconds. `bench.sh` sleeps
  3 s between hyperfine runs to keep benchmarks in the fast state.
- **Deeper pipelining bought nothing here.** A demo variant decoding
  JPEGs (and preparing frames) on a worker thread — GPU N+1 under CPU
  back-half N, enabled by the double-buffered slots — measured 3.2–3.4 s
  cool, i.e. equal to the simple lookahead, because the power-coupled
  GPU stretches until the GPU-wait + back-half chain matches the decode
  it was hiding; and with no idle duty cycle the throttle hits within a
  run (15 s, user time doubling as clocks halve). Not adopted; the
  library support remains.
- **The dispatch chain is latency-bound, not ALU-bound.** Rewriting the
  radix sort from 6×5-bit passes (Hillis-Steele [256][32] matrix) to
  4×8-bit passes with simdgroup-ballot ranking cut corpus-mean GPU time
  only 15.5 → 14.8 ms/frame despite removing a third of the sort work.
  Kept for the smaller footprint.
- Apple GPUs have no fp64, so the NUC15's bit-exact GPU quad fitter
  cannot port; `fit_quads` stays on the CPU, where it is the dominant
  detector cost. This branch rebases the Metal work onto the NEON port
  (`faster3`), whose vectorized fit_quads/refine-edges take the Metal
  detector from 14.5 to 13.4 ms/image at 4 threads (fit_quads 11.6 ->
  10.8); at 12 saturated threads the gain drowns in noise (~8 ms either
  way). Still byte-identical: 4583/4583, max delta 0.0 px.

## Environment toggles

- `APRILTAG_METAL=1` — enable the GPU front-end (off by default).
- `APRILTAG_METAL_VALIDATE=1` — run the CPU reference stages each frame
  and byte-compare intermediates (slow; single-threaded use).
- `APRILTAG_METAL_PROF=1` — per-frame encode/wait/GPU times on stderr.

## Gotchas / decisions log

- `pt` coordinates are 2×actual (half-pixel grid); gx/gy ∈ {−255,0,255}.
- Cluster keys pack two 15-bit dense component ids (`comps_cap` ≤ 32768).
- The kernel source is embedded via `xxd` and compiled at
  `apriltag_detector_create`; first frame pays ~60 ms of PSO compilation.
- ~20 of ~2300 clusters per frame differ from the CPU sequence in point
  membership (240/719k points on the probe frame); every detection and
  coordinate still matches the CPU pipeline exactly across the corpus —
  these clusters never survive quad fitting differently. Inherited from
  the original port; documented, not yet root-caused.
- macOS benchmarking: interleave A/B runs and use `bench.sh`'s cool-down;
  single-image `-i N` loops are unstable as the GPU shifts power states.
