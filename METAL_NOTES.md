# Metal GPU detector port (Apple Silicon) — design + progress

Goal: replace every profitable stage of the AprilTag detector with Metal GPU
compute on Apple Silicon (M3 Pro: 6P+6E CPU, 18-core GPU, unified memory),
benchmarked over the 287-image `vide_images/` corpus (3088x2064 grayscale,
tagStandard52h13, `-t N -i 1 -x 1.0`).

## Baselines on this machine (M3 Pro, Release, -mcpu=native)

CPU per-stage (t=12, mean ms/image, stable across reruns; machine noise can
produce 2x outliers — always interleave A/B for comparisons):

| stage          | t=12  |
|----------------|-------|
| threshold      | 2.02  |
| unionfind      | 1.81  |
| make clusters  | 5.29  |
| fit quads      | 6.04  |
| decode+refine  | 1.03  |
| total          | 16.22 |

t=4: 30.1 ms. Reference detections: `benchmark_results/dets-macos-cpu.tsv`
(9784 dets, byte-identical for t=4..12). Wall for 287 images t=12: ~10.4 s
(includes JPEG decode, single process).

Fixed en route: `ctasks[16]` stack overflow in gradient_clusters when
nthreads > 4 (now VLA sized 4*nthreads+1). ASan-clean at t=6/12 now.

## Architecture (milestone A)

GPU front-end replaces threshold + unionfind + make-clusters; CPU keeps
fit_quads + decode (they only need `im` + clusters). Hook at the top of
`apriltag_quad_thresh()`: if `td->metal` is non-NULL, get clusters from
`apriltag_metal_clusters()` instead of threshold/CCL/gradient_clusters.
Enabled via env `APRILTAG_METAL=1` (detector_create tries to init; falls
back to CPU silently if no device). `APRILTAG_METAL_VALIDATE=1` runs the
CPU reference stages too and byte-compares intermediates per frame.

Kernel chain (one shared-memory MTLCommandBuffer per frame, bounds-checked
writes + overflow flag -> host grows buffers and retries the frame):

1. tile 4x4 min/max -> 2 (tw x th) u8 planes
2. 3x3 clamped dilate/erode blur of the tile planes
3. threshold -> threshim (0/255/127) incl. right-edge cols using last
   tile, bottom partial rows (no 127 there)
4. RLE per row over x in [0, w-2], 127 runs skipped: count -> scan ->
   fill {start u16, end u16, v u8 (pad to 6B)} at row_off[y]
5. run-indexed union-find (parent u32), nodes = runs + per-row virtual
   last-column node (vcol_base + y, vcol_base = total runs):
   connect run pairs of adjacent rows per connect_runs_to_prev rules:
   - same v, vertical overlap clamped to x>=1
   - v==255 also diagonal +-1 contact (same clamps)
   - white run ending at w-2 + buf[y-1][w-1]==255 && buf[y-1][w-2]!=255
     -> union with vcol node (y-1)
   GPU union: CAS-hooking, min-root convention; then flatten; component
   pixel counts via atomic add (run length; vcol nodes +1)
6. label image u32 (w x h): INVALID everywhere, then per usable run
   (count >= min_cluster_pixels) paint root; vcol pixels painted too
7. boundary emission, per pixel x in [1, w-2], y in [0, h-2], conn order
   (1,0),(0,1),(-1,1),(1,1); emit iff v0+v1==255 && both labels valid;
   (-1,1) suppressed iff pixel x-1 emitted its (1,1) (pure local recompute:
   labels(x-1,y), labels(x,y+1), values). Points: x=2x+dx, y=2y+dy,
   gradient (dx*(v1-v0), dy*(v1-v0)). Record = {key u64 = clusterid,
   pt 8B} where clusterid = minmax pair of root ids exactly like CPU.
   Two-pass: per-threadgroup counts -> scan -> ordered scatter (preserves
   the legacy (y, x, conn) global order -> per-cluster point order legacy)
8. stable LSD radix sort of records by 40-bit key (reps < 2^20-ish)
9. cluster boundaries -> per-cluster {offset,size}; arena buffer written
   as packed pt_list {i32 size, i32 pad, pts[]}; CPU builds zarray of
   pt_list* pointing into the arena (do_quad_task must NOT free those:
   guarded by td->metal flag)

Equivalence claim to verify: partition + per-cluster point sequences are
exactly CPU's; only inter-cluster zarray order differs (CPU: hash-bucket
then id; GPU: id asc). t=4..12 CPU runs are byte-identical despite
different chunking, suggesting order-insensitivity downstream; verified
empirically against dets-macos-cpu.tsv (epsilon: ids/hamming exact,
coords <= 0.1 px). If order matters, sort clusters by (u64hash_2(id) &
(nclustermap-1), id) instead.

## Files

- `metal/apriltag_kernels.metal` — all GPU kernels
- `apriltag_metal.h` / `apriltag_metal.m` — C API + ObjC host (ARC)
- `apriltag_quad_internal.h` — shared struct pt/pt_list/internal decls
- CMake: APRILTAG_METAL option (default ON on APPLE): compiles metallib
  via xcrun at build time, embeds bytes via generated C array; links
  Metal/Foundation frameworks
- Detector integration: `void *metal` field appended to apriltag_detector

## Milestones / status

- [x] M0: CPU baseline + thread-scaling fix
- [ ] M-A: GPU threshold+CCL+clusters, CPU fit_quads+decode, dets-equal
- [ ] M-B: fit_quads on GPU (fp32 — epsilon drift expected, gate 0.1px)
- [ ] M-C: decode placement decision, frame pipelining, max throughput
- [ ] Report

## Gotchas / decisions log

- The Bash tool shell here is zsh: `$var` does NOT word-split; `echo ===`
  eats `===`. Use globs directly.
- benchmark.sh/ab.sh are Linux (taskset/nproc); use direct runs + the awk
  one-liners on macOS. Warmup run first; interleave A/B; expect occasional
  2x outliers (background load).
- timing tsv: stage rows per image; `awk -F'\t' 'NR>1{s+=$5;img[$1]=1}
  END{c=0;for(i in img)c++;print s/c}'` = detector ms/image.
- pt coordinates are 2*actual (half-pixel grid); gx/gy in {-255,0,255}.
- uf size semantics: size[root]+1 == component pixel count.
- Metal: no 64-bit CAS; use u32 parents (runs < 2^32). newBufferWithBytesNoCopy
  needs 16KB-page-aligned base + length; image_u8 allocs are not page-aligned
  -> copy in (6.4MB ~0.2ms) for now, or patch allocator later.
