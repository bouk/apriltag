---
name: autoresearch
description: Autonomous research loop to optimize apriltag detector performance
user_invocable: true
---

# autoresearch — apriltag performance optimization

This is an autonomous research loop that iteratively modifies the apriltag detector code, benchmarks it, and keeps or discards changes based on whether they improve performance.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files**: Read these files for full context:
   - `apriltag.h` — public API and data structures
   - `apriltag.c` — main detector logic, decoding, threading
   - `apriltag_quad_thresh.c` — quad detection pipeline (threshold, unionfind, clustering, quad fitting)
   - `common/image_u8.h` — image type
   - `common/unionfind.h` — union-find for connected components
   - `common/workerpool.h` — thread pool
   - `common/homography.h` — homography computation
   - `common/g2d.h` — 2D geometry utilities
   - `example/apriltag_demo.c` — the benchmark harness
   - Any other `.c` or `.h` files in `common/` that are relevant to what you want to change
4. **Verify build works**: Run the build command and the benchmark to confirm setup.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good and kick off the experimentation.

## Build & benchmark commands

```bash
# Configure (only needed once or after CMakeLists.txt changes):
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build:
cmake --build build -j8

# Benchmark (run 10 iterations for stable timing):
./build/apriltag_demo -f tagStandard52h13 -a 1 -t 4 -i 10 vide3.jpg
```

## What you CAN modify

Any C source or header file in the repository:
- `apriltag.c` — detector pipeline, decoding, threading
- `apriltag_quad_thresh.c` — quad detection (threshold, unionfind, clustering, quad fitting)
- Files in `common/` — image operations, math, data structures, thread pool
- `CMakeLists.txt` — if you need to add new source files or change compiler flags
- You may create new `.c` / `.h` files if needed for larger architectural changes

## What you CANNOT modify

- `example/apriltag_demo.c` — this is the fixed benchmark harness
- Tag family files (`tag36h11.c`, etc.) — these are generated lookup tables
- Do NOT install new system dependencies

## Goal

**Get the lowest total detection time (in ms) while maintaining correct detections.**

The metric is the total time reported by the timing profile in `apriltag_demo`, measured over 10 iterations (`-i 10`) for stability. Lower is better.

**Correctness constraint**: The baseline run establishes the expected number of detections and hamming distribution. All subsequent experiments must match the baseline exactly — same number of detections, same hamming histogram. If a change causes missed detections or changes the hamming distribution, that change is a **failure** — discard it.

**Performance is the priority.** Unlike a simplicity-first approach, you should pursue larger architectural changes if they yield meaningful speed improvements. SIMD intrinsics, algorithmic improvements, better memory layouts, cache optimization, parallel pipeline changes — all fair game. Write the code in an understandable way, but do not sacrifice performance for simplicity.

**The first run**: Your very first run should always establish the baseline by building and running the code as-is.

## Output format

The demo prints a timing profile like this:

```
 0                             init        0.000000 ms        0.000000 ms
 1                         decimate        0.602000 ms        0.602000 ms
 2                       blur/sharp        0.000000 ms        0.602000 ms
 3                        threshold        0.584000 ms        1.186000 ms
 4                        unionfind        3.741000 ms        4.927000 ms
 5                    make clusters        5.573000 ms       10.500000 ms
 6            fit quads to clusters        6.616000 ms       17.116000 ms
 7                            quads        0.146000 ms       17.262000 ms
 8                decode+refinement        1.501000 ms       18.763000 ms
 9                        reconcile        0.010000 ms       18.773000 ms
10                     debug output        0.000000 ms       18.773000 ms
11                          cleanup        0.015000 ms       18.788000 ms
hamm    63     4     0     0     0     0     0     0     0     0       18.788   285
```

The key metric is the **total time** on the summary line (18.788 ms in this example). Also track detection count (67) and quad count (285).

To extract results from the log:

```bash
# Total time and detections from the last summary line:
tail -1 run.log
# Detailed timing:
grep "ms$" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	time_ms	detections	quads	status	description
```

1. git commit hash (short, 7 chars)
2. total time in ms from the 10-iteration average (e.g. 15.316)
3. number of correct detections (must be 12)
4. number of quads
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	time_ms	detections	quads	status	description
a1b2c3d	18.788	67	285	keep	baseline
b2c3d4e	16.200	67	285	keep	NEON SIMD for threshold
c3d4e5f	17.800	65	250	discard	aggressive quad filter (lost 2 detections)
d4e5f6g	0.000	0	0	crash	bad pointer in unionfind rewrite
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar24`).

LOOP FOREVER:

1. Look at the git state and current results.tsv to understand where you are.
2. Read the relevant source files and identify a performance optimization to try.
3. Make your code changes.
4. `git add` the changed files and `git commit` with a descriptive message.
5. Build: `cmake --build build -j8 > build.log 2>&1`
6. If build fails, read `tail -n 50 build.log`, fix the issue, and rebuild. If you can't fix after a few attempts, discard.
7. Run the benchmark: `./build/apriltag_demo -f tagStandard52h13 -a 1 -t 4 -i 10 vide3.jpg > run.log 2>&1`
8. Read out the results: check `run.log` for the summary line (total time, detection count) and verify all 67 detections are present with correct hamming distribution (63 hamming=0, 4 hamming=1).
9. If the run crashed, run `tail -n 50 run.log` to read the error. Try to fix if it's simple; otherwise discard.
10. Record the results in `results.tsv` (do NOT commit results.tsv — leave it untracked).
11. If total time improved AND detections match baseline (same count, same hamming distribution): **keep** — advance the branch.
12. If time is worse OR detections don't match baseline: **discard** — `git reset --hard HEAD~1` to revert.

**Timeout**: Each build+benchmark cycle should take under 2 minutes. If a run hangs for more than 2 minutes, kill it and treat as failure.

**Crashes**: If a run crashes due to something trivial (typo, missing include), fix and re-run. If the idea is fundamentally broken, discard and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human may be away and expects you to work indefinitely until manually stopped. If you run out of ideas, think harder — re-read the source files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
