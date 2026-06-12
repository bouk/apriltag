// AprilTag detector GPU front-end: threshold -> RLE -> run-indexed
// connected components -> label paint -> boundary point emission ->
// stable group-by-cluster. Mirrors the CPU pipeline in
// apriltag_quad_thresh.c exactly (integer arithmetic throughout); see
// METAL_NOTES.md for the semantics contract.
#include <metal_stdlib>
using namespace metal;

#define INVALID_LABEL 0xFFFFFFFFu

// overflow flag bits in ATState.overflow
#define OV_RUNS     1u
#define OV_RECORDS  2u
#define OV_COMPS    4u
#define OV_CLUSTERS 8u

struct ATParams {
    uint w, h, s;            // image width/height/stride
    uint tw, th;             // tile grid
    uint min_wb_diff;        // qtp.min_white_black_diff
    uint min_cluster_pixels; // component-size gate
    uint runs_cap;
    uint records_cap;
    uint comps_cap;          // must be <= 32768 (pair key packs 2x15 bits)
    uint clusters_cap;
    uint emit_blocks_x;      // ceil(w/256)
};

struct ATState {
    uint nruns;       // row_off[h]
    uint nnodes;      // nruns + h (virtual last-column node per row)
    uint ncomp;       // usable (size-gated) components
    uint nrecords;
    uint nclusters;
    atomic_uint overflow;
    // indirect dispatch args (threadgroup counts)
    uint runs_tg[3];
    uint nodes_tg[3];
    uint rec_tg[3];   // ceil(nrecords/256)
    uint radix_tg[3]; // ceil(nrecords/RADIX_BLOCK)
    uint cl_tg[3];    // ceil(nclusters/256)
    uint quad_tg[3];  // nclusters (one threadgroup per cluster)
};

// 8-byte GPU run record (CPU's row_run + row index)
struct gpu_run {
    ushort start, end; // inclusive
    ushort y;
    uchar v;
    uchar pad;
};

// packed boundary point, identical layout to struct pt
struct gpu_pt {
    ushort x, y;
    short gx, gy;
};

///////////////////////////////////////////////////////////////////////////
// stage 1: threshold

kernel void k_minmax(device const uchar *im [[buffer(0)]],
                     device uchar *tmin [[buffer(1)]],
                     device uchar *tmax [[buffer(2)]],
                     constant ATParams &p [[buffer(3)]],
                     uint2 g [[thread_position_in_grid]])
{
    if (g.x >= p.tw || g.y >= p.th)
        return;
    uchar mn = 255, mx = 0;
    for (uint dy = 0; dy < 4; dy++) {
        uint off = (g.y*4 + dy)*p.s + g.x*4;
        for (uint dx = 0; dx < 4; dx++) {
            uchar v = im[off + dx];
            mn = min(mn, v);
            mx = max(mx, v);
        }
    }
    tmin[g.y*p.tw + g.x] = mn;
    tmax[g.y*p.tw + g.x] = mx;
}

kernel void k_blur(device const uchar *tmin [[buffer(0)]],
                   device const uchar *tmax [[buffer(1)]],
                   device uchar *bmin [[buffer(2)]],
                   device uchar *bmax [[buffer(3)]],
                   constant ATParams &p [[buffer(4)]],
                   uint2 g [[thread_position_in_grid]])
{
    if (g.x >= p.tw || g.y >= p.th)
        return;
    uchar mn = 255, mx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        int ty = (int)g.y + dy;
        if (ty < 0 || ty >= (int)p.th)
            continue;
        for (int dx = -1; dx <= 1; dx++) {
            int tx = (int)g.x + dx;
            if (tx < 0 || tx >= (int)p.tw)
                continue;
            mn = min(mn, tmin[ty*p.tw + tx]);
            mx = max(mx, tmax[ty*p.tw + tx]);
        }
    }
    bmin[g.y*p.tw + g.x] = mn;
    bmax[g.y*p.tw + g.x] = mx;
}

// one thread per pixel; replicates the CPU edge semantics:
// - full-tile pixels: 127 on low contrast, else 0/255
// - right-edge / bottom-tail pixels: clamped tile, always 0/255
kernel void k_threshold(device const uchar *im [[buffer(0)]],
                        device const uchar *bmin [[buffer(1)]],
                        device const uchar *bmax [[buffer(2)]],
                        device uchar *out [[buffer(3)]],
                        constant ATParams &p [[buffer(4)]],
                        uint2 g [[thread_position_in_grid]])
{
    if (g.x >= p.w || g.y >= p.h)
        return;
    uint tx = min(g.x/4, p.tw - 1);
    uint ty = min(g.y/4, p.th - 1);
    int mn = bmin[ty*p.tw + tx];
    int mx = bmax[ty*p.tw + tx];
    bool in_tiles = (g.x < p.tw*4) && (g.y < p.th*4);
    uchar v = im[g.y*p.s + g.x];
    uchar r;
    if (in_tiles && (mx - mn < (int)p.min_wb_diff)) {
        r = 127;
    } else {
        int thresh = mn + (mx - mn)/2;
        r = (v > (uchar)thresh) ? 255 : 0;
    }
    out[g.y*p.s + g.x] = r;
}

///////////////////////////////////////////////////////////////////////////
// stage 2: RLE (one thread per row; rows are independent)

// count runs in row y over x in [0, w-2], skipping 127 spans
kernel void k_rle_count(device const uchar *t [[buffer(0)]],
                        device uint *row_off [[buffer(1)]],
                        constant ATParams &p [[buffer(2)]],
                        uint y [[thread_position_in_grid]])
{
    if (y >= p.h)
        return;
    device const uchar *row = t + y*p.s;
    int xmax = (int)p.w - 2;
    uint count = (row[0] != 127) ? 1u : 0u;
    for (int x = 1; x <= xmax; x++)
        count += (row[x] != row[x-1] && row[x] != 127) ? 1u : 0u;
    row_off[y + 1] = count;
}

// single-threadgroup exclusive scan of row counts -> row offsets;
// thread 0 finalizes state (nruns, nnodes, indirect args, overflow)
kernel void k_scan_rows(device uint *row_off [[buffer(0)]],
                        device ATState *st [[buffer(1)]],
                        constant ATParams &p [[buffer(2)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tsz [[threads_per_threadgroup]])
{
    threadgroup uint partials[256];
    threadgroup uint carry;
    uint n = p.h; // entries row_off[1..h]
    uint chunk = (n + tsz - 1) / tsz;
    uint lo = 1 + tid*chunk;
    uint hi = min(1 + (tid + 1)*chunk, n + 1);

    uint sum = 0;
    for (uint i = lo; i < hi; i++)
        sum += row_off[i];
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint acc = 0;
        for (uint i = 0; i < tsz; i++) {
            uint v = partials[i];
            partials[i] = acc;
            acc += v;
        }
        carry = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // inclusive prefix: row_off[i] = total runs of rows 0..i-1
    uint acc = partials[tid];
    for (uint i = lo; i < hi; i++) {
        acc += row_off[i];
        row_off[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        row_off[0] = 0;
        uint nruns = carry;
        st->nruns = nruns;
        st->nnodes = nruns + p.h;
        if (nruns > p.runs_cap)
            atomic_fetch_or_explicit(&st->overflow, OV_RUNS, memory_order_relaxed);
        st->runs_tg[0] = (nruns + 255) / 256;
        st->runs_tg[1] = 1;
        st->runs_tg[2] = 1;
        st->nodes_tg[0] = (nruns + p.h + 255) / 256;
        st->nodes_tg[1] = 1;
        st->nodes_tg[2] = 1;
    }
}

kernel void k_rle_fill(device const uchar *t [[buffer(0)]],
                       device const uint *row_off [[buffer(1)]],
                       device gpu_run *runs [[buffer(2)]],
                       constant ATParams &p [[buffer(3)]],
                       uint y [[thread_position_in_grid]])
{
    if (y >= p.h)
        return;
    device const uchar *row = t + y*p.s;
    uint o = row_off[y];
    if (o >= p.runs_cap)
        return; // overflow already flagged
    int xmax = (int)p.w - 2;
    int start = 0;
    uchar v = row[0];
    for (int x = 1; x <= xmax; x++) {
        if (row[x] != v) {
            if (v != 127 && o < p.runs_cap) {
                runs[o].start = (ushort)start;
                runs[o].end = (ushort)(x - 1);
                runs[o].y = (ushort)y;
                runs[o].v = v;
                runs[o].pad = 0;
                o++;
            }
            start = x;
            v = row[x];
        }
    }
    if (v != 127 && o < p.runs_cap) {
        runs[o].start = (ushort)start;
        runs[o].end = (ushort)xmax;
        runs[o].y = (ushort)y;
        runs[o].v = v;
        runs[o].pad = 0;
    }
}

///////////////////////////////////////////////////////////////////////////
// stage 3: union-find over runs

static inline uint uf_find(device atomic_uint *parent, uint i)
{
#ifdef AT_UF_PATH_HALVING
    uint p = atomic_load_explicit(&parent[i], memory_order_relaxed);
    while (p != i) {
        uint pp = atomic_load_explicit(&parent[p], memory_order_relaxed);
        if (pp != p) {
            // path halving; benign race
            atomic_store_explicit(&parent[i], pp, memory_order_relaxed);
        }
        i = p;
        p = pp;
    }
    return i;
#else
    uint p = atomic_load_explicit(&parent[i], memory_order_relaxed);
    while (p != i) {
        i = p;
        p = atomic_load_explicit(&parent[i], memory_order_relaxed);
    }
    return i;
#endif
}

static inline void uf_union(device atomic_uint *parent, uint a, uint b)
{
    for (;;) {
        a = uf_find(parent, a);
        b = uf_find(parent, b);
        if (a == b)
            return;
        uint hi = max(a, b), lo = min(a, b);
        uint expected = hi;
        // weak CAS can fail spuriously; retry from the (possibly updated)
        // roots either way
        if (atomic_compare_exchange_weak_explicit(&parent[hi], &expected, lo,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed))
            return;
        a = lo;
        b = hi;
    }
}

kernel void k_uf_init(device atomic_uint *parent [[buffer(0)]],
                      device const ATState *st [[buffer(1)]],
                      uint i [[thread_position_in_grid]])
{
    if (i < st->nnodes)
        atomic_store_explicit(&parent[i], i, memory_order_relaxed);
}

// one thread per run: connect to the previous row's runs exactly per
// connect_runs_to_prev() (vertical contact clamped to x>=1; white also
// diagonal; white run ending at w-2 may connect the previous row's
// virtual last-column node)
kernel void k_uf_connect(device atomic_uint *parent [[buffer(0)]],
                         device const gpu_run *runs [[buffer(1)]],
                         device const uint *row_off [[buffer(2)]],
                         device const uchar *t [[buffer(3)]],
                         device const ATState *st [[buffer(4)]],
                         constant ATParams &p [[buffer(5)]],
                         uint ri [[thread_position_in_grid]])
{
    if (ri >= st->nruns)
        return;
    gpu_run r = runs[ri];
    uint y = r.y;
    if (y == 0)
        return;

    int a0 = r.start, a1 = r.end;
    uchar v = r.v;
    uint prev_lo = row_off[y - 1], prev_hi = row_off[y];

    // first candidate: lowest k with prev[k].end + 1 >= a0 (ends ascend)
    uint klo = prev_lo, khi = prev_hi;
    while (klo < khi) {
        uint mid = (klo + khi)/2;
        if ((int)runs[mid].end + 1 < a0)
            klo = mid + 1;
        else
            khi = mid;
    }

    for (uint k = klo; k < prev_hi && (int)runs[k].start <= a1 + 1; k++) {
        if (runs[k].v != v)
            continue;
        int b0 = runs[k].start, b1 = runs[k].end;

        int lo = max(max(a0, b0), 1);
        int hi = min(a1, b1);
        if (lo <= hi) {
            uf_union(parent, ri, k);
        } else if (v == 255) {
            int xl = max(max(a0, b0 + 1), 1);
            if (xl <= min(a1, b1 + 1)) {
                uf_union(parent, ri, k);
            } else {
                int xr = max(max(a0, b0 - 1), 1);
                if (xr <= min(a1, b1 - 1)) {
                    uf_union(parent, ri, k);
                }
            }
        }
    }

    if (v == 255 && a1 == (int)p.w - 2 &&
        t[(y - 1)*p.s + (p.w - 1)] == 255 &&
        t[(y - 1)*p.s + (p.w - 2)] != 255) {
        uf_union(parent, ri, st->nruns + (y - 1));
    }
}

kernel void k_uf_flatten(device atomic_uint *parent [[buffer(0)]],
                         device const ATState *st [[buffer(1)]],
                         uint i [[thread_position_in_grid]])
{
    if (i >= st->nnodes)
        return;
    uint root = uf_find(parent, i);
    atomic_store_explicit(&parent[i], root, memory_order_relaxed);
}

// component pixel counts at roots: runs add their length, virtual
// last-column nodes add 1
kernel void k_comp_count(device const uint *parent [[buffer(0)]],
                         device atomic_uint *count [[buffer(1)]],
                         device const gpu_run *runs [[buffer(2)]],
                         device const ATState *st [[buffer(3)]],
                         uint i [[thread_position_in_grid]])
{
    if (i >= st->nnodes)
        return;
    uint add = 1;
    if (i < st->nruns)
        add = (uint)(runs[i].end - runs[i].start + 1);
    atomic_fetch_add_explicit(&count[parent[i]], add, memory_order_relaxed);
}

kernel void k_zero_u32(device uint *buf [[buffer(0)]],
                       device const ATState *st [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    if (i < st->nnodes)
        buf[i] = 0;
}

// flag usable roots (component pixel count >= min_cluster_pixels)
kernel void k_root_flag(device const uint *parent [[buffer(0)]],
                        device const uint *count [[buffer(1)]],
                        device uint *flag [[buffer(2)]],
                        device const ATState *st [[buffer(3)]],
                        constant ATParams &p [[buffer(4)]],
                        uint i [[thread_position_in_grid]])
{
    if (i >= st->nnodes)
        return;
    flag[i] = (parent[i] == i && count[i] >= p.min_cluster_pixels) ? 1u : 0u;
}

// single-threadgroup exclusive scan over node flags -> dense component ids
// (deterministic: rank among usable roots in node-index order)
kernel void k_scan_nodes(device uint *flag_then_dense [[buffer(0)]],
                         device ATState *st [[buffer(1)]],
                         constant ATParams &p [[buffer(2)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tsz [[threads_per_threadgroup]])
{
    threadgroup uint partials[256];
    threadgroup uint total;
    uint n = st->nnodes;
    uint chunk = (n + tsz - 1) / tsz;
    uint lo = tid*chunk;
    uint hi = min(lo + chunk, n);

    uint sum = 0;
    for (uint i = lo; i < hi; i++)
        sum += flag_then_dense[i];
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint acc = 0;
        for (uint i = 0; i < tsz; i++) {
            uint v = partials[i];
            partials[i] = acc;
            acc += v;
        }
        total = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint acc = partials[tid];
    for (uint i = lo; i < hi; i++) {
        uint v = flag_then_dense[i];
        flag_then_dense[i] = acc;
        acc += v;
    }

    if (tid == 0) {
        st->ncomp = total;
        if (total > p.comps_cap)
            atomic_fetch_or_explicit(&st->overflow, OV_COMPS, memory_order_relaxed);
    }
}

// per-node label: dense id of its root when usable, else INVALID
kernel void k_node_label(device const uint *parent [[buffer(0)]],
                         device const uint *count [[buffer(1)]],
                         device const uint *dense [[buffer(2)]],
                         device uint *node_label [[buffer(3)]],
                         device const ATState *st [[buffer(4)]],
                         constant ATParams &p [[buffer(5)]],
                         uint i [[thread_position_in_grid]])
{
    if (i >= st->nnodes)
        return;
    uint root = parent[i];
    node_label[i] = (count[root] >= p.min_cluster_pixels) ? dense[root] : INVALID_LABEL;
}

kernel void k_label_clear(device uint *labels [[buffer(0)]],
                          constant ATParams &p [[buffer(1)]],
                          uint i [[thread_position_in_grid]])
{
    if (i < p.w*p.h)
        labels[i] = INVALID_LABEL;
}

// paint per-pixel labels: each run paints its span; each virtual node
// paints its last-column pixel
kernel void k_label_paint(device const uint *node_label [[buffer(0)]],
                          device const gpu_run *runs [[buffer(1)]],
                          device uint *labels [[buffer(2)]],
                          device const ATState *st [[buffer(3)]],
                          constant ATParams &p [[buffer(4)]],
                          uint i [[thread_position_in_grid]])
{
    if (i >= st->nnodes)
        return;
    uint lbl = node_label[i];
    if (lbl == INVALID_LABEL)
        return;
    if (i < st->nruns) {
        gpu_run r = runs[i];
        device uint *row = labels + (uint)r.y*p.w;
        for (uint x = r.start; x <= (uint)r.end; x++)
            row[x] = lbl;
    } else {
        uint y = i - st->nruns;
        labels[y*p.w + (p.w - 1)] = lbl;
    }
}

///////////////////////////////////////////////////////////////////////////
// stage 4: boundary point emission
//
// Per pixel (x in [1, w-2], y in [0, h-2]), connections in legacy order
// (1,0), (0,1), (-1,1), (1,1). A connection fires iff v0 + v1 == 255 and
// both pixels carry valid labels. The (-1,1) connection is suppressed when
// pixel x-1 fired its (1,1) (recomputed locally, no serial state).

static inline uint emit_mask(device const uchar *t, device const uint *labels,
                             constant ATParams &p, uint x, uint y)
{
    uchar v0 = t[y*p.s + x];
    if (v0 == 127)
        return 0;
    uint l0 = labels[y*p.w + x];
    if (l0 == INVALID_LABEL)
        return 0;

    uint mask = 0;
    // (1,0)
    {
        uchar v1 = t[y*p.s + x + 1];
        if ((uint)v0 + v1 == 255 && labels[y*p.w + x + 1] != INVALID_LABEL)
            mask |= 1;
    }
    // (0,1)
    {
        uchar v1 = t[(y + 1)*p.s + x];
        if ((uint)v0 + v1 == 255 && labels[(y + 1)*p.w + x] != INVALID_LABEL)
            mask |= 2;
    }
    // (-1,1), unless the left neighbor fired its (1,1)
    {
        bool suppressed = false;
        if (x >= 2) {
            uchar vp = t[y*p.s + x - 1];
            uchar vd = t[(y + 1)*p.s + x];
            suppressed = (vp != 127) && ((uint)vp + vd == 255) &&
                         labels[y*p.w + x - 1] != INVALID_LABEL &&
                         labels[(y + 1)*p.w + x] != INVALID_LABEL;
        }
        if (!suppressed) {
            uchar v1 = t[(y + 1)*p.s + x - 1];
            if ((uint)v0 + v1 == 255 && labels[(y + 1)*p.w + x - 1] != INVALID_LABEL)
                mask |= 4;
        }
    }
    // (1,1)
    {
        uchar v1 = t[(y + 1)*p.s + x + 1];
        if ((uint)v0 + v1 == 255 && labels[(y + 1)*p.w + x + 1] != INVALID_LABEL)
            mask |= 8;
    }
    return mask;
}

// per-block point counts; one threadgroup covers 256 consecutive pixels
// of one row
kernel void k_emit_count(device const uchar *t [[buffer(0)]],
                         device const uint *labels [[buffer(1)]],
                         device uint *block_counts [[buffer(2)]],
                         constant ATParams &p [[buffer(3)]],
                         uint2 gid [[threadgroup_position_in_grid]],
                         uint2 tid2 [[thread_position_in_threadgroup]])
{
    uint tid = tid2.x;
    threadgroup atomic_uint tg_count;
    if (tid == 0)
        atomic_store_explicit(&tg_count, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint x = gid.x*256 + tid;
    uint y = gid.y;
    uint c = 0;
    if (x >= 1 && x <= p.w - 2 && y <= p.h - 2)
        c = popcount(emit_mask(t, labels, p, x, y));
    if (c)
        atomic_fetch_add_explicit(&tg_count, c, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0)
        block_counts[gid.y*p.emit_blocks_x + gid.x] =
            atomic_load_explicit(&tg_count, memory_order_relaxed);
}

// single-threadgroup exclusive scan of block counts; finalizes nrecords
// and the record-indexed indirect args
#define RADIX_BLOCK 1024u
kernel void k_scan_blocks(device uint *block_counts [[buffer(0)]],
                          device ATState *st [[buffer(1)]],
                          constant ATParams &p [[buffer(2)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tsz [[threads_per_threadgroup]])
{
    threadgroup uint partials[256];
    threadgroup uint total;
    uint n = p.emit_blocks_x * p.h;
    uint chunk = (n + tsz - 1) / tsz;
    uint lo = tid*chunk;
    uint hi = min(lo + chunk, n);

    uint sum = 0;
    for (uint i = lo; i < hi; i++)
        sum += block_counts[i];
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint acc = 0;
        for (uint i = 0; i < tsz; i++) {
            uint v = partials[i];
            partials[i] = acc;
            acc += v;
        }
        total = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint acc = partials[tid];
    for (uint i = lo; i < hi; i++) {
        uint v = block_counts[i];
        block_counts[i] = acc;
        acc += v;
    }

    if (tid == 0) {
        st->nrecords = total;
        if (total > p.records_cap)
            atomic_fetch_or_explicit(&st->overflow, OV_RECORDS, memory_order_relaxed);
        st->rec_tg[0] = (total + 255)/256;
        st->rec_tg[1] = 1;
        st->rec_tg[2] = 1;
        st->radix_tg[0] = (total + RADIX_BLOCK - 1)/RADIX_BLOCK;
        st->radix_tg[1] = 1;
        st->radix_tg[2] = 1;
    }
}

// ordered scatter: same traversal as k_emit_count, but writes
// {key, pt} records at exact offsets, preserving the legacy
// (y, x, conn) order
kernel void k_emit_scatter(device const uchar *t [[buffer(0)]],
                           device const uint *labels [[buffer(1)]],
                           device const uint *block_offsets [[buffer(2)]],
                           device uint *keys [[buffer(3)]],
                           device gpu_pt *pts [[buffer(4)]],
                           constant ATParams &p [[buffer(5)]],
                           uint2 gid [[threadgroup_position_in_grid]],
                           uint2 tid2 [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]])
{
    uint tid = tid2.x;
    threadgroup uint sg_totals[8];

    uint x = gid.x*256 + tid;
    uint y = gid.y;
    uint mask = 0;
    if (x >= 1 && x <= p.w - 2 && y <= p.h - 2)
        mask = emit_mask(t, labels, p, x, y);
    uint c = popcount(mask);

    // threadgroup-ordered exclusive prefix of c (simd, then simdgroups)
    uint pre = simd_prefix_exclusive_sum(c);
    if (lane == 31)
        sg_totals[sg] = pre + c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint v = (lane < 8) ? sg_totals[lane] : 0;
        uint pv = simd_prefix_exclusive_sum(v);
        if (lane < 8)
            sg_totals[lane] = pv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = block_offsets[gid.y*p.emit_blocks_x + gid.x] + sg_totals[sg] + pre;

    if (!mask || base >= p.records_cap)
        return;

    uint l0 = labels[y*p.w + x];
    uint o = base;
    // conn order: (1,0), (0,1), (-1,1), (1,1)
    const int dxs[4] = {1, 0, -1, 1};
    const int dys[4] = {0, 1, 1, 1};
    uchar v0 = t[y*p.s + x];
    int vdiff_white = 255 - 2*(int)v0; // v1 - v0 for opposite pair
    for (uint conn = 0; conn < 4; conn++) {
        if (!(mask & (1u << conn)))
            continue;
        int dx = dxs[conn], dy = dys[conn];
        uint l1 = labels[(y + dy)*p.w + (x + dx)];
        uint kmin = min(l0, l1), kmax = max(l0, l1);
        uint key = (kmin << 15) | kmax; // comps_cap <= 32768
        if (o < p.records_cap) {
            keys[o] = key;
            gpu_pt q;
            q.x = (ushort)(2*x + dx);
            q.y = (ushort)(2*y + dy);
            q.gx = (short)(dx*vdiff_white);
            q.gy = (short)(dy*vdiff_white);
            pts[o] = q;
        }
        o++;
    }
}

///////////////////////////////////////////////////////////////////////////
// stage 5: stable LSD radix sort of records by 30-bit key, 8 bits/pass.
// Within a block the 1024 elements are processed in 4 strided rounds
// (idx = block*1024 + round*256 + tid), so lane order == index order and
// the simdgroup-ballot ranking below is stable.

#define RADIX_BITS 8u
#define RADIX_BINS 256u
#define RADIX_EPT 4u // rounds per block; 256 threads -> 1024 per block

kernel void k_radix_hist(device const uint *keys [[buffer(0)]],
                         device uint *hist [[buffer(1)]], // [bin][block]
                         device const ATState *st [[buffer(2)]],
                         constant uint &shift [[buffer(3)]],
                         uint b [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint local_hist[RADIX_BINS];
    atomic_store_explicit(&local_hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n = st->nrecords;
    for (uint r = 0; r < RADIX_EPT; r++) {
        uint idx = b*RADIX_BLOCK + r*256 + tid;
        if (idx < n) {
            uint d = (keys[idx] >> shift) & (RADIX_BINS - 1);
            atomic_fetch_add_explicit(&local_hist[d], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nblocks = st->radix_tg[0];
    hist[tid*nblocks + b] =
        atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
}

kernel void k_radix_scan(device uint *hist [[buffer(0)]],
                         device const ATState *st [[buffer(1)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tsz [[threads_per_threadgroup]])
{
    threadgroup uint partials[256];
    uint n = RADIX_BINS * st->radix_tg[0];
    uint chunk = (n + tsz - 1) / tsz;
    uint lo = tid*chunk;
    uint hi = min(lo + chunk, n);

    uint sum = 0;
    for (uint i = lo; i < hi; i++)
        sum += hist[i];
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint acc = 0;
        for (uint i = 0; i < tsz; i++) {
            uint v = partials[i];
            partials[i] = acc;
            acc += v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint acc = partials[tid];
    for (uint i = lo; i < hi; i++) {
        uint v = hist[i];
        hist[i] = acc;
        acc += v;
    }
}

// stable scatter: ranks within each round via simdgroup digit-match
// ballots, across simdgroups via per-bin leader counts, across rounds via
// a running per-bin total
kernel void k_radix_scatter(device const uint *keys_in [[buffer(0)]],
                            device const gpu_pt *pts_in [[buffer(1)]],
                            device uint *keys_out [[buffer(2)]],
                            device gpu_pt *pts_out [[buffer(3)]],
                            device const uint *hist [[buffer(4)]],
                            device const ATState *st [[buffer(5)]],
                            constant uint &shift [[buffer(6)]],
                            uint b [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint running[RADIX_BINS];   // placed in earlier rounds
    threadgroup ushort sg_cnt[8][RADIX_BINS]; // per-simdgroup counts, this round
    uint n = st->nrecords;
    uint nblocks = st->radix_tg[0];

    running[tid] = 0;

    for (uint r = 0; r < RADIX_EPT; r++) {
        // zero this round's counts (each thread owns one bin column)
        for (uint s = 0; s < 8; s++)
            sg_cnt[s][tid] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint idx = b*RADIX_BLOCK + r*256 + tid;
        bool valid = idx < n;
        uint key = 0, d = 0;
        if (valid) {
            key = keys_in[idx];
            d = (key >> shift) & (RADIX_BINS - 1);
        }

        // mask of lanes in this simdgroup carrying the same digit
        uint match = valid ? 0xFFFFFFFFu : 0u;
        for (uint bit = 0; bit < RADIX_BITS; bit++) {
            uint bal = (uint)(simd_vote::vote_t)simd_ballot((d >> bit) & 1);
            match &= ((d >> bit) & 1) ? bal : ~bal;
        }
        match &= (uint)(simd_vote::vote_t)simd_ballot(valid);

        uint rank_in_sg = popcount(match & ((1u << lane) - 1u));
        if (valid && rank_in_sg == 0) // one leader per distinct digit
            sg_cnt[sg][d] = (ushort)popcount(match);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint prior_sgs = 0;
            for (uint s = 0; s < sg; s++)
                prior_sgs += sg_cnt[s][d];
            uint pos = hist[d*nblocks + b] + running[d] + prior_sgs + rank_in_sg;
            keys_out[pos] = key;
            pts_out[pos] = pts_in[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // fold this round's counts into the running per-bin totals
        uint round_total = 0;
        for (uint s = 0; s < 8; s++)
            round_total += sg_cnt[s][tid];
        running[tid] += round_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

///////////////////////////////////////////////////////////////////////////
// stage 6: cluster bounds + arena layout
//
// Single threadgroup; three sequential phases over the sorted records:
// 1) flag cluster starts, 2) scan flags -> cluster index per record +
// starts[], 3) compute arena offsets (8-byte header + 8 bytes per point).

kernel void k_cluster_bounds(device const uint *keys [[buffer(0)]],
                             device uint *c_of [[buffer(1)]],
                             device uint *starts [[buffer(2)]],
                             device uint *arena_off [[buffer(3)]],
                             device ATState *st [[buffer(4)]],
                             constant ATParams &p [[buffer(5)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tsz [[threads_per_threadgroup]])
{
    threadgroup uint partials[256];
    threadgroup uint total;
    uint n = st->nrecords;
    uint chunk = (n + tsz - 1) / tsz;
    uint lo = tid*chunk;
    uint hi = min(lo + chunk, n);

    // phase 1+2 fused: each thread serially scans its chunk twice
    uint sum = 0;
    for (uint i = lo; i < hi; i++)
        sum += (i == 0 || keys[i] != keys[i-1]) ? 1u : 0u;
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint acc = 0;
        for (uint i = 0; i < tsz; i++) {
            uint v = partials[i];
            partials[i] = acc;
            acc += v;
        }
        total = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cidx = partials[tid];
    for (uint i = lo; i < hi; i++) {
        if (i == 0 || keys[i] != keys[i-1]) {
            if (cidx < p.clusters_cap)
                starts[cidx] = i;
            cidx++;
        }
        c_of[i] = cidx - 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ncl = total;
    if (tid == 0) {
        st->nclusters = ncl;
        if (ncl > p.clusters_cap)
            atomic_fetch_or_explicit(&st->overflow, OV_CLUSTERS, memory_order_relaxed);
        st->cl_tg[0] = (ncl + 255)/256;
        st->cl_tg[1] = 1;
        st->cl_tg[2] = 1;
        st->quad_tg[0] = min(ncl, p.clusters_cap);
        st->quad_tg[1] = 1;
        st->quad_tg[2] = 1;
        starts[min(ncl, p.clusters_cap)] = n; // sentinel for size computation
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // arena offset for cluster c = 8*c + 8*starts[c]
    uint ncl_c = min(ncl, p.clusters_cap);
    uint chunk2 = (ncl_c + tsz - 1) / tsz;
    uint lo2 = tid*chunk2;
    uint hi2 = min(lo2 + chunk2, ncl_c);
    for (uint c = lo2; c < hi2; c++)
        arena_off[c] = 8*c + 8*starts[c];
}

// per-cluster header {size, pad}
kernel void k_arena_headers(device const uint *starts [[buffer(0)]],
                            device const uint *arena_off [[buffer(1)]],
                            device uint *arena [[buffer(2)]],
                            device const ATState *st [[buffer(3)]],
                            constant ATParams &p [[buffer(4)]],
                            uint c [[thread_position_in_grid]])
{
    uint ncl = min(st->nclusters, p.clusters_cap);
    if (c >= ncl)
        return;
    uint off = arena_off[c] / 4;
    arena[off] = starts[c + 1] - starts[c]; // size
    arena[off + 1] = 0;                     // pad
}

// per-record point copy into the arena
kernel void k_arena_scatter(device const gpu_pt *pts [[buffer(0)]],
                            device const uint *c_of [[buffer(1)]],
                            device const uint *starts [[buffer(2)]],
                            device const uint *arena_off [[buffer(3)]],
                            device gpu_pt *arena [[buffer(4)]],
                            device const ATState *st [[buffer(5)]],
                            constant ATParams &p [[buffer(6)]],
                            uint i [[thread_position_in_grid]])
{
    if (i >= st->nrecords)
        return;
    uint c = c_of[i];
    if (c >= p.clusters_cap)
        return;
    uint rank = i - starts[c];
    // arena_off is in bytes; header is 8 bytes, then 8-byte points
    uint slot = arena_off[c]/8 + 1 + rank;
    arena[slot] = pts[i];
}

///////////////////////////////////////////////////////////////////////////
// stage 7: per-cluster quad-fit preparation. Mirrors the head of the
// CPU's fit_quad(): bounding box, gradient-dot orientation sign, the
// angle sort keys ((slope bits << 32) | ~index), and an in-threadgroup
// bitonic sort of those keys, so the CPU skips straight to the line-fit
// prefix sums. Keys are fp32-faithful to the CPU computation; ulp-level
// divergence (the CPU rounds cx/cy through double) and bitonic tie
// placement fall under the detection epsilon gate, not byte equality.

#define QSORT_MAX 2048u // clusters longer than this sort on the CPU

struct quad_aux {
    ushort xmin, xmax, ymin, ymax;
    float dot;
    uint sorted; // 1 when this cluster's keys are sorted on the GPU
};

kernel void k_quad_bbox(device const gpu_pt *arena [[buffer(0)]],
                        device const uint *arena_off [[buffer(1)]],
                        device const uint *starts [[buffer(2)]],
                        device quad_aux *aux [[buffer(3)]],
                        device const ATState *st [[buffer(4)]],
                        constant ATParams &p [[buffer(5)]],
                        uint c [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup ushort red[4][8];
    uint ncl = min(st->nclusters, p.clusters_cap);
    if (c >= ncl)
        return;
    uint n = starts[c + 1] - starts[c];
    device const gpu_pt *pts = arena + arena_off[c]/8 + 1;

    ushort xmn = 0xffff, xmx = 0, ymn = 0xffff, ymx = 0;
    for (uint i = tid; i < n; i += 256) {
        gpu_pt q = pts[i];
        xmn = min(xmn, q.x);
        xmx = max(xmx, q.x);
        ymn = min(ymn, q.y);
        ymx = max(ymx, q.y);
    }
    xmn = simd_min(xmn);
    xmx = simd_max(xmx);
    ymn = simd_min(ymn);
    ymx = simd_max(ymx);
    if (lane == 0) {
        red[0][sg] = xmn;
        red[1][sg] = xmx;
        red[2][sg] = ymn;
        red[3][sg] = ymx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        quad_aux a;
        a.xmin = 0xffff; a.xmax = 0; a.ymin = 0xffff; a.ymax = 0;
        for (uint s = 0; s < 8; s++) {
            a.xmin = min(a.xmin, red[0][s]);
            a.xmax = max(a.xmax, red[1][s]);
            a.ymin = min(a.ymin, red[2][s]);
            a.ymax = max(a.ymax, red[3][s]);
        }
        a.dot = 0.0f;
        a.sorted = n <= QSORT_MAX ? 1u : 0u;
        aux[c] = a;
    }
}

// the CPU's slope_sort_key: monotone float-bits -> u32 transform
static inline uint slope_key_bits(float slope)
{
    uint u = as_type<uint>(slope);
    return u ^ ((uint)((int)u >> 31) | 0x80000000u);
}

kernel void k_quad_keys(device const gpu_pt *arena [[buffer(0)]],
                        device const uint *arena_off [[buffer(1)]],
                        device const uint *starts [[buffer(2)]],
                        device quad_aux *aux [[buffer(3)]],
                        device ulong *keys_arena [[buffer(4)]],
                        device const float *cxy_table [[buffer(5)]],
                        device const ATState *st [[buffer(6)]],
                        constant ATParams &p [[buffer(7)]],
                        uint c [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float dred[8];
    uint ncl = min(st->nclusters, p.clusters_cap);
    if (c >= ncl)
        return;
    uint n = starts[c + 1] - starts[c];
    uint base = arena_off[c]/8 + 1;
    device const gpu_pt *pts = arena + base;
    device ulong *keys = keys_arena + base;

    quad_aux a = aux[c];
    // host-computed in double then rounded, exactly like the CPU's
    // (xmin + xmax) * 0.5 + noise expression (Metal has no fp64)
    // pt coords are 2*actual, so min+max spans [0, 4*max_dim)
    uint tstride = 4u*p.w;
    float cx = cxy_table[(uint)a.xmin + a.xmax];
    float cy = cxy_table[tstride + (uint)a.ymin + a.ymax];

    float dot = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        gpu_pt q = pts[i];
        float dx = (float)q.x - cx;
        float dy = (float)q.y - cy;

        dot += dx*(float)q.gx + dy*(float)q.gy;

        // quadrants[dy > 0][dx > 0], as on the CPU
        float quadrant = (dy > 0.0f)
            ? ((dx > 0.0f) ? (float)(2 << 15) : (float)(2*(2 << 15)))
            : ((dx > 0.0f) ? 0.0f : (float)(-1*(2 << 15)));
        if (dy < 0.0f) {
            dy = -dy;
            dx = -dx;
        }
        if (dx < 0.0f) {
            float tmp = dx;
            dx = dy;
            dy = -tmp;
        }
        keys[i] = ((ulong)slope_key_bits(quadrant + dy/dx) << 32)
                | (uint)~i;
    }

    // reduction order differs from the CPU's; the dot only decides the
    // border-orientation sign, far from zero for any usable cluster
    dot = simd_sum(dot);
    if (lane == 0)
        dred[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float d = 0.0f;
        for (uint s = 0; s < 8; s++)
            d += dred[s];
        aux[c].dot = d;
    }
}

// Tie-rank of original index i under the CPU merge sort: the CPU sorts
// <=5-element leaves (recursive halving of [0, n)) with networks that
// compare only the slope word — equal-slope elements end up in a
// configuration-dependent network order — and every merge above compares
// the full unique key, which places equal-slope runs leaf-descending.
// Returns ((~leaf_lo & 0x7ff) << 3) | network_pos: ascending order of
// this value reproduces the CPU's order within an equal-slope run.
static inline uint sort_tie_rank(device const ulong *keys, uint i, uint n)
{
    uint lo = 0, ln = n;
    while (ln > 5) {
        uint a = ln >> 1;
        if (i < lo + a) {
            ln = a;
        } else {
            lo += a;
            ln -= a;
        }
    }
    ulong k[5];
    for (uint t = 0; t < ln; t++)
        k[t] = keys[lo + t];
#define MAYBE_SWAP(A, B)     if ((k[A] >> 32) > (k[B] >> 32)) { ulong tmp = k[A]; k[A] = k[B]; k[B] = tmp; }
    if (ln == 2) {
        MAYBE_SWAP(0, 1)
    } else if (ln == 3) {
        MAYBE_SWAP(0, 1) MAYBE_SWAP(1, 2) MAYBE_SWAP(0, 1)
    } else if (ln == 4) {
        MAYBE_SWAP(0, 1) MAYBE_SWAP(2, 3) MAYBE_SWAP(0, 2)
        MAYBE_SWAP(1, 3) MAYBE_SWAP(1, 2)
    } else if (ln == 5) {
        MAYBE_SWAP(0, 1) MAYBE_SWAP(3, 4) MAYBE_SWAP(1, 2)
        MAYBE_SWAP(0, 1) MAYBE_SWAP(0, 3) MAYBE_SWAP(2, 4)
        MAYBE_SWAP(1, 2) MAYBE_SWAP(2, 3) MAYBE_SWAP(1, 2)
    }
#undef MAYBE_SWAP
    ulong mine = keys[i];
    uint pos = 0;
    for (uint t = 0; t < ln; t++)
        if (k[t] == mine)
            pos = t;
    return ((~lo & 0x7ffu) << 3) | pos;
}

// in-threadgroup bitonic sort of one cluster's keys. Full u64 compares
// give the unique ascending order (the low complemented-index word makes
// every key unique); a fixup pass then reorders equal-slope runs into the
// CPU merge sort's tie order via sort_tie_rank, so the output matches
// pt_key_sort exactly.
kernel void k_quad_sort(device const uint *arena_off [[buffer(0)]],
                        device const uint *starts [[buffer(1)]],
                        device const quad_aux *aux [[buffer(2)]],
                        device ulong *keys_arena [[buffer(3)]],
                        device const ATState *st [[buffer(4)]],
                        constant ATParams &p [[buffer(5)]],
                        uint c [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]])
{
    threadgroup ulong buf[QSORT_MAX];
    threadgroup ushort dest[QSORT_MAX];
    uint ncl = min(st->nclusters, p.clusters_cap);
    if (c >= ncl)
        return;
    uint n = starts[c + 1] - starts[c];
    if (n > QSORT_MAX || n < 2)
        return;
    device ulong *keys = keys_arena + arena_off[c]/8 + 1;

    // next power of two
    uint m = 2;
    while (m < n)
        m <<= 1;

    for (uint i = tid; i < m; i += 256)
        buf[i] = i < n ? keys[i] : 0xFFFFFFFFFFFFFFFFul;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= m; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < m/2; i += 256) {
                // index of the i-th comparator's low element
                uint a = ((i & ~(j - 1)) << 1) | (i & (j - 1));
                uint b = a | j;
                bool up = (a & k) == 0;
                ulong va = buf[a], vb = buf[b];
                if ((va > vb) == up) {
                    buf[a] = vb;
                    buf[b] = va;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // `keys` still holds the original order here; sort_tie_rank reads it
    for (uint t = tid; t < n; t += 256) {
        ulong v = buf[t];
        uint hi = (uint)(v >> 32);
        // untied elements (the common case) stay put
        if ((t == 0 || (uint)(buf[t - 1] >> 32) != hi) &&
            (t == n - 1 || (uint)(buf[t + 1] >> 32) != hi)) {
            dest[t] = (ushort)t;
            continue;
        }
        uint st = t;
        while (st > 0 && (uint)(buf[st - 1] >> 32) == hi)
            st--;
        uint my = sort_tie_rank(keys, ~(uint)v, n);
        uint rank = 0;
        for (uint u = st; u < n; u++) {
            ulong w = buf[u];
            if ((uint)(w >> 32) != hi)
                break;
            if (u == t)
                continue;
            rank += sort_tie_rank(keys, ~(uint)w, n) < my ? 1u : 0u;
        }
        dest[t] = (ushort)(st + rank);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid; t < n; t += 256)
        keys[dest[t]] = buf[t];
}
