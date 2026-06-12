/* Metal host for the AprilTag GPU front-end. See METAL_NOTES.md. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pthread.h>
#include <stdatomic.h>

#include "apriltag_metal.h"
#include "apriltag_quad_internal.h"
#include "common/timeprofile.h"
#include "common/unionfind.h"

// MSL source embedded at build time (generated from metal/apriltag_kernels.metal)
extern unsigned char apriltag_kernels_metal[];
extern unsigned int apriltag_kernels_metal_len;

#define INVALID_LABEL 0xFFFFFFFFu

#define OV_RUNS     1u
#define OV_RECORDS  2u
#define OV_COMPS    4u
#define OV_CLUSTERS 8u

// must match the .metal definitions exactly
typedef struct {
    uint32_t w, h, s;
    uint32_t tw, th;
    uint32_t min_wb_diff;
    uint32_t min_cluster_pixels;
    uint32_t runs_cap;
    uint32_t records_cap;
    uint32_t comps_cap;
    uint32_t clusters_cap;
    uint32_t emit_blocks_x;
} ATParams;

typedef struct {
    uint32_t nruns;
    uint32_t nnodes;
    uint32_t ncomp;
    uint32_t nrecords;
    uint32_t nclusters;
    uint32_t overflow;
    uint32_t runs_tg[3];
    uint32_t nodes_tg[3];
    uint32_t rec_tg[3];
    uint32_t radix_tg[3];
    uint32_t cl_tg[3];
} ATState;

#define RADIX_BLOCK 1024u
#define RADIX_BINS 256u
#define N_RADIX_PASSES 4u // 30-bit keys, 8 bits per pass

struct apriltag_metal {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;

    // pipelines
    id<MTLComputePipelineState> minmax, blur, threshold;
    id<MTLComputePipelineState> rle_count, scan_rows, rle_fill;
    id<MTLComputePipelineState> uf_init, uf_connect, uf_flatten;
    id<MTLComputePipelineState> zero_u32, comp_count, root_flag, scan_nodes;
    id<MTLComputePipelineState> node_label, label_clear, label_paint;
    id<MTLComputePipelineState> emit_count, scan_blocks, emit_scatter;
    id<MTLComputePipelineState> radix_hist, radix_scan, radix_scatter;
    id<MTLComputePipelineState> cluster_bounds, arena_headers, arena_scatter;

    // persistent buffers (grow-only). The five buffers the CPU touches
    // while another frame's GPU work may be in flight (image staging in,
    // params/state, cluster arena out) exist once per pipeline slot so a
    // prepared frame N+1 cannot clobber what frame N's CPU half reads;
    // everything else is GPU-internal and serialized by Metal's hazard
    // tracking across command buffers.
    id<MTLBuffer> img[2];     // stride*h bytes, copy-in
    id<MTLBuffer> threshim;   // stride*h
    id<MTLBuffer> tiles;      // 4 * tw*th (min, max, blurred min, blurred max)
    id<MTLBuffer> row_off;    // (h+1) u32
    id<MTLBuffer> runs;       // runs_cap * 8B
    id<MTLBuffer> parent;     // (runs_cap + h) u32
    id<MTLBuffer> count;      // (runs_cap + h) u32
    id<MTLBuffer> dense;      // (runs_cap + h) u32 (flags, then dense ids)
    id<MTLBuffer> node_lbl;   // (runs_cap + h) u32
    id<MTLBuffer> labels;     // w*h u32
    id<MTLBuffer> block_counts; // emit_blocks_x*h u32
    id<MTLBuffer> keys[2];    // records_cap u32 (ping/pong)
    id<MTLBuffer> pts[2];     // records_cap 8B
    id<MTLBuffer> hist;       // RADIX_BINS * (records_cap/RADIX_BLOCK+1) u32
    id<MTLBuffer> c_of;       // records_cap u32
    id<MTLBuffer> starts;     // (clusters_cap+1) u32
    id<MTLBuffer> arena_off[2]; // clusters_cap u32
    id<MTLBuffer> arena[2];   // 8*clusters_cap + 8*records_cap bytes
    id<MTLBuffer> params_buf[2]; // ATParams
    id<MTLBuffer> state_buf[2];  // ATState
    id<MTLBuffer> shifts;     // N_RADIX_PASSES u32 (radix pass shifts)

    ATParams p;
    int validate;
    int prof;

    // front-ends committed by apriltag_metal_frontend_begin and not yet
    // consumed by apriltag_metal_clusters (FIFO, at most one per slot).
    // pend_im is only compared by pointer; the pixels were already copied
    // into the slot's img buffer at begin time. mu protects this queue,
    // the slot allocator, and all encode/buffer setup, so frontend_begin
    // may be called from a different thread than detect.
    id<MTLCommandBuffer> pend_cb[2];
    const image_u8_t *pend_im[2];
    uint32_t pend_w[2], pend_h[2], pend_s[2];
    int pend_slot[2];
    int npend;
    int next_slot;
    pthread_mutex_t mu;
};

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                            NSString *name)
{
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "apriltag_metal: missing kernel %s\n", name.UTF8String);
        return nil;
    }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn
                                                                         error:&err];
    if (!pso)
        fprintf(stderr, "apriltag_metal: pso %s: %s\n", name.UTF8String,
                err.localizedDescription.UTF8String);
    return pso;
}

apriltag_metal_t *apriltag_metal_create(void)
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev)
            return NULL;

        NSString *src = [[NSString alloc] initWithBytes:apriltag_kernels_metal
                                                 length:apriltag_kernels_metal_len
                                               encoding:NSUTF8StringEncoding];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            fprintf(stderr, "apriltag_metal: shader compile failed: %s\n",
                    err.localizedDescription.UTF8String);
            return NULL;
        }

        apriltag_metal_t *m = calloc(1, sizeof(*m));
        m->dev = dev;
        m->queue = [dev newCommandQueue];

#define PSO(field, kname) do { \
        m->field = make_pso(dev, lib, @kname); \
        if (!m->field) { apriltag_metal_destroy(m); return NULL; } \
    } while (0)

        PSO(minmax, "k_minmax");
        PSO(blur, "k_blur");
        PSO(threshold, "k_threshold");
        PSO(rle_count, "k_rle_count");
        PSO(scan_rows, "k_scan_rows");
        PSO(rle_fill, "k_rle_fill");
        PSO(uf_init, "k_uf_init");
        PSO(uf_connect, "k_uf_connect");
        PSO(uf_flatten, "k_uf_flatten");
        PSO(zero_u32, "k_zero_u32");
        PSO(comp_count, "k_comp_count");
        PSO(root_flag, "k_root_flag");
        PSO(scan_nodes, "k_scan_nodes");
        PSO(node_label, "k_node_label");
        PSO(label_clear, "k_label_clear");
        PSO(label_paint, "k_label_paint");
        PSO(emit_count, "k_emit_count");
        PSO(scan_blocks, "k_scan_blocks");
        PSO(emit_scatter, "k_emit_scatter");
        PSO(radix_hist, "k_radix_hist");
        PSO(radix_scan, "k_radix_scan");
        PSO(radix_scatter, "k_radix_scatter");
        PSO(cluster_bounds, "k_cluster_bounds");
        PSO(arena_headers, "k_arena_headers");
        PSO(arena_scatter, "k_arena_scatter");
#undef PSO

        pthread_mutex_init(&m->mu, NULL);
        m->validate = getenv("APRILTAG_METAL_VALIDATE") != NULL;
        m->prof = getenv("APRILTAG_METAL_PROF") != NULL;
        return m;
    }
}

void apriltag_metal_destroy(apriltag_metal_t *m)
{
    if (!m)
        return;
    for (int i = 0; i < m->npend; i++) {
        [m->pend_cb[i] waitUntilCompleted];
        m->pend_cb[i] = nil;
    }
    m->npend = 0;
    // ARC releases the ObjC objects when the struct fields are nilled;
    // under MRC-with-ARC-file this file is compiled with ARC, so just
    // free the C allocation after clearing references.
    m->dev = nil;
    m->queue = nil;
    m->minmax = m->blur = m->threshold = nil;
    m->rle_count = m->scan_rows = m->rle_fill = nil;
    m->uf_init = m->uf_connect = m->uf_flatten = nil;
    m->zero_u32 = m->comp_count = m->root_flag = m->scan_nodes = nil;
    m->node_label = m->label_clear = m->label_paint = nil;
    m->emit_count = m->scan_blocks = m->emit_scatter = nil;
    m->radix_hist = m->radix_scan = m->radix_scatter = nil;
    m->cluster_bounds = m->arena_headers = m->arena_scatter = nil;
    m->img[0] = m->img[1] = nil;
    m->threshim = m->tiles = m->row_off = m->runs = nil;
    m->parent = m->count = m->dense = m->node_lbl = m->labels = nil;
    m->block_counts = nil;
    m->keys[0] = m->keys[1] = m->pts[0] = m->pts[1] = nil;
    m->hist = m->c_of = m->starts = nil;
    m->arena_off[0] = m->arena_off[1] = m->arena[0] = m->arena[1] = nil;
    m->params_buf[0] = m->params_buf[1] = nil;
    m->state_buf[0] = m->state_buf[1] = nil;
    m->shifts = nil;
    pthread_mutex_destroy(&m->mu);
    free(m);
}

static id<MTLBuffer> ensure_buf(apriltag_metal_t *m, id<MTLBuffer> cur, size_t len)
{
    if (cur && cur.length >= len)
        return cur;
    return [m->dev newBufferWithLength:len options:MTLResourceStorageModeShared];
}

static void setup_buffers(apriltag_metal_t *m, apriltag_detector_t *td,
                          image_u8_t *im, int slot)
{
    uint32_t w = im->width, h = im->height, s = im->stride;
    ATParams *p = &m->p;
    p->w = w;
    p->h = h;
    p->s = s;
    p->tw = w/4;
    p->th = h/4;
    p->min_wb_diff = td->qtp.min_white_black_diff;
    p->min_cluster_pixels = td->qtp.min_cluster_pixels;
    // caps sized from corpus stats (max seen: 1.37M runs, 3.74M points,
    // 13.8k clusters); generous headroom, overflow flagged if exceeded
    if (p->runs_cap == 0) {
        p->runs_cap = 2*1024*1024;
        p->records_cap = 6*1024*1024;
        p->comps_cap = 32768;
        p->clusters_cap = 65536;
    }
    p->emit_blocks_x = (w + 255)/256;

    m->img[slot] = ensure_buf(m, m->img[slot], (size_t)s*h);
    m->threshim = ensure_buf(m, m->threshim, (size_t)s*h);
    m->tiles = ensure_buf(m, m->tiles, (size_t)4*p->tw*p->th);
    m->row_off = ensure_buf(m, m->row_off, (size_t)(h + 1)*4);
    m->runs = ensure_buf(m, m->runs, (size_t)p->runs_cap*8);
    size_t nodes = (size_t)p->runs_cap + h;
    m->parent = ensure_buf(m, m->parent, nodes*4);
    m->count = ensure_buf(m, m->count, nodes*4);
    m->dense = ensure_buf(m, m->dense, nodes*4);
    m->node_lbl = ensure_buf(m, m->node_lbl, nodes*4);
    m->labels = ensure_buf(m, m->labels, (size_t)w*h*4);
    m->block_counts = ensure_buf(m, m->block_counts, (size_t)p->emit_blocks_x*h*4);
    for (int i = 0; i < 2; i++) {
        m->keys[i] = ensure_buf(m, m->keys[i], (size_t)p->records_cap*4);
        m->pts[i] = ensure_buf(m, m->pts[i], (size_t)p->records_cap*8);
    }
    m->hist = ensure_buf(m, m->hist,
                         (size_t)RADIX_BINS*(p->records_cap/RADIX_BLOCK + 1)*4);
    m->c_of = ensure_buf(m, m->c_of, (size_t)p->records_cap*4);
    m->starts = ensure_buf(m, m->starts, (size_t)(p->clusters_cap + 1)*4);
    m->arena_off[slot] = ensure_buf(m, m->arena_off[slot], (size_t)p->clusters_cap*4);
    m->arena[slot] = ensure_buf(m, m->arena[slot],
                                (size_t)8*p->clusters_cap + (size_t)8*p->records_cap);
    m->params_buf[slot] = ensure_buf(m, m->params_buf[slot], sizeof(ATParams));
    m->state_buf[slot] = ensure_buf(m, m->state_buf[slot], sizeof(ATState));
    m->shifts = ensure_buf(m, m->shifts, N_RADIX_PASSES*4);

    memcpy(m->params_buf[slot].contents, p, sizeof(*p));
    uint32_t *sh = m->shifts.contents;
    for (uint32_t i = 0; i < N_RADIX_PASSES; i++)
        sh[i] = 8*i;
}

// dispatch helpers
static void disp2d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                   uint32_t nx, uint32_t ny, uint32_t tgx, uint32_t tgy)
{
    [enc setComputePipelineState:pso];
    MTLSize tg = MTLSizeMake(tgx, tgy, 1);
    MTLSize grid = MTLSizeMake((nx + tgx - 1)/tgx, (ny + tgy - 1)/tgy, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}

static void disp1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                   uint32_t n, uint32_t tgx)
{
    disp2d(enc, pso, n, 1, tgx, 1);
}

static void disp_indirect(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          id<MTLBuffer> state, size_t args_off, uint32_t tgx)
{
    [enc setComputePipelineState:pso];
    [enc dispatchThreadgroupsWithIndirectBuffer:state
                           indirectBufferOffset:args_off
                          threadsPerThreadgroup:MTLSizeMake(tgx, 1, 1)];
}

static void validate_frame(apriltag_metal_t *m, apriltag_detector_t *td,
                           image_u8_t *im, zarray_t *gpu_clusters, int slot);

// Encode the whole front-end chain (threshold/RLE, union-find/labels,
// emission/sort/arena) into one serial command buffer and commit it
// without waiting; the caller owns the wait. Sequential dispatches in a
// serial compute encoder are ordered with memory coherence, including
// the indirect dispatch arguments written by the scan kernels.
static id<MTLCommandBuffer> frontend_commit(apriltag_metal_t *m,
                                            apriltag_detector_t *td,
                                            image_u8_t *im, int slot)
{
        setup_buffers(m, td, im, slot);
        id<MTLBuffer> img = m->img[slot];
        id<MTLBuffer> arena_off = m->arena_off[slot];
        id<MTLBuffer> arena = m->arena[slot];
        id<MTLBuffer> params_buf = m->params_buf[slot];
        id<MTLBuffer> state_buf = m->state_buf[slot];
        ATParams *p = &m->p;
        uint32_t w = p->w, h = p->h;

        memcpy(img.contents, im->buf, (size_t)p->s*h);
        ATState *st = state_buf.contents;
        memset(st, 0, sizeof(*st));

        const size_t off_runs_tg = offsetof(ATState, runs_tg);
        const size_t off_nodes_tg = offsetof(ATState, nodes_tg);
        const size_t off_rec_tg = offsetof(ATState, rec_tg);
        const size_t off_radix_tg = offsetof(ATState, radix_tg);
        const size_t off_cl_tg = offsetof(ATState, cl_tg);

        id<MTLCommandBuffer> cb = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // ---- threshold + RLE -------------------------------------
        {
            [enc setBuffer:img offset:0 atIndex:0];
            [enc setBuffer:m->tiles offset:0 atIndex:1];
            [enc setBuffer:m->tiles offset:(size_t)p->tw*p->th atIndex:2];
            [enc setBuffer:params_buf offset:0 atIndex:3];
            disp2d(enc, m->minmax, p->tw, p->th, 16, 16);

            [enc setBuffer:m->tiles offset:0 atIndex:0];
            [enc setBuffer:m->tiles offset:(size_t)p->tw*p->th atIndex:1];
            [enc setBuffer:m->tiles offset:(size_t)2*p->tw*p->th atIndex:2];
            [enc setBuffer:m->tiles offset:(size_t)3*p->tw*p->th atIndex:3];
            [enc setBuffer:params_buf offset:0 atIndex:4];
            disp2d(enc, m->blur, p->tw, p->th, 16, 16);

            [enc setBuffer:img offset:0 atIndex:0];
            [enc setBuffer:m->tiles offset:(size_t)2*p->tw*p->th atIndex:1];
            [enc setBuffer:m->tiles offset:(size_t)3*p->tw*p->th atIndex:2];
            [enc setBuffer:m->threshim offset:0 atIndex:3];
            [enc setBuffer:params_buf offset:0 atIndex:4];
            disp2d(enc, m->threshold, w, h, 32, 8);

            [enc setBuffer:m->threshim offset:0 atIndex:0];
            [enc setBuffer:m->row_off offset:0 atIndex:1];
            [enc setBuffer:params_buf offset:0 atIndex:2];
            disp1d(enc, m->rle_count, h, 64);

            [enc setBuffer:m->row_off offset:0 atIndex:0];
            [enc setBuffer:state_buf offset:0 atIndex:1];
            [enc setBuffer:params_buf offset:0 atIndex:2];
            [enc setComputePipelineState:m->scan_rows];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:m->threshim offset:0 atIndex:0];
            [enc setBuffer:m->row_off offset:0 atIndex:1];
            [enc setBuffer:m->runs offset:0 atIndex:2];
            [enc setBuffer:params_buf offset:0 atIndex:3];
            disp1d(enc, m->rle_fill, h, 64);
        }

        // ---- union-find + labels ----------------------------------
        {
            [enc setBuffer:m->parent offset:0 atIndex:0];
            [enc setBuffer:state_buf offset:0 atIndex:1];
            disp_indirect(enc, m->uf_init, state_buf, off_nodes_tg, 256);

            [enc setBuffer:m->parent offset:0 atIndex:0];
            [enc setBuffer:m->runs offset:0 atIndex:1];
            [enc setBuffer:m->row_off offset:0 atIndex:2];
            [enc setBuffer:m->threshim offset:0 atIndex:3];
            [enc setBuffer:state_buf offset:0 atIndex:4];
            [enc setBuffer:params_buf offset:0 atIndex:5];
            disp_indirect(enc, m->uf_connect, state_buf, off_runs_tg, 256);

            [enc setBuffer:m->parent offset:0 atIndex:0];
            [enc setBuffer:state_buf offset:0 atIndex:1];
            disp_indirect(enc, m->uf_flatten, state_buf, off_nodes_tg, 256);

            [enc setBuffer:m->count offset:0 atIndex:0];
            [enc setBuffer:state_buf offset:0 atIndex:1];
            disp_indirect(enc, m->zero_u32, state_buf, off_nodes_tg, 256);

            [enc setBuffer:m->parent offset:0 atIndex:0];
            [enc setBuffer:m->count offset:0 atIndex:1];
            [enc setBuffer:m->runs offset:0 atIndex:2];
            [enc setBuffer:state_buf offset:0 atIndex:3];
            disp_indirect(enc, m->comp_count, state_buf, off_nodes_tg, 256);

            [enc setBuffer:m->parent offset:0 atIndex:0];
            [enc setBuffer:m->count offset:0 atIndex:1];
            [enc setBuffer:m->dense offset:0 atIndex:2];
            [enc setBuffer:state_buf offset:0 atIndex:3];
            [enc setBuffer:params_buf offset:0 atIndex:4];
            disp_indirect(enc, m->root_flag, state_buf, off_nodes_tg, 256);

            [enc setBuffer:m->dense offset:0 atIndex:0];
            [enc setBuffer:state_buf offset:0 atIndex:1];
            [enc setBuffer:params_buf offset:0 atIndex:2];
            [enc setComputePipelineState:m->scan_nodes];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:m->parent offset:0 atIndex:0];
            [enc setBuffer:m->count offset:0 atIndex:1];
            [enc setBuffer:m->dense offset:0 atIndex:2];
            [enc setBuffer:m->node_lbl offset:0 atIndex:3];
            [enc setBuffer:state_buf offset:0 atIndex:4];
            [enc setBuffer:params_buf offset:0 atIndex:5];
            disp_indirect(enc, m->node_label, state_buf, off_nodes_tg, 256);

            [enc setBuffer:m->labels offset:0 atIndex:0];
            [enc setBuffer:params_buf offset:0 atIndex:1];
            disp1d(enc, m->label_clear, w*h, 256);

            [enc setBuffer:m->node_lbl offset:0 atIndex:0];
            [enc setBuffer:m->runs offset:0 atIndex:1];
            [enc setBuffer:m->labels offset:0 atIndex:2];
            [enc setBuffer:state_buf offset:0 atIndex:3];
            [enc setBuffer:params_buf offset:0 atIndex:4];
            disp_indirect(enc, m->label_paint, state_buf, off_nodes_tg, 256);
        }

        // ---- emission + sort + arena -------------------------------
        {
            [enc setBuffer:m->threshim offset:0 atIndex:0];
            [enc setBuffer:m->labels offset:0 atIndex:1];
            [enc setBuffer:m->block_counts offset:0 atIndex:2];
            [enc setBuffer:params_buf offset:0 atIndex:3];
            [enc setComputePipelineState:m->emit_count];
            [enc dispatchThreadgroups:MTLSizeMake(p->emit_blocks_x, h, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:m->block_counts offset:0 atIndex:0];
            [enc setBuffer:state_buf offset:0 atIndex:1];
            [enc setBuffer:params_buf offset:0 atIndex:2];
            [enc setComputePipelineState:m->scan_blocks];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:m->threshim offset:0 atIndex:0];
            [enc setBuffer:m->labels offset:0 atIndex:1];
            [enc setBuffer:m->block_counts offset:0 atIndex:2];
            [enc setBuffer:m->keys[0] offset:0 atIndex:3];
            [enc setBuffer:m->pts[0] offset:0 atIndex:4];
            [enc setBuffer:params_buf offset:0 atIndex:5];
            [enc setComputePipelineState:m->emit_scatter];
            [enc dispatchThreadgroups:MTLSizeMake(p->emit_blocks_x, h, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // radix passes: ping-pong 0 -> 1 -> 0 ...
            for (uint32_t pass = 0; pass < N_RADIX_PASSES; pass++) {
                int src = pass & 1, dst = src ^ 1;

                [enc setBuffer:m->keys[src] offset:0 atIndex:0];
                [enc setBuffer:m->hist offset:0 atIndex:1];
                [enc setBuffer:state_buf offset:0 atIndex:2];
                [enc setBuffer:m->shifts offset:(size_t)pass*4 atIndex:3];
                disp_indirect(enc, m->radix_hist, state_buf, off_radix_tg, 256);

                [enc setBuffer:m->hist offset:0 atIndex:0];
                [enc setBuffer:state_buf offset:0 atIndex:1];
                [enc setComputePipelineState:m->radix_scan];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setBuffer:m->keys[src] offset:0 atIndex:0];
                [enc setBuffer:m->pts[src] offset:0 atIndex:1];
                [enc setBuffer:m->keys[dst] offset:0 atIndex:2];
                [enc setBuffer:m->pts[dst] offset:0 atIndex:3];
                [enc setBuffer:m->hist offset:0 atIndex:4];
                [enc setBuffer:state_buf offset:0 atIndex:5];
                [enc setBuffer:m->shifts offset:(size_t)pass*4 atIndex:6];
                disp_indirect(enc, m->radix_scatter, state_buf, off_radix_tg, 256);
            }

            // after an even number of passes the data is in set 0
            [enc setBuffer:m->keys[0] offset:0 atIndex:0];
            [enc setBuffer:m->c_of offset:0 atIndex:1];
            [enc setBuffer:m->starts offset:0 atIndex:2];
            [enc setBuffer:arena_off offset:0 atIndex:3];
            [enc setBuffer:state_buf offset:0 atIndex:4];
            [enc setBuffer:params_buf offset:0 atIndex:5];
            [enc setComputePipelineState:m->cluster_bounds];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:m->starts offset:0 atIndex:0];
            [enc setBuffer:arena_off offset:0 atIndex:1];
            [enc setBuffer:arena offset:0 atIndex:2];
            [enc setBuffer:state_buf offset:0 atIndex:3];
            [enc setBuffer:params_buf offset:0 atIndex:4];
            disp_indirect(enc, m->arena_headers, state_buf, off_cl_tg, 256);

            [enc setBuffer:m->pts[0] offset:0 atIndex:0];
            [enc setBuffer:m->c_of offset:0 atIndex:1];
            [enc setBuffer:m->starts offset:0 atIndex:2];
            [enc setBuffer:arena_off offset:0 atIndex:3];
            [enc setBuffer:arena offset:0 atIndex:4];
            [enc setBuffer:state_buf offset:0 atIndex:5];
            [enc setBuffer:params_buf offset:0 atIndex:6];
            disp_indirect(enc, m->arena_scatter, state_buf, off_rec_tg, 256);
        }

        [enc endEncoding];
        [cb commit];
        return cb;
}

// Kick off the GPU front-end for an image about to be detected; the GPU
// runs while the caller does unrelated CPU work (the next frame's load,
// or — because the CPU-visible buffers are per-slot — the previous
// frame's quad fitting and decode, so this may be called from a loader
// thread while apriltag_detector_detect runs. The pixels are copied out
// synchronously, but `im` must still be the live pointer later passed to
// apriltag_detector_detect.
void apriltag_metal_frontend_begin(apriltag_metal_t *m, apriltag_detector_t *td,
                                   image_u8_t *im)
{
    @autoreleasepool {
        pthread_mutex_lock(&m->mu);
        if (m->npend == 2) {
            // queue full (callers normally keep at most one outstanding);
            // drop the oldest prepared frame
            [m->pend_cb[0] waitUntilCompleted];
            m->pend_cb[0] = m->pend_cb[1];
            m->pend_im[0] = m->pend_im[1];
            m->pend_w[0] = m->pend_w[1];
            m->pend_h[0] = m->pend_h[1];
            m->pend_s[0] = m->pend_s[1];
            m->pend_slot[0] = m->pend_slot[1];
            m->pend_cb[1] = nil;
            m->npend = 1;
        }
        int slot = m->next_slot;
        m->next_slot ^= 1;
        // never encode into a slot a queued frame still owns
        for (int i = 0; i < m->npend; i++) {
            if (m->pend_slot[i] == slot) {
                [m->pend_cb[i] waitUntilCompleted];
                m->pend_cb[i] = nil;
                for (int j = i + 1; j < m->npend; j++) {
                    m->pend_cb[j-1] = m->pend_cb[j];
                    m->pend_im[j-1] = m->pend_im[j];
                    m->pend_w[j-1] = m->pend_w[j];
                    m->pend_h[j-1] = m->pend_h[j];
                    m->pend_s[j-1] = m->pend_s[j];
                    m->pend_slot[j-1] = m->pend_slot[j];
                }
                m->pend_cb[m->npend-1] = nil;
                m->npend--;
                break;
            }
        }
        CFAbsoluteTime t0 = m->prof ? CFAbsoluteTimeGetCurrent() : 0;
        id<MTLCommandBuffer> cb = frontend_commit(m, td, im, slot);
        int i = m->npend++;
        m->pend_cb[i] = cb;
        m->pend_im[i] = im;
        m->pend_w[i] = im->width;
        m->pend_h[i] = im->height;
        m->pend_s[i] = im->stride;
        m->pend_slot[i] = slot;
        if (m->prof)
            fprintf(stderr, "[prof] prepare      encode+commit %6.3f ms\n",
                    (CFAbsoluteTimeGetCurrent() - t0)*1e3);
        pthread_mutex_unlock(&m->mu);
    }
}

zarray_t *apriltag_metal_clusters(apriltag_metal_t *m, apriltag_detector_t *td,
                                  image_u8_t *im)
{
    @autoreleasepool {
        pthread_mutex_lock(&m->mu);
        id<MTLCommandBuffer> cb = nil;
        int slot = -1;
        int found = -1;
        for (int i = 0; i < m->npend; i++) {
            if (m->pend_im[i] == im && m->pend_w[i] == (uint32_t)im->width &&
                m->pend_h[i] == (uint32_t)im->height &&
                m->pend_s[i] == (uint32_t)im->stride) {
                found = i;
                break;
            }
        }
        if (found >= 0) {
            // drop prepared frames older than the one being detected
            for (int i = 0; i < found; i++) {
                [m->pend_cb[i] waitUntilCompleted];
                m->pend_cb[i] = nil;
            }
            cb = m->pend_cb[found];
            slot = m->pend_slot[found];
            int k = 0;
            for (int i = found + 1; i < m->npend; i++, k++) {
                m->pend_cb[k] = m->pend_cb[i];
                m->pend_im[k] = m->pend_im[i];
                m->pend_w[k] = m->pend_w[i];
                m->pend_h[k] = m->pend_h[i];
                m->pend_s[k] = m->pend_s[i];
                m->pend_slot[k] = m->pend_slot[i];
            }
            for (int i = k; i < m->npend; i++)
                m->pend_cb[i] = nil;
            m->npend = k;
        } else {
            // nothing (or only stale frames) prepared: drain and encode now
            for (int i = 0; i < m->npend; i++) {
                [m->pend_cb[i] waitUntilCompleted];
                m->pend_cb[i] = nil;
            }
            m->npend = 0;
            slot = m->next_slot;
            m->next_slot ^= 1;
            cb = frontend_commit(m, td, im, slot);
        }
        pthread_mutex_unlock(&m->mu);
        timeprofile_stamp(td->tp, "threshold");
        timeprofile_stamp(td->tp, "unionfind");

        CFAbsoluteTime t0 = m->prof ? CFAbsoluteTimeGetCurrent() : 0;
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError)
            fprintf(stderr, "apriltag_metal: command buffer error: %s\n",
                    cb.error.localizedDescription.UTF8String);
        if (m->prof) {
            CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
            fprintf(stderr, "[prof] frontend     wall %6.3f ms  gpu %6.3f ms  "
                    "(kernel %6.3f ms)\n", (t1 - t0)*1e3,
                    (cb.GPUEndTime - cb.GPUStartTime)*1e3,
                    (cb.kernelEndTime - cb.kernelStartTime)*1e3);
        }

        ATState *st = m->state_buf[slot].contents;
        if (st->overflow) {
            fprintf(stderr, "apriltag_metal: overflow flags 0x%x; "
                            "increase caps\n", st->overflow);
            return zarray_create(sizeof(struct pt_list *));
        }

        // build the zarray of pt_list* pointing into the arena
        uint32_t ncl = st->nclusters;
        uint32_t *arena_off = m->arena_off[slot].contents;
        uint8_t *arena = m->arena[slot].contents;
        zarray_t *clusters = zarray_create(sizeof(struct pt_list *));
        zarray_ensure_capacity(clusters, ncl);
        for (uint32_t c = 0; c < ncl; c++) {
            struct pt_list *cl = (struct pt_list *)(arena + arena_off[c]);
            zarray_add(clusters, &cl);
        }
        timeprofile_stamp(td->tp, "make clusters");

        if (m->validate) {
            // hold the queue lock so a concurrent frontend_begin cannot
            // overwrite the shared GPU scratch being compared
            pthread_mutex_lock(&m->mu);
            validate_frame(m, td, im, clusters, slot);
            pthread_mutex_unlock(&m->mu);
        }

        return clusters;
    }
}

///////////////////////////////////////////////////////////////////////////
// validation: run the CPU reference stages and compare

static uint64_t fnv64(const void *data, size_t len, uint64_t h)
{
    const uint8_t *b = data;
    for (size_t i = 0; i < len; i++) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static void validate_frame(apriltag_metal_t *m, apriltag_detector_t *td,
                           image_u8_t *im, zarray_t *gpu_clusters, int slot)
{
    int w = im->width, h = im->height;

    struct row_run *cpu_runs;
    uint32_t *cpu_row_off;
    image_u8_t *threshim = threshold(td, im, &cpu_runs, &cpu_row_off);
    int ts = threshim->stride;

    // 1. threshim bytes
    const uint8_t *gpu_t = m->threshim.contents;
    long tdiff = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (gpu_t[y*ts + x] != threshim->buf[y*ts + x])
                tdiff++;
    fprintf(stderr, "[validate] threshim: %ld byte diffs\n", tdiff);

    // 2. runs
    ATState *st = m->state_buf[slot].contents;
    uint32_t *gpu_row_off = m->row_off.contents;
    long rdiff = (long)cpu_row_off[h] - (long)st->nruns;
    long rbad = 0;
    if (memcmp(gpu_row_off, cpu_row_off, (h + 1)*sizeof(uint32_t)) != 0)
        rbad = -1;
    struct { uint16_t start, end, y; uint8_t v, pad; } *gruns = m->runs.contents;
    if (rbad == 0) {
        for (uint32_t i = 0; i < st->nruns; i++) {
            if (gruns[i].start != cpu_runs[i].start ||
                gruns[i].end != cpu_runs[i].end ||
                gruns[i].v != cpu_runs[i].v) {
                rbad++;
                if (rbad < 4)
                    fprintf(stderr, "[validate] run %u: gpu (%u,%u,%u) cpu (%u,%u,%u)\n",
                            i, gruns[i].start, gruns[i].end, gruns[i].v,
                            cpu_runs[i].start, cpu_runs[i].end, cpu_runs[i].v);
            }
        }
    }
    fprintf(stderr, "[validate] runs: count delta %ld, row_off %s, %ld bad runs\n",
            rdiff, rbad == -1 ? "MISMATCH" : "ok", rbad > 0 ? rbad : 0);

    // 3. component partition: CPU per-pixel labels (usable components only)
    // vs the GPU label image, up to a bijection of label values
    unionfind_t *uf = connected_components(td, threshim, w, h, ts, cpu_runs,
                                           cpu_row_off);
    {
        uint32_t *gpu_labels = m->labels.contents;
        uint32_t vcol_base = cpu_row_off[h];
        int min_px = td->qtp.min_cluster_pixels;
        // bijection check: map cpu rep -> gpu label and detect conflicts
        // (reps fit in nnodes; use a flat array)
        uint32_t nnodes = vcol_base + h;
        uint32_t *fwd = malloc(sizeof(uint32_t)*nnodes);
        memset(fwd, 0xFF, sizeof(uint32_t)*nnodes);
        long part_bad = 0, gate_bad = 0;
        for (int y = 0; y < h && part_bad < 5; y++) {
            // runs of this row
            for (uint32_t ri = cpu_row_off[y]; ri < cpu_row_off[y+1]; ri++) {
                uint32_t rep = unionfind_get_representative(uf, ri);
                int usable = (int)(unionfind_get_set_size(uf, rep)) >= min_px;
                for (int x = cpu_runs[ri].start; x <= cpu_runs[ri].end; x++) {
                    uint32_t g = gpu_labels[y*w + x];
                    if (!usable) {
                        if (g != INVALID_LABEL) { gate_bad++; }
                        continue;
                    }
                    if (g == INVALID_LABEL) { gate_bad++; continue; }
                    if (fwd[rep] == INVALID_LABEL)
                        fwd[rep] = g;
                    else if (fwd[rep] != g) {
                        part_bad++;
                        if (part_bad < 4)
                            fprintf(stderr, "[validate] partition split: cpu rep %u "
                                    "-> gpu %u and %u at (%d,%d)\n",
                                    rep, fwd[rep], g, x, y);
                    }
                }
            }
            // virtual last-column node
            uint32_t vn = vcol_base + y;
            uint32_t rep = unionfind_get_representative(uf, vn);
            int usable = (int)(unionfind_get_set_size(uf, rep)) >= min_px;
            uint32_t g = gpu_labels[y*w + (w-1)];
            if (!usable) {
                if (g != INVALID_LABEL) gate_bad++;
            } else if (g == INVALID_LABEL) {
                gate_bad++;
            } else if (fwd[rep] == INVALID_LABEL) {
                fwd[rep] = g;
            } else if (fwd[rep] != g) {
                part_bad++;
            }
        }
        // reverse injectivity: two cpu reps -> same gpu label?
        uint32_t *rev = malloc(sizeof(uint32_t)*65536);
        memset(rev, 0xFF, sizeof(uint32_t)*65536);
        long merge_bad = 0;
        for (uint32_t i = 0; i < nnodes; i++) {
            uint32_t g = fwd[i];
            if (g == INVALID_LABEL || g >= 65536)
                continue;
            if (rev[g] == INVALID_LABEL)
                rev[g] = i;
            else if (rev[g] != i)
                merge_bad++;
        }
        fprintf(stderr, "[validate] partition: splits %ld, merges %ld, gate diffs %ld\n",
                part_bad, merge_bad, gate_bad);
        free(fwd);
        free(rev);
    }

    zarray_t *cpu_clusters = gradient_clusters(td, threshim, w, h, ts, uf,
                                               cpu_runs, cpu_row_off);

    int ncpu = zarray_size(cpu_clusters), ngpu = zarray_size(gpu_clusters);
    long cpu_pts = 0, gpu_pts = 0;
    uint64_t *hcpu = malloc(sizeof(uint64_t)*ncpu);
    uint64_t *hgpu = malloc(sizeof(uint64_t)*ngpu);
    uint64_t *mcpu = malloc(sizeof(uint64_t)*ncpu);
    uint64_t *mgpu = malloc(sizeof(uint64_t)*ngpu);
    size_t tmp_cap = 1024;
    struct pt *tmp = malloc(sizeof(struct pt)*tmp_cap);
    for (int i = 0; i < ncpu; i++) {
        struct pt_list *cl;
        zarray_get(cpu_clusters, i, &cl);
        cpu_pts += cl->size;
        hcpu[i] = fnv64(cl->pts, sizeof(struct pt)*cl->size, 0xcbf29ce484222325ULL);
        if ((size_t)cl->size > tmp_cap) {
            tmp_cap = cl->size;
            tmp = realloc(tmp, sizeof(struct pt)*tmp_cap);
        }
        memcpy(tmp, cl->pts, sizeof(struct pt)*cl->size);
        qsort(tmp, cl->size, sizeof(struct pt), cmp_u64);
        mcpu[i] = fnv64(tmp, sizeof(struct pt)*cl->size, 0xcbf29ce484222325ULL);
    }
    for (int i = 0; i < ngpu; i++) {
        struct pt_list *cl;
        zarray_get(gpu_clusters, i, &cl);
        gpu_pts += cl->size;
        hgpu[i] = fnv64(cl->pts, sizeof(struct pt)*cl->size, 0xcbf29ce484222325ULL);
        if ((size_t)cl->size > tmp_cap) {
            tmp_cap = cl->size;
            tmp = realloc(tmp, sizeof(struct pt)*tmp_cap);
        }
        memcpy(tmp, cl->pts, sizeof(struct pt)*cl->size);
        qsort(tmp, cl->size, sizeof(struct pt), cmp_u64);
        mgpu[i] = fnv64(tmp, sizeof(struct pt)*cl->size, 0xcbf29ce484222325ULL);
    }
    free(tmp);
    qsort(hcpu, ncpu, sizeof(uint64_t), cmp_u64);
    qsort(hgpu, ngpu, sizeof(uint64_t), cmp_u64);
    qsort(mcpu, ncpu, sizeof(uint64_t), cmp_u64);
    qsort(mgpu, ngpu, sizeof(uint64_t), cmp_u64);
    long match = 0, mmatch = 0;
    {
        int i = 0, j = 0;
        while (i < ncpu && j < ngpu) {
            if (hcpu[i] == hgpu[j]) { match++; i++; j++; }
            else if (hcpu[i] < hgpu[j]) i++;
            else j++;
        }
        i = 0; j = 0;
        while (i < ncpu && j < ngpu) {
            if (mcpu[i] == mgpu[j]) { mmatch++; i++; j++; }
            else if (mcpu[i] < mgpu[j]) i++;
            else j++;
        }
    }
    fprintf(stderr, "[validate] clusters: cpu %d gpu %d, pts %ld/%ld, "
            "seq match %ld, multiset match %ld\n",
            ncpu, ngpu, cpu_pts, gpu_pts, match, mmatch);
    free(hcpu);
    free(hgpu);
    free(mcpu);
    free(mgpu);

    // 4. exact cluster pairing via the rep -> gpu-label bijection: for each
    // CPU cluster, recompute its GPU pair key and diff the point sequences
    if (getenv("APRILTAG_METAL_VALIDATE_DEEP")) {
        uint32_t vcol_base = cpu_row_off[h];
        uint32_t nnodes = vcol_base + h;
        // rebuild fwd: rep -> gpu label, from label images
        uint32_t *gpu_labels = m->labels.contents;
        uint32_t *fwd = malloc(sizeof(uint32_t)*nnodes);
        memset(fwd, 0xFF, sizeof(uint32_t)*nnodes);
        for (int y = 0; y < h; y++) {
            for (uint32_t ri = cpu_row_off[y]; ri < cpu_row_off[y+1]; ri++) {
                uint32_t rep = unionfind_get_representative(uf, ri);
                uint32_t g = gpu_labels[y*w + cpu_runs[ri].start];
                if (g != INVALID_LABEL)
                    fwd[rep] = g;
            }
            uint32_t rep = unionfind_get_representative(uf, vcol_base + y);
            uint32_t g = gpu_labels[y*w + (w-1)];
            if (g != INVALID_LABEL)
                fwd[rep] = g;
        }
        // index GPU clusters by key: keys live in m->keys[0] sorted; cluster
        // c covers records [starts[c], starts[c+1])
        ATState *stt = m->state_buf[slot].contents;
        uint32_t *gkeys = m->keys[0].contents;
        uint32_t *gstarts = m->starts.contents;
        uint32_t ngcl = stt->nclusters;
        // CPU clusters carry no id here (pt_list only), so recompute the id
        // from the first point's pixel pair is messy; instead diff by key
        // lookup: build map key -> gpu cluster index
        // (keys are unique per cluster; binary search over starts)
        int printed = 0;
        for (int i = 0; i < ncpu && printed < 6; i++) {
            struct pt_list *cl;
            zarray_get(cpu_clusters, i, &cl);
            // identify the pair key from the cluster's first point: the
            // point lies between two pixels; recover them from coords
            struct pt p0 = cl->pts[0];
            int px = p0.x, py = p0.y; // 2*x+dx, 2*y+dy
            int x0 = px/2, y0 = py/2; // source pixel (dx,dy in {0,1...})
            int dx = px - 2*x0, dy = py - 2*y0;
            // source pixel (x0,y0), target (x0+dx, y0+dy) -- but for
            // dx=-1 the encoding 2*x-1 gives x0 = (px+1)/2... handle:
            // gradient gx<0,gy>0 means dx=-1,dy=1
            if (p0.gx < 0 && p0.gy > 0 && dx == 1) {
                // px = 2*x0' - 1 with x0' = x0+1
                x0 = x0 + 1;
                dx = -1;
            }
            uint32_t l0 = gpu_labels[y0*w + x0];
            uint32_t l1 = gpu_labels[(y0+dy)*w + (x0+dx)];
            if (l0 == INVALID_LABEL || l1 == INVALID_LABEL)
                continue;
            uint32_t key = (MIN(l0, l1) << 15) | MAX(l0, l1);
            // binary search gpu clusters for key
            uint32_t loi = 0, hii = ngcl;
            int found = -1;
            while (loi < hii) {
                uint32_t mid = (loi + hii)/2;
                uint32_t km = gkeys[gstarts[mid]];
                if (km == key) { found = (int)mid; break; }
                if (km < key) loi = mid + 1; else hii = mid;
            }
            if (found < 0)
                continue;
            uint32_t gsz = gstarts[found+1] - gstarts[found];
            if ((int)gsz == cl->size) {
                struct pt *gpts = (struct pt *)((uint8_t *)m->arena[slot].contents +
                                  ((uint32_t *)m->arena_off[slot].contents)[found] + 8);
                if (memcmp(gpts, cl->pts, sizeof(struct pt)*cl->size) == 0)
                    continue; // identical
            }
            // mismatch: print both sequences around first difference
            struct pt *gpts = (struct pt *)((uint8_t *)m->arena[slot].contents +
                              ((uint32_t *)m->arena_off[slot].contents)[found] + 8);
            fprintf(stderr, "[deep] cluster key %u: cpu %d pts, gpu %u pts\n",
                    key, cl->size, gsz);
            int n = MIN(cl->size, (int)gsz);
            for (int j = 0; j < n; j++) {
                if (memcmp(&gpts[j], &cl->pts[j], sizeof(struct pt)) != 0) {
                    for (int k = MAX(0, j-2); k < MIN(n, j+4); k++)
                        fprintf(stderr, "[deep]   [%d] cpu (%d,%d,g %d,%d)  "
                                "gpu (%d,%d,g %d,%d)\n", k,
                                cl->pts[k].x, cl->pts[k].y, cl->pts[k].gx, cl->pts[k].gy,
                                gpts[k].x, gpts[k].y, gpts[k].gx, gpts[k].gy);
                    break;
                }
            }
            if (cl->size != (int)gsz) {
                // print tail of the longer one
                if ((int)gsz > cl->size)
                    for (int k = cl->size; k < MIN((int)gsz, cl->size + 4); k++)
                        fprintf(stderr, "[deep]   gpu extra [%d] (%d,%d,g %d,%d)\n",
                                k, gpts[k].x, gpts[k].y, gpts[k].gx, gpts[k].gy);
                else
                    for (int k = gsz; k < MIN(cl->size, (int)gsz + 4); k++)
                        fprintf(stderr, "[deep]   cpu extra [%d] (%d,%d,g %d,%d)\n",
                                k, cl->pts[k].x, cl->pts[k].y, cl->pts[k].gx, cl->pts[k].gy);
            }
            printed++;
        }
        free(fwd);
    }

    for (int i = 0; i < ncpu; i++) {
        struct pt_list *cl;
        zarray_get(cpu_clusters, i, &cl);
        free(cl);
    }
    zarray_destroy(cpu_clusters);
}
