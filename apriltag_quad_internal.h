/* Internal structures and stage entry points shared between the
 * quad-threshold pipeline and the Metal GPU front-end. Not installed;
 * not part of the public API. */
#pragma once

#include <stdint.h>

#include "apriltag.h"
#include "common/image_u8.h"
#include "common/unionfind.h"
#include "common/zarray.h"

#ifdef __cplusplus
extern "C" {
#endif

// a maximal horizontal segment of equal non-127 pixels, x in [0, w-2]
// (the last column never participates in runs; it is only reachable as a
// diagonal neighbor of a white run ending at w-2)
struct row_run
{
    uint16_t start, end; // inclusive
    uint8_t v;
};

// one boundary point of a gradient cluster
struct pt
{
    // Note: these represent 2*actual value.
    uint16_t x, y;
    int16_t gx, gy;
};

// a finished cluster: header and points in one allocation
struct pt_list
{
    int size;
    int pad; // keep pts 8-byte aligned
    struct pt pts[];
};

// stage entry points (defined in apriltag_quad_thresh.c)
image_u8_t *threshold(apriltag_detector_t *td, image_u8_t *im,
                      struct row_run **runs_out, uint32_t **row_off_out);
unionfind_t *connected_components(apriltag_detector_t *td, image_u8_t *threshim,
                                  int w, int h, int ts,
                                  struct row_run *runs, uint32_t *row_off);
zarray_t *gradient_clusters(apriltag_detector_t *td, image_u8_t *threshim,
                            int w, int h, int ts, unionfind_t *uf,
                            struct row_run *runs, uint32_t *row_off);
zarray_t *fit_quads(apriltag_detector_t *td, int w, int h, zarray_t *clusters,
                    image_u8_t *im);

#ifdef __cplusplus
}
#endif
