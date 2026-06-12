/* Metal GPU front-end for the AprilTag detector (Apple Silicon).
 *
 * Replaces threshold + connected components + gradient clustering with
 * GPU compute; fit_quads and decode stay on the CPU. Enable by setting
 * the environment variable APRILTAG_METAL=1 before creating a detector.
 * APRILTAG_METAL_VALIDATE=1 additionally runs the CPU reference stages
 * each frame and compares intermediates (slow; for development).
 */
#pragma once

#include "apriltag.h"
#include "common/image_u8.h"
#include "common/zarray.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apriltag_metal apriltag_metal_t;

// returns NULL when no Metal device is available or shader compilation
// fails (callers fall back to the CPU pipeline)
apriltag_metal_t *apriltag_metal_create(void);
void apriltag_metal_destroy(apriltag_metal_t *m);

// run the GPU front-end; returns a zarray of struct pt_list* whose
// allocations are owned by the metal context (do NOT free the elements;
// they are valid until the next call or destroy)
zarray_t *apriltag_metal_clusters(apriltag_metal_t *m, apriltag_detector_t *td,
                                  image_u8_t *im);

// asynchronously start the front-end for an image that will be passed to
// apriltag_detector_detect next; apriltag_metal_clusters then only waits
// for the in-flight work instead of running it on the critical path.
// Called via apriltag_detector_detect_prepare.
void apriltag_metal_frontend_begin(apriltag_metal_t *m, apriltag_detector_t *td,
                                   image_u8_t *im);

#ifdef __cplusplus
}
#endif
