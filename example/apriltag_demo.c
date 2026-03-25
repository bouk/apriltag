/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.

This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
    #include <unistd.h>
#endif

#include "apriltag.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"

#include "common/getopt.h"
#include "common/image_u8.h"
#include "common/pjpeg.h"
#include "common/zarray.h"

#define  HAMM_HIST_MAX 10

struct expected_detection {
    char file[512];
    int id;
    int hamming;
    float margin;
    double cx, cy;
    double p[4][2];
};

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double compute_median(double *vals, int n)
{
    qsort(vals, n, sizeof(double), cmp_double);
    if (n % 2 == 1) return vals[n / 2];
    return (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
}

static double compute_stddev(double *vals, int n)
{
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) {
        sum += vals[i];
        sum2 += vals[i] * vals[i];
    }
    double mean = sum / n;
    return sqrt(sum2 / n - mean * mean);
}

static int cmp_expected_detection(const void *a, const void *b)
{
    const struct expected_detection *da = a, *db = b;
    int c = strcmp(da->file, db->file);
    if (c != 0) return c;
    return da->id - db->id;
}

static struct expected_detection *load_expected_detections(const char *path, int *count)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "error: cannot open expected detections file: %s\n", path);
        exit(1);
    }

    int cap = 256;
    int n = 0;
    struct expected_detection *dets = calloc(cap, sizeof(struct expected_detection));

    char line[2048];
    // skip header
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        *count = 0;
        return dets;
    }

    while (fgets(line, sizeof(line), f)) {
        if (n >= cap) {
            cap *= 2;
            dets = realloc(dets, cap * sizeof(struct expected_detection));
        }
        struct expected_detection *d = &dets[n];
        int matched = sscanf(line, "%511[^\t]\t%d\t%d\t%f\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
                             d->file, &d->id, &d->hamming, &d->margin,
                             &d->cx, &d->cy,
                             &d->p[0][0], &d->p[0][1],
                             &d->p[1][0], &d->p[1][1],
                             &d->p[2][0], &d->p[2][1],
                             &d->p[3][0], &d->p[3][1]);
        if (matched == 14)
            n++;
    }

    fclose(f);
    qsort(dets, n, sizeof(struct expected_detection), cmp_expected_detection);
    *count = n;
    return dets;
}

static int validate_detections(const char *file, zarray_t *detections,
                               struct expected_detection *expected, int nexpected)
{
    // Build sorted list of actual detections for this file
    int nactual = zarray_size(detections);
    if (nactual <= 0) {
        // Report any expected detections as missing
        int mismatches = 0;
        for (int i = 0; i < nexpected; i++) {
            if (!strcmp(expected[i].file, file)) {
                fprintf(stderr, "MISSING %s: id %d\n", file, expected[i].id);
                mismatches++;
            }
        }
        return mismatches;
    }
    struct expected_detection *actual = calloc((size_t)nactual, sizeof(struct expected_detection));
    for (int i = 0; i < nactual; i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        strncpy(actual[i].file, file, sizeof(actual[i].file) - 1);
        actual[i].id = det->id;
        actual[i].hamming = det->hamming;
        actual[i].margin = det->decision_margin;
        actual[i].cx = det->c[0];
        actual[i].cy = det->c[1];
        for (int j = 0; j < 4; j++) {
            actual[i].p[j][0] = det->p[j][0];
            actual[i].p[j][1] = det->p[j][1];
        }
    }
    qsort(actual, nactual, sizeof(struct expected_detection), cmp_expected_detection);

    // Find expected detections for this file using binary search
    int exp_start = -1, exp_end = -1;
    for (int i = 0; i < nexpected; i++) {
        if (!strcmp(expected[i].file, file)) {
            if (exp_start < 0) exp_start = i;
            exp_end = i + 1;
        } else if (exp_start >= 0) {
            break;
        }
    }
    if (exp_start < 0) { exp_start = 0; exp_end = 0; }

    int nexp = exp_end - exp_start;
    int mismatches = 0;

    // Two-pointer merge on sorted id lists
    int ai = 0, ei = 0;
    while (ai < nactual && ei < nexp) {
        int act_id = actual[ai].id;
        int exp_id = expected[exp_start + ei].id;
        if (act_id < exp_id) {
            fprintf(stderr, "EXTRA   %s: id %d\n", file, act_id);
            mismatches++;
            ai++;
        } else if (act_id > exp_id) {
            fprintf(stderr, "MISSING %s: id %d\n", file, exp_id);
            mismatches++;
            ei++;
        } else {
            // Same id — check hamming and corners
            struct expected_detection *a = &actual[ai];
            struct expected_detection *e = &expected[exp_start + ei];
            if (a->hamming != e->hamming) {
                fprintf(stderr, "MISMATCH %s: id %d hamming %d (expected %d)\n",
                        file, act_id, a->hamming, e->hamming);
                mismatches++;
            }
            for (int j = 0; j < 4; j++) {
                double dx = fabs(a->p[j][0] - e->p[j][0]);
                double dy = fabs(a->p[j][1] - e->p[j][1]);
                if (dx > 1.0 || dy > 1.0) {
                    fprintf(stderr, "CORNER_DRIFT %s: id %d corner %d moved by (%.2f, %.2f)\n",
                            file, act_id, j, dx, dy);
                    mismatches++;
                    break;
                }
            }
            ai++;
            ei++;
        }
    }
    while (ai < nactual) {
        fprintf(stderr, "EXTRA   %s: id %d\n", file, actual[ai].id);
        mismatches++;
        ai++;
    }
    while (ei < nexp) {
        fprintf(stderr, "MISSING %s: id %d\n", file, expected[exp_start + ei].id);
        mismatches++;
        ei++;
    }

    free(actual);
    return mismatches;
}

int main(int argc, char *argv[])
{
    getopt_t *getopt = getopt_create();

    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
    getopt_add_int(getopt, 'i', "iters", "1", "Repeat processing on input set this many times");
    getopt_add_int(getopt, 't', "threads", "1", "Use this many CPU threads");
    getopt_add_int(getopt, 'a', "hamming", "1", "Detect tags with up to this many bit errors.");
    getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input; negative sharpens");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");
    getopt_add_int(getopt, '\0', "min-cluster-pixels", "24", "Reject quads containing too few pixels");
    getopt_add_int(getopt, '\0', "max-nmaxima", "10", "Max number of corner candidates to consider");
    getopt_add_double(getopt, '\0', "critical-rad", "0.174533", "Reject quads with angles close to straight or 180 deg (radians)");
    getopt_add_double(getopt, '\0', "max-line-fit-mse", "10.0", "Max mean squared error when fitting lines to contours");
    getopt_add_int(getopt, '\0', "min-white-black-diff", "5", "Minimum brightness difference between white and black models");
    getopt_add_bool(getopt, '\0', "deglitch", 0, "Deglitch thresholded image (useful for noisy images)");
    getopt_add_string(getopt, '\0', "save-detections", "", "Save detections to TSV file");
    getopt_add_string(getopt, '\0', "expected-detections", "", "Validate against expected detections TSV");
    getopt_add_string(getopt, '\0', "save-timing", "", "Save timing results to TSV file");

    if (!getopt_parse(getopt, argc, argv, 1) || getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options] <input files>\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }

    const zarray_t *inputs = getopt_get_extra_args(getopt);

    apriltag_family_t *tf = NULL;
    const char *famname = getopt_get_string(getopt, "family");
    if (!strcmp(famname, "tag36h11")) {
        tf = tag36h11_create();
    } else if (!strcmp(famname, "tag25h9")) {
        tf = tag25h9_create();
    } else if (!strcmp(famname, "tag16h5")) {
        tf = tag16h5_create();
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tf = tagCircle21h7_create();
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tf = tagCircle49h12_create();
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tf = tagStandard41h12_create();
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tf = tagStandard52h13_create();
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tf = tagCustom48h12_create();
    } else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family_bits(td, tf, getopt_get_int(getopt, "hamming"));

    switch(errno){
        case EINVAL:
            printf("\"hamming\" parameter is out-of-range.\n");
            exit(-1);
        case ENOMEM:
            printf("Unable to add family to detector due to insufficient memory to allocate the tag-family decoder. Try reducing \"hamming\" from %d or choose an alternative tag family.\n", getopt_get_int(getopt, "hamming"));
            exit(-1);
    }

    td->quad_decimate = getopt_get_double(getopt, "decimate");
    td->quad_sigma = getopt_get_double(getopt, "blur");
    td->nthreads = getopt_get_int(getopt, "threads");
    td->debug = getopt_get_bool(getopt, "debug");
    td->refine_edges = getopt_get_bool(getopt, "refine-edges");
    td->qtp.min_cluster_pixels = getopt_get_int(getopt, "min-cluster-pixels");
    td->qtp.max_nmaxima = getopt_get_int(getopt, "max-nmaxima");
    td->qtp.cos_critical_rad = cos(getopt_get_double(getopt, "critical-rad"));
    td->qtp.max_line_fit_mse = getopt_get_double(getopt, "max-line-fit-mse");
    td->qtp.min_white_black_diff = getopt_get_int(getopt, "min-white-black-diff");
    td->qtp.deglitch = getopt_get_bool(getopt, "deglitch");

    int quiet = getopt_get_bool(getopt, "quiet");
    int maxiters = getopt_get_int(getopt, "iters");

    int do_save_detections = getopt_was_specified(getopt, "save-detections");
    int do_validate = getopt_was_specified(getopt, "expected-detections");
    int do_save_timing = getopt_was_specified(getopt, "save-timing");

    const char *save_detections_path = getopt_get_string(getopt, "save-detections");
    const char *expected_detections_path = getopt_get_string(getopt, "expected-detections");
    const char *save_timing_path = getopt_get_string(getopt, "save-timing");

    // Load expected detections for validation
    struct expected_detection *expected_dets = NULL;
    int nexpected = 0;
    if (do_validate) {
        expected_dets = load_expected_detections(expected_detections_path, &nexpected);
    }

    // Open detections output file
    FILE *det_file = NULL;
    if (do_save_detections) {
        det_file = fopen(save_detections_path, "w");
        if (!det_file) {
            fprintf(stderr, "error: cannot open detections output file: %s\n", save_detections_path);
            exit(1);
        }
        fprintf(det_file, "file\tid\thamming\tmargin\tcx\tcy\tp0x\tp0y\tp1x\tp1y\tp2x\tp2y\tp3x\tp3y\n");
    }

    int num_inputs = zarray_size(inputs);

    // Timing storage: when benchmarking, add a warmup iteration
    int do_warmup = do_save_timing && maxiters > 0;
    int total_runs = maxiters + (do_warmup ? 1 : 0);
    double *timings = NULL;
    int *detection_counts = NULL;
    int *nquads_last = NULL;
    if (do_save_timing) {
        timings = calloc(maxiters * num_inputs, sizeof(double));
        detection_counts = calloc(num_inputs, sizeof(int));
        nquads_last = calloc(num_inputs, sizeof(int));
    }

    int validation_failures = 0;

    for (int iter = 0; iter < total_runs; iter++) {

        int is_warmup = do_warmup && (iter == 0);
        int timed_iter = iter - (do_warmup ? 1 : 0);
        int is_last = (iter == total_runs - 1);

        int total_quads = 0;
        int total_hamm_hist[HAMM_HIST_MAX];
        memset(total_hamm_hist, 0, sizeof(int)*HAMM_HIST_MAX);
        double total_time = 0;

        if (total_runs > 1) {
            if (is_warmup)
                printf("warmup\n");
            else
                printf("iter %d / %d\n", timed_iter + 1, maxiters);
        }

        for (int input = 0; input < num_inputs; input++) {

            int hamm_hist[HAMM_HIST_MAX];
            memset(hamm_hist, 0, sizeof(hamm_hist));

            char *path;
            zarray_get(inputs, input, &path);
            if (!quiet)
                printf("loading %s\n", path);
            else
                printf("%20s ", path);

            image_u8_t *im = NULL;
            if (str_ends_with(path, "pnm") || str_ends_with(path, "PNM") ||
                str_ends_with(path, "pgm") || str_ends_with(path, "PGM"))
                im = image_u8_create_from_pnm(path);
            else if (str_ends_with(path, "jpg") || str_ends_with(path, "JPG")) {
                int err = 0;
                pjpeg_t *pjpeg = pjpeg_create_from_file(path, 0, &err);
                if (pjpeg == NULL) {
                    printf("pjpeg failed to load: %s, error %d\n", path, err);
                    continue;
                }

                if (1) {
                    im = pjpeg_to_u8_baseline(pjpeg);
                } else {
                    printf("illumination invariant\n");

                    image_u8x3_t *imc =  pjpeg_to_u8x3_baseline(pjpeg);

                    im = image_u8_create(imc->width, imc->height);

                    for (int y = 0; y < imc->height; y++) {
                        for (int x = 0; x < imc->width; x++) {
                            double r = imc->buf[y*imc->stride + 3*x + 0] / 255.0;
                            double g = imc->buf[y*imc->stride + 3*x + 1] / 255.0;
                            double b = imc->buf[y*imc->stride + 3*x + 2] / 255.0;

                            double alpha = 0.42;
                            double v = 0.5 + log(g) - alpha*log(b) - (1-alpha)*log(r);
                            int iv = v * 255;
                            if (iv < 0)
                                iv = 0;
                            if (iv > 255)
                                iv = 255;

                            im->buf[y*im->stride + x] = iv;
                        }
                    }
                    image_u8x3_destroy(imc);
                    if (td->debug)
                        image_u8_write_pnm(im, "debug_invariant.pnm");
                }

                pjpeg_destroy(pjpeg);
            }

            if (im == NULL) {
                printf("couldn't load %s\n", path);
                continue;
            }

            printf("image: %s %dx%d\n", path, im->width, im->height);

            zarray_t *detections = apriltag_detector_detect(td, im);

            if (errno == EAGAIN) {
                printf("Unable to create the %d threads requested.\n",td->nthreads);
                exit(-1);
            }

            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t *det;
                zarray_get(detections, i, &det);

                if (!quiet)
                    printf("detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                           i, det->family->nbits, det->family->h, det->id, det->hamming, det->decision_margin);

                hamm_hist[det->hamming]++;
                total_hamm_hist[det->hamming]++;
            }

            // Save detections on last iteration
            if (is_last && det_file) {
                for (int i = 0; i < zarray_size(detections); i++) {
                    apriltag_detection_t *det;
                    zarray_get(detections, i, &det);
                    fprintf(det_file, "%s\t%d\t%d\t%.3f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                            path, det->id, det->hamming, det->decision_margin,
                            det->c[0], det->c[1],
                            det->p[0][0], det->p[0][1],
                            det->p[1][0], det->p[1][1],
                            det->p[2][0], det->p[2][1],
                            det->p[3][0], det->p[3][1]);
                }
            }

            // Validate detections on last iteration
            if (is_last && do_validate) {
                validation_failures += validate_detections(path, detections,
                                                          expected_dets, nexpected);
            }

            // Store timing
            if (!is_warmup && timings) {
                timings[timed_iter * num_inputs + input] = timeprofile_total_utime(td->tp) / 1.0E3;
            }
            if (is_last && detection_counts) {
                detection_counts[input] = zarray_size(detections);
                nquads_last[input] = td->nquads;
            }

            apriltag_detections_destroy(detections);

            if (!quiet) {
                timeprofile_display(td->tp);
            }

            total_quads += td->nquads;

            if (!quiet)
                printf("hamm ");

            for (int i = 0; i < HAMM_HIST_MAX; i++)
                printf("%5d ", hamm_hist[i]);

            double t =  timeprofile_total_utime(td->tp) / 1.0E3;
            total_time += t;
            printf("%12.3f ", t);
            printf("%5d", td->nquads);

            printf("\n");

            image_u8_destroy(im);
        }


        printf("Summary\n");

        printf("hamm ");

        for (int i = 0; i < HAMM_HIST_MAX; i++)
            printf("%5d ", total_hamm_hist[i]);
        printf("%12.3f ", total_time);
        printf("%5d", total_quads);
        printf("\n");

    }

    // Write timing TSV
    if (do_save_timing && maxiters > 0) {
        FILE *tf_out = fopen(save_timing_path, "w");
        if (!tf_out) {
            fprintf(stderr, "error: cannot open timing output file: %s\n", save_timing_path);
            exit(1);
        }
        fprintf(tf_out, "file\titers\tmedian_ms\tmin_ms\tmax_ms\tstddev_ms\tnquads\tdetections\n");

        double *img_times = malloc(maxiters * sizeof(double));
        for (int input = 0; input < num_inputs; input++) {
            char *path;
            zarray_get(inputs, input, &path);

            for (int j = 0; j < maxiters; j++)
                img_times[j] = timings[j * num_inputs + input];

            double median = compute_median(img_times, maxiters);
            // After compute_median, img_times is sorted
            double min_ms = img_times[0];
            double max_ms = img_times[maxiters - 1];
            double stddev = maxiters > 1 ? compute_stddev(img_times, maxiters) : 0.0;

            fprintf(tf_out, "%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\n",
                    path, maxiters, median, min_ms, max_ms, stddev,
                    nquads_last[input], detection_counts[input]);
        }
        free(img_times);
        fclose(tf_out);
    }

    if (det_file)
        fclose(det_file);

    free(timings);
    free(detection_counts);
    free(nquads_last);
    free(expected_dets);

    // don't deallocate contents of inputs; those are the argv
    apriltag_detector_destroy(td);

    if (!strcmp(famname, "tag36h11")) {
        tag36h11_destroy(tf);
    } else if (!strcmp(famname, "tag25h9")) {
        tag25h9_destroy(tf);
    } else if (!strcmp(famname, "tag16h5")) {
        tag16h5_destroy(tf);
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tagCircle21h7_destroy(tf);
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tagCircle49h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tagStandard41h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tagStandard52h13_destroy(tf);
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tagCustom48h12_destroy(tf);
    }

    getopt_destroy(getopt);

    return validation_failures > 0 ? 1 : 0;
}
