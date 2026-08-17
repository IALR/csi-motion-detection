// CSI feature extraction and inference, on-device.
//
// This is a direct port of csi_common.py. That file is the project's single
// source of truth for feature math, and the whole point of it is that offline
// training and online inference can never silently diverge - so this port has
// to stay bit-faithful, not merely "close". csi_selftest.c replays real
// recorded windows through it and compares against what scikit-learn produced
// on the PC for those exact windows; if this file drifts, that test fails.
//
// Deliberate correspondences with the Python, each of which changes results if
// broken:
//   * std is POPULATION std (divide by N), matching numpy's default ddof=0.
//   * energy is the mean over subcarriers of |frame[i] - frame[i-1]|, then
//     averaged over the window's N-1 differences - not over N.
//   * the baseline's amp_std_floor is 10% of the MEDIAN baseline std, which
//     stops structurally dead guard-band subcarriers producing runaway ratios.
//   * feature ORDER must match csi_common.feature_names() exactly:
//       [0 .. NS-1]        amp_mean_delta per subcarrier
//       [NS .. 2*NS-1]     amp_std_ratio per subcarrier
//       [2*NS + 0]         motion_energy_mean_ratio
//       [2*NS + 1]         motion_energy_std_ratio
//       [2*NS + 2]         rssi_mean_delta
//       [2*NS + 3]         rssi_std_ratio
//       [2*NS + 4 .. +9]   the six order-invariant summaries
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "csi_model_data.h"

// CSI_WINDOW_FRAMES and CSI_CALIB_FRAMES come from csi_model_data.h, where the
// exporter emits them as plain integers computed by the same Python rounding
// the training pipeline uses. Deriving them here from the float seconds would
// not be an integer constant expression, so C would reject them as array sizes.
#define CSI_NS          CSI_N_SUBCARRIERS

// csi_raw_window_stats() is called with BOTH a scoring window (8 frames) and a
// calibration block (100 frames), so its scratch buffers must be sized for the
// larger of the two. Sizing them for the window alone overflows the stack the
// moment a baseline is computed.
#define CSI_MAX_FRAMES  (CSI_CALIB_FRAMES > CSI_WINDOW_FRAMES ? CSI_CALIB_FRAMES \
                                                              : CSI_WINDOW_FRAMES)

// Must mirror csi_common.py's module-level constants.
#define CSI_EPS              1e-6f
#define CSI_TOP_K            10
#define CSI_ELEVATED_RATIO   1.5f
#define CSI_ENERGY_STD_FLOOR 0.05f

// One calibration baseline: what an empty room looks like right now.
typedef struct {
    float amp_mean[CSI_NS];
    float amp_std[CSI_NS];
    float energy_mean;
    float energy_std;
    float rssi_mean;
    float rssi_std;
    float amp_std_floor;
    float rssi_std_floor;
    bool  valid;
} csi_baseline_t;

// Unnormalised stats for one window, before calibration is applied.
typedef struct {
    float amp_mean[CSI_NS];
    float amp_std[CSI_NS];
    float energy_mean;
    float energy_std;
    float rssi_mean;
    float rssi_std;
} csi_stats_t;

// amps: n_frames rows of CSI_NS amplitudes, row-major.
void csi_raw_window_stats(const float *amps, const float *rssi, int n_frames,
                          csi_stats_t *out);

// Baseline from a block of confirmed-empty frames (adds the two std floors).
void csi_compute_baseline(const float *amps, const float *rssi, int n_frames,
                          csi_baseline_t *out);

// stats + baseline -> the CSI_N_FEATURES vector, in feature_names() order.
void csi_calibrate_features(const csi_stats_t *s, const csi_baseline_t *b,
                            float *features);

// Convenience: window -> features in one call (mirrors make_features()).
void csi_make_features(const float *amps, const float *rssi, int n_frames,
                       const csi_baseline_t *b, float *features);

// Random Forest inference. Returns the predicted class LABEL (0 or 2, from
// csi_classes[]) and optionally the averaged probability of the second class.
int csi_predict(const float *features, float *p1_out);

// ---- prediction smoothing (mirrors PredictionSmoother, majority of N) ----
#define CSI_SMOOTH_SIZE 5

typedef struct {
    int8_t history[CSI_SMOOTH_SIZE];
    int    count;      // how many slots are filled
    int    head;       // ring buffer write position
    int    confirmed;  // -1 until a majority has ever been reached
    float  vote_fraction;
} csi_smoother_t;

void csi_smoother_reset(csi_smoother_t *s);
// Returns the confirmed label, or -1 if no majority has been reached yet.
int  csi_smoother_update(csi_smoother_t *s, int raw_label);

// ---- rolling recalibration (mirrors RollingCalibrator) ----
// Blends toward fresh empty samples but never below a fraction of the original
// startup baseline, so one unusually quiet sample cannot ratchet sensitivity up.
#define CSI_BLEND_ALPHA      0.3f
#define CSI_MIN_STD_FRACTION 0.6f

typedef struct {
    float  amp_sum[CSI_NS];
    float  amp_sq_sum[CSI_NS];
    float  rssi_buf[CSI_CALIB_FRAMES];
    float  amp_buf[CSI_CALIB_FRAMES][CSI_NS];
    int    count;
    csi_baseline_t floor_ref;   // the startup calibration; never blended down past this
    bool   have_floor;
} csi_roller_t;

void csi_roller_reset(csi_roller_t *r);
void csi_roller_set_floor(csi_roller_t *r, const csi_baseline_t *startup);
// Feed one frame once a CONFIRMED label exists. Returns true and updates
// `baseline` in place when a fresh blended baseline is ready.
bool csi_roller_observe(csi_roller_t *r, int confirmed_label,
                        const float *amps, float rssi, csi_baseline_t *baseline);

// ---- empty-room noise floor (mirrors assess_noise_floor) ----
#define CSI_NOISE_REL_QUIET 0.06f
#define CSI_NOISE_REL_LOUD  0.15f

typedef enum { CSI_NOISE_QUIET = 0, CSI_NOISE_MODERATE, CSI_NOISE_LOUD } csi_noise_level_t;

// Scale-free: jitter as a fraction of received amplitude, averaged over ACTIVE
// subcarriers only. Comparing raw jitter between boards is wrong - a board
// receiving half the signal shows half the jitter and looks misleadingly clean.
csi_noise_level_t csi_assess_noise(const csi_baseline_t *b, float *relative_out);
