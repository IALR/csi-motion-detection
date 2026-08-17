// Proves the on-device pipeline reproduces the PC's.
//
// There is no host C compiler available for this project, so the port cannot
// be tested by compiling it on the laptop. Running the check on the device is
// the stronger test regardless: it exercises the real single-precision FPU and
// the real xtensa compiler, which is exactly where a float-heavy port like
// this goes wrong.
//
// Every vector supplies only RAW amplitude frames. The device recomputes the
// calibration baseline, all 266 features and the forest prediction from
// scratch, so a fault anywhere in the chain surfaces here rather than as a
// mysteriously worse detector later.

#include "csi_selftest.h"

#include <math.h>
#include <stdio.h>

#include "esp_log.h"

#include "csi_features.h"
#include "csi_testvectors.h"

static const char *TAG = "csi_selftest";

// float32 cannot reproduce float64 bit-for-bit, so features are compared with
// a relative tolerance. The PREDICTION has no tolerance: a differing label
// means the port is untrustworthy however close the numbers look. (On the PC,
// float32 vs float64 gave 0 disagreements across all 3302 recorded windows,
// so demanding an exact label match is a fair bar, not a lucky one.)
#define FEATURE_REL_TOL 1e-3f
#define FEATURE_ABS_TOL 1e-4f

static bool close_enough(float got, float want)
{
    float diff = fabsf(got - want);
    if (diff <= FEATURE_ABS_TOL) return true;
    float scale = fabsf(want);
    return diff <= FEATURE_REL_TOL * (scale > 1.0f ? scale : 1.0f);
}

bool csi_selftest_run(void)
{
    ESP_LOGI(TAG, "=== on-device parity check vs the PC pipeline ===");

    // Compile-time agreement between the model header and the vectors. If
    // these ever disagree the vectors were generated for a different model.
    if (CSI_TV_SUBCARRIERS != CSI_NS) {
        ESP_LOGE(TAG, "subcarrier mismatch: vectors %d, model %d",
                 CSI_TV_SUBCARRIERS, CSI_NS);
        return false;
    }
    if (CSI_TV_N_FEATURES != CSI_N_FEATURES) {
        ESP_LOGE(TAG, "feature-count mismatch: vectors %d, model %d",
                 CSI_TV_N_FEATURES, CSI_N_FEATURES);
        return false;
    }

    // Recompute the baseline from the raw calibration frames, exactly as the
    // live path will after the "leave the room" window.
    static csi_baseline_t baseline;
    csi_compute_baseline(csi_tv_calib_amps, csi_tv_calib_rssi,
                         CSI_TV_CALIB_FRAMES, &baseline);

    float rel = 0.0f;
    csi_noise_level_t lvl = csi_assess_noise(&baseline, &rel);
    static const char *lvl_name[] = { "quiet", "moderate", "loud" };
    ESP_LOGI(TAG, "baseline from %d frames: noise %.4f (%s), energy_mean %.4f",
             CSI_TV_CALIB_FRAMES, rel, lvl_name[lvl], baseline.energy_mean);

    static float fv[CSI_N_FEATURES];
    int pred_ok = 0, feat_ok = 0;
    float worst_diff = 0.0f;
    int worst_idx = -1, worst_vec = -1;

    for (int v = 0; v < CSI_TV_COUNT; v++) {
        csi_make_features(csi_tv_win_amps[v], csi_tv_win_rssi[v],
                          CSI_TV_WINDOW_FRAMES, &baseline, fv);

        int bad = 0;
        for (int i = 0; i < CSI_N_FEATURES; i++) {
            float want = csi_tv_expect_features[v][i];
            if (!close_enough(fv[i], want)) {
                bad++;
                float d = fabsf(fv[i] - want);
                if (d > worst_diff) { worst_diff = d; worst_idx = i; worst_vec = v; }
            }
        }
        if (bad == 0) feat_ok++;

        float p1 = 0.0f;
        int pred = csi_predict(fv, &p1);
        bool match = (pred == csi_tv_expect_pred[v]);
        if (match) pred_ok++;

        ESP_LOGI(TAG, "  vec %2d: pred %d (want %d) %s | p1=%.4f | true=%d | "
                      "features off: %d",
                 v, pred, csi_tv_expect_pred[v], match ? "OK " : "BAD",
                 p1, csi_tv_true_label[v], bad);
    }

    ESP_LOGI(TAG, "predictions matching sklearn : %d/%d", pred_ok, CSI_TV_COUNT);
    ESP_LOGI(TAG, "vectors with all features in tolerance: %d/%d", feat_ok, CSI_TV_COUNT);
    if (worst_vec >= 0) {
        ESP_LOGW(TAG, "largest feature deviation %.6g at vector %d feature %d",
                 worst_diff, worst_vec, worst_idx);
    }

    bool pass = (pred_ok == CSI_TV_COUNT);
    if (pass) {
        ESP_LOGI(TAG, "=== PASS - the device reproduces the PC's predictions ===");
    } else {
        ESP_LOGE(TAG, "=== FAIL - %d/%d predictions differ. Do NOT trust on-device "
                      "detection until this is fixed. ===",
                 CSI_TV_COUNT - pred_ok, CSI_TV_COUNT);
    }
    return pass;
}
