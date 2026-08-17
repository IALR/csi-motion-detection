#include "csi_standalone.h"

#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "csi_sa";

// Frames to discard after CSI starts, before trusting anything for
// calibration. The first packets after association are unrepresentative -
// rate adaptation and AGC are still settling - and folding them into the
// baseline biases every later comparison against it.
#define WARMUP_FRAMES 30

// How often to print a status line even when nothing changed, so there is
// always visible evidence the detector is alive amid the CSI_AMP flood.
#define STATUS_LOG_PERIOD_S 5.0f

// Big buffers are static, not stack or heap: ~107KB total, which is
// affordable against the ESP32-S3's SRAM but would blow any task stack.
// (If the HTTP server later runs short of memory, these are the obvious
// candidates to move into the 8MB PSRAM - they are touched at most 10x a
// second, so the slower access would not matter.)
static struct {
    bool             inited;
    csi_sa_state_t   state;
    int              warmup_seen;

    // Sliding scoring window, kept as a ring buffer.
    float            win_amps[CSI_WINDOW_FRAMES][CSI_NS];
    float            win_rssi[CSI_WINDOW_FRAMES];
    int              win_count;
    int              win_head;

    // Startup calibration block.
    float            calib_amps[CSI_CALIB_FRAMES][CSI_NS];
    float            calib_rssi[CSI_CALIB_FRAMES];
    int              calib_count;

    csi_baseline_t   baseline;
    csi_smoother_t   smoother;
    csi_roller_t     roller;

    int              confirmed;
    float            confidence;
    int              raw_prediction;
    float            raw_confidence;
    float            rssi;
    float            energy;
    float            prev_amps[CSI_NS];
    bool             have_prev;

    float            noise_relative;
    csi_noise_level_t noise_level;
    uint32_t         recal_count;
    int64_t          last_recal_us;
    int64_t          last_log_us;
    bool             mismatch_warned;
    bool             subcarrier_mismatch;
    int              n_subcarriers;
} s;

static float g_features[CSI_N_FEATURES];

static void enter_calibrating(void)
{
    s.state = CSI_SA_CALIBRATING;
    s.calib_count = 0;
    s.win_count = 0;
    s.win_head = 0;
    s.confirmed = -1;
    s.confidence = 0.0f;
    csi_smoother_reset(&s.smoother);
    csi_roller_reset(&s.roller);
    ESP_LOGW(TAG, ">>> LEAVE THE ROOM - calibrating on the next %d frames (%.0fs) <<<",
             CSI_CALIB_FRAMES, (float)CSI_CALIB_FRAMES / CSI_FRAME_HZ);
}

void csi_standalone_init(void)
{
    memset(&s, 0, sizeof(s));
    s.confirmed = -1;
    s.state = CSI_SA_WARMUP;
    s.inited = true;
    ESP_LOGI(TAG, "standalone detection armed: %d subcarriers, %d-frame window, "
                  "%d-frame calibration",
             CSI_NS, CSI_WINDOW_FRAMES, CSI_CALIB_FRAMES);
}

void csi_standalone_request_recalibration(void)
{
    if (!s.inited) return;
    ESP_LOGI(TAG, "recalibration requested");
    enter_calibrating();
}

void csi_standalone_get_status(csi_sa_status_t *out)
{
    if (!out) return;
    out->state = s.state;
    out->calib_buffered = s.calib_count;
    out->calib_needed = CSI_CALIB_FRAMES;
    out->calib_remaining_s = (float)(CSI_CALIB_FRAMES - s.calib_count) / CSI_FRAME_HZ;
    out->confirmed = s.confirmed;
    out->confidence = s.confidence;
    out->raw_prediction = s.raw_prediction;
    out->raw_confidence = s.raw_confidence;
    out->rssi = s.rssi;
    out->energy = s.energy;
    out->noise_relative = s.noise_relative;
    out->noise_level = s.noise_level;
    out->recal_count = s.recal_count;
    out->last_recal_us = s.last_recal_us;
    out->subcarrier_mismatch = s.subcarrier_mismatch;
    out->n_subcarriers = s.n_subcarriers;
}

void csi_standalone_on_frame(const float *amps, int n_sub, float rssi)
{
    if (!s.inited) return;

    s.n_subcarriers = n_sub;

    // A stream whose width differs from the model's can never produce a valid
    // feature vector, so predictions are skipped rather than fed garbage. This
    // happens when the AP renegotiates its channel width (20MHz/128 <->
    // 40MHz/192) - a failure this project has hit for real, twice.
    s.subcarrier_mismatch = (n_sub != CSI_NS);
    if (s.subcarrier_mismatch) {
        if (!s.mismatch_warned) {
            s.mismatch_warned = true;
            ESP_LOGE(TAG, "stream has %d subcarriers, model expects %d - standalone "
                          "predictions disabled until it matches (the AP likely "
                          "changed channel width; a 20MHz-only AP is required)",
                     n_sub, CSI_NS);
        }
        return;
    }
    if (s.mismatch_warned) {
        s.mismatch_warned = false;
        ESP_LOGW(TAG, "subcarrier count back to %d - recalibrating", CSI_NS);
        enter_calibrating();
        return;
    }

    s.rssi = rssi;

    // Frame-to-frame amplitude change, the same quantity the dashboard plots.
    if (s.have_prev) {
        float acc = 0.0f;
        for (int i = 0; i < CSI_NS; i++) {
            float d = amps[i] - s.prev_amps[i];
            acc += d < 0 ? -d : d;
        }
        s.energy = acc / (float)CSI_NS;
    }
    memcpy(s.prev_amps, amps, sizeof(float) * CSI_NS);
    s.have_prev = true;

    switch (s.state) {
    case CSI_SA_WARMUP:
        if (++s.warmup_seen >= WARMUP_FRAMES) {
            enter_calibrating();
        }
        return;

    case CSI_SA_CALIBRATING:
        memcpy(s.calib_amps[s.calib_count], amps, sizeof(float) * CSI_NS);
        s.calib_rssi[s.calib_count] = rssi;
        s.calib_count++;
        if (s.calib_count >= CSI_CALIB_FRAMES) {
            csi_compute_baseline((const float *)s.calib_amps, s.calib_rssi,
                                 CSI_CALIB_FRAMES, &s.baseline);
            csi_roller_set_floor(&s.roller, &s.baseline);
            s.noise_level = csi_assess_noise(&s.baseline, &s.noise_relative);
            s.last_recal_us = esp_timer_get_time();
            s.state = CSI_SA_RUNNING;

            static const char *lvl[] = { "quiet", "moderate", "loud" };
            ESP_LOGW(TAG, "calibrated on %d frames | noise %.4f (%s) | now detecting",
                     CSI_CALIB_FRAMES, s.noise_relative, lvl[s.noise_level]);
            if (s.noise_level == CSI_NOISE_LOUD) {
                // Same honesty as the dashboard: a loud room is a RISK
                // indicator, not a prediction of failure. Sessions recorded in
                // this regime scored 85-95% against 95-99% in quiet rooms.
                ESP_LOGW(TAG, "noisy environment: an empty room here looks nearly as "
                              "disturbed as an occupied one. Expect more variable "
                              "results and false MOVING readings. Common causes: "
                              "Wi-Fi congestion, a fan/AC, or something moving "
                              "during calibration.");
            }
        }
        return;

    case CSI_SA_RUNNING:
        break;
    }

    // ---- sliding window, predicting every frame once full (as the PC does) --
    memcpy(s.win_amps[s.win_head], amps, sizeof(float) * CSI_NS);
    s.win_rssi[s.win_head] = rssi;
    s.win_head = (s.win_head + 1) % CSI_WINDOW_FRAMES;
    if (s.win_count < CSI_WINDOW_FRAMES) {
        s.win_count++;
        if (s.win_count < CSI_WINDOW_FRAMES) return;
    }

    // Unwrap the ring into chronological order; csi_make_features expects the
    // oldest frame first, and feeding it rotated would scramble the
    // frame-to-frame differences that motion energy is built from.
    static float ordered_amps[CSI_WINDOW_FRAMES][CSI_NS];
    static float ordered_rssi[CSI_WINDOW_FRAMES];
    for (int i = 0; i < CSI_WINDOW_FRAMES; i++) {
        int src = (s.win_head + i) % CSI_WINDOW_FRAMES;
        memcpy(ordered_amps[i], s.win_amps[src], sizeof(float) * CSI_NS);
        ordered_rssi[i] = s.win_rssi[src];
    }

    csi_make_features((const float *)ordered_amps, ordered_rssi,
                      CSI_WINDOW_FRAMES, &s.baseline, g_features);

    float p1 = 0.0f;
    int pred = csi_predict(g_features, &p1);
    s.raw_prediction = pred;
    s.raw_confidence = (pred == csi_classes[1]) ? p1 : (1.0f - p1);

    int prev_confirmed = s.confirmed;
    s.confirmed = csi_smoother_update(&s.smoother, pred);
    s.confidence = s.smoother.vote_fraction;

    // Log on every state CHANGE, and also on a periodic heartbeat. The change
    // log alone is nearly invisible in practice: each CSI_AMP line is ~750
    // bytes at 10Hz, so a single line that only appears on a transition
    // scrolls past instantly and a quiet room produces no output at all -
    // leaving no way to tell "detecting, all empty" from "not running".
    bool changed = (s.confirmed != prev_confirmed && s.confirmed >= 0);
    int64_t now_us = esp_timer_get_time();
    bool heartbeat = (now_us - s.last_log_us) >= (int64_t)(STATUS_LOG_PERIOD_S * 1000000);

    if (changed || heartbeat) {
        s.last_log_us = now_us;
        const char *state = (s.confirmed < 0) ? "warming up"
                          : (s.confirmed == csi_classes[1] ? "MOVING" : "EMPTY");
        ESP_LOGW(TAG, "[%s] %s | vote %.0f%% | raw %s p=%.2f | rssi %.0f | "
                      "energy %.2f | noise %.4f | recal %lu",
                 changed ? "CHANGE" : "status", state,
                 s.confidence * 100.0f,
                 s.raw_prediction == csi_classes[1] ? "MOVING" : "EMPTY",
                 s.raw_confidence, s.rssi, s.energy, s.noise_relative,
                 (unsigned long)s.recal_count);
    }

    // Roll the baseline forward during sustained confirmed-empty stretches,
    // mirroring train_model.py's per-empty-block recalibration. Fed the LATEST
    // frame, so what accumulates is the most recent clean data.
    if (s.confirmed >= 0) {
        if (csi_roller_observe(&s.roller, s.confirmed, amps, rssi, &s.baseline)) {
            s.recal_count++;
            s.last_recal_us = esp_timer_get_time();
            s.noise_level = csi_assess_noise(&s.baseline, &s.noise_relative);
            ESP_LOGI(TAG, "rolling recalibration #%lu | noise %.4f",
                     (unsigned long)s.recal_count, s.noise_relative);
        }
    }
}
