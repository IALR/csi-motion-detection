// Standalone detection: calibrate and predict on the board itself.
//
// ADDITIVE, never a replacement. The firmware keeps printing CSI_AMP lines
// over serial exactly as before, so csi_live_server.py and the data collector
// continue to work unchanged while this runs alongside. If standalone
// detection misbehaves it can be ignored entirely - the PC path is untouched.
//
// Mirrors what csi_live_server.py's pump() does per node:
//   WARMUP      - let the stream settle after CSI starts
//   CALIBRATING - buffer CSI_CALIB_FRAMES of (assumed empty) frames
//   RUNNING     - slide a CSI_WINDOW_FRAMES window, predict every frame,
//                 smooth, and roll the baseline forward on confirmed-empty
//                 stretches
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "csi_features.h"

typedef enum {
    CSI_SA_WARMUP = 0,
    CSI_SA_CALIBRATING,
    CSI_SA_RUNNING,
} csi_sa_state_t;

typedef struct {
    csi_sa_state_t state;
    int   calib_buffered;      // frames collected so far, while calibrating
    int   calib_needed;        // CSI_CALIB_FRAMES
    float calib_remaining_s;   // seconds left of calibration
    int   confirmed;           // 0 / 2, or -1 before a majority is reached
    float confidence;          // vote fraction behind `confirmed`
    int   raw_prediction;      // this window alone, unsmoothed
    float raw_confidence;      // the forest's own probability for it
    float rssi;
    float energy;              // latest frame-to-frame amplitude change
    float noise_relative;      // scale-free empty-room noise floor
    csi_noise_level_t noise_level;
    uint32_t recal_count;
    int64_t last_recal_us;
    bool  subcarrier_mismatch; // stream width != what the model expects
    int   n_subcarriers;
} csi_sa_status_t;

// Call once before frames start arriving.
void csi_standalone_init(void);

// Feed one CSI frame. `amps` holds n_sub amplitudes. Safe to call from the
// printer task; does nothing until csi_standalone_init() has run.
void csi_standalone_on_frame(const float *amps, int n_sub, float rssi);

// Restart calibration ("leave the room" again). Used by the recalibrate
// button once the web server exists.
void csi_standalone_request_recalibration(void);

// Snapshot of the current state, for logging and for the web server.
void csi_standalone_get_status(csi_sa_status_t *out);

// ---- for the web server ----------------------------------------------------
// The dashboard's waterfall wants only the ACTIVE subcarriers: ~20 of the 128
// are structurally dead guard bands that would draw as a permanent black
// stripe. Which ones are active is decided once, from the first frame.
int  csi_standalone_active_count(void);

// Copy the latest frame's active-subcarrier amplitudes into `out` (which must
// hold csi_standalone_active_count() floats) and return the frame sequence
// number. The web task polls this rather than being called from the printer
// task, so a slow or blocked socket can never delay serial output or the
// detector itself.
uint32_t csi_standalone_get_row(float *out, int max);
