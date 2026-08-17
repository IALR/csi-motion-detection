// Port of csi_common.py. See csi_features.h for the correspondences that must
// hold. csi_selftest.c proves this against real recorded windows.

#include "csi_features.h"

#include <math.h>
#include <stdlib.h>   // qsort
#include <string.h>

// NOT REENTRANT, deliberately.
//
// The scratch arrays below are `static` rather than local because as locals
// they needed ~3.4KB of stack for one feature computation (five 128-float
// arrays in csi_calibrate_features plus two 100-float arrays in
// raw_window_stats), against FreeRTOS's 3584-byte default main task stack.
// That overflowed and boot-looped the board on the very first flash - after
// the parity self-test had already passed, so the maths was right and only
// the memory placement was wrong.
//
// Only one task ever runs inference, so sharing them is safe. If a second
// task is ever added that calls into this file, these must become per-task
// buffers or be guarded by a mutex.

// ---------------------------------------------------------------- helpers --

static float mean_f(const float *v, int n)
{
    // Straight sequential sum. numpy uses pairwise summation for large arrays,
    // which is more accurate, but at these sizes (8 frames, 128 subcarriers)
    // the two agree to well within float32 resolution - and the self-test
    // checks that empirically rather than assuming it.
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i];
    return s / (float)n;
}

// Population standard deviation (ddof=0), matching numpy's default.
static float stddev_f(const float *v, int n, float mean)
{
    if (n <= 0) return 0.0f;
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = v[i] - mean;
        acc += d * d;
    }
    return sqrtf(acc / (float)n);
}

static int cmp_desc(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    if (x < y) return 1;
    if (x > y) return -1;
    return 0;
}

// Linear-interpolated percentile, matching numpy.percentile's default method
// on an ASCENDING array. Anything simpler (nearest-rank) shifts std_ratio_p90
// enough to change predictions.
static float percentile_asc(const float *sorted_asc, int n, float pct)
{
    if (n <= 0) return 0.0f;
    if (n == 1) return sorted_asc[0];
    float pos = (pct / 100.0f) * (float)(n - 1);
    int lo = (int)floorf(pos);
    int hi = lo + 1;
    if (hi >= n) return sorted_asc[n - 1];
    float frac = pos - (float)lo;
    return sorted_asc[lo] + (sorted_asc[hi] - sorted_asc[lo]) * frac;
}

static float median_f(const float *v, int n)
{
    static float tmp[CSI_NS];
    memcpy(tmp, v, sizeof(float) * (size_t)n);
    // ascending sort via the descending comparator, then read backwards
    qsort(tmp, (size_t)n, sizeof(float), cmp_desc);
    // tmp is descending; median of a descending array is the same element(s)
    if (n % 2) return tmp[n / 2];
    return 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
}

// ------------------------------------------------------------ window stats --

void csi_raw_window_stats(const float *amps, const float *rssi, int n_frames,
                          csi_stats_t *out)
{
    // Sized for the CALIBRATION block, not just a scoring window - this same
    // function computes both, and a baseline is 100 frames against a window's 8.
    static float col[CSI_MAX_FRAMES];

    for (int sc = 0; sc < CSI_NS; sc++) {
        for (int f = 0; f < n_frames; f++) col[f] = amps[f * CSI_NS + sc];
        float m = mean_f(col, n_frames);
        out->amp_mean[sc] = m;
        out->amp_std[sc] = stddev_f(col, n_frames, m);
    }

    // Motion energy: per frame-pair, the mean over subcarriers of the absolute
    // change; then mean/std over the window's n-1 pairs. With a single frame
    // Python yields array([0.0]), i.e. mean 0 and std 0.
    if (n_frames > 1) {
        static float diffs[CSI_MAX_FRAMES];  // n_frames-1 used; sized for calibration
        for (int f = 1; f < n_frames; f++) {
            float acc = 0.0f;
            for (int sc = 0; sc < CSI_NS; sc++) {
                acc += fabsf(amps[f * CSI_NS + sc] - amps[(f - 1) * CSI_NS + sc]);
            }
            diffs[f - 1] = acc / (float)CSI_NS;
        }
        float dm = mean_f(diffs, n_frames - 1);
        out->energy_mean = dm;
        out->energy_std = stddev_f(diffs, n_frames - 1, dm);
    } else {
        out->energy_mean = 0.0f;
        out->energy_std = 0.0f;
    }

    float rm = mean_f(rssi, n_frames);
    out->rssi_mean = rm;
    out->rssi_std = stddev_f(rssi, n_frames, rm);
}

void csi_compute_baseline(const float *amps, const float *rssi, int n_frames,
                          csi_baseline_t *out)
{
    csi_stats_t s;
    csi_raw_window_stats(amps, rssi, n_frames, &s);

    memcpy(out->amp_mean, s.amp_mean, sizeof(s.amp_mean));
    memcpy(out->amp_std, s.amp_std, sizeof(s.amp_std));
    out->energy_mean = s.energy_mean;
    out->energy_std = s.energy_std;
    out->rssi_mean = s.rssi_mean;
    out->rssi_std = s.rssi_std;

    // Guard-band subcarriers sit at exactly 0, so their baseline std is 0 and
    // any ratio against them explodes. Floor at 10% of the median std.
    float med = median_f(s.amp_std, CSI_NS);
    float f1 = med * 0.1f;
    out->amp_std_floor = f1 > CSI_EPS ? f1 : CSI_EPS;
    float f2 = s.rssi_std * 0.1f;
    out->rssi_std_floor = f2 > CSI_EPS ? f2 : CSI_EPS;
    out->valid = true;
}

// --------------------------------------------------------------- features --

void csi_calibrate_features(const csi_stats_t *s, const csi_baseline_t *b,
                            float *fv)
{
    static float std_ratio[CSI_NS];
    static float abs_delta[CSI_NS];

    for (int i = 0; i < CSI_NS; i++) {
        float delta = s->amp_mean[i] - b->amp_mean[i];
        fv[i] = delta;
        abs_delta[i] = fabsf(delta);

        float den = b->amp_std[i] > b->amp_std_floor ? b->amp_std[i] : b->amp_std_floor;
        float r = s->amp_std[i] / den;
        fv[CSI_NS + i] = r;
        std_ratio[i] = r;
    }

    float e_den = b->energy_mean > CSI_ENERGY_STD_FLOOR ? b->energy_mean : CSI_ENERGY_STD_FLOOR;
    float es_den = b->energy_std > CSI_ENERGY_STD_FLOOR ? b->energy_std : CSI_ENERGY_STD_FLOOR;
    float r_den = b->rssi_std > b->rssi_std_floor ? b->rssi_std : b->rssi_std_floor;

    int k = 2 * CSI_NS;
    fv[k + 0] = s->energy_mean / e_den;
    fv[k + 1] = s->energy_std / es_den;
    fv[k + 2] = s->rssi_mean - b->rssi_mean;
    fv[k + 3] = s->rssi_std / r_den;

    // Order-invariant summaries: "is SOMETHING in the band disturbed", without
    // caring which index - these are what let the model transfer between rooms.
    static float sr_desc[CSI_NS], ad_desc[CSI_NS];
    memcpy(sr_desc, std_ratio, sizeof(sr_desc));
    memcpy(ad_desc, abs_delta, sizeof(ad_desc));
    qsort(sr_desc, CSI_NS, sizeof(float), cmp_desc);
    qsort(ad_desc, CSI_NS, sizeof(float), cmp_desc);

    int topk = CSI_TOP_K < CSI_NS ? CSI_TOP_K : CSI_NS;
    float sr_top = 0.0f, ad_top = 0.0f;
    for (int i = 0; i < topk; i++) { sr_top += sr_desc[i]; ad_top += ad_desc[i]; }
    sr_top /= (float)topk;
    ad_top /= (float)topk;

    int elevated = 0;
    for (int i = 0; i < CSI_NS; i++) if (std_ratio[i] > CSI_ELEVATED_RATIO) elevated++;

    // percentile_asc expects ascending; sr_desc is descending, so mirror it.
    static float sr_asc[CSI_NS];
    for (int i = 0; i < CSI_NS; i++) sr_asc[i] = sr_desc[CSI_NS - 1 - i];

    fv[k + 4] = sr_desc[0];                                   // std_ratio_max
    fv[k + 5] = sr_top;                                       // std_ratio_topK_mean
    fv[k + 6] = percentile_asc(sr_asc, CSI_NS, 90.0f);        // std_ratio_p90
    fv[k + 7] = (float)elevated / (float)CSI_NS;              // std_ratio_frac_elevated
    fv[k + 8] = ad_desc[0];                                   // mean_delta_absmax
    fv[k + 9] = ad_top;                                       // mean_delta_absmax_topK_mean
}

void csi_make_features(const float *amps, const float *rssi, int n_frames,
                       const csi_baseline_t *b, float *features)
{
    csi_stats_t s;
    csi_raw_window_stats(amps, rssi, n_frames, &s);
    csi_calibrate_features(&s, b, features);
}

// -------------------------------------------------------------- inference --

int csi_predict(const float *fv, float *p1_out)
{
    // sklearn averages each tree's PROBABILITY, then argmax - it does not take
    // a hard majority vote. Leaves therefore store p(class 1), not a label.
    float acc = 0.0f;
    for (int t = 0; t < CSI_N_TREES; t++) {
        uint32_t base = csi_tree_offset[t];
        int node = 0;
        while (csi_feature[base + node] != CSI_LEAF) {
            int f = csi_feature[base + node];
            node = (fv[f] <= csi_threshold[base + node]) ? csi_left[base + node]
                                                         : csi_right[base + node];
        }
        acc += csi_threshold[base + node];
    }
    float p1 = acc / (float)CSI_N_TREES;
    if (p1_out) *p1_out = p1;
    // argmax returns the FIRST maximum, so an exact 0.5 tie is class 0.
    return (p1 > 0.5f) ? csi_classes[1] : csi_classes[0];
}

// -------------------------------------------------------------- smoothing --

void csi_smoother_reset(csi_smoother_t *s)
{
    memset(s, 0, sizeof(*s));
    s->confirmed = -1;
}

int csi_smoother_update(csi_smoother_t *s, int raw_label)
{
    s->history[s->head] = (int8_t)raw_label;
    s->head = (s->head + 1) % CSI_SMOOTH_SIZE;
    if (s->count < CSI_SMOOTH_SIZE) s->count++;

    // Only two labels exist, so counting one of them is enough.
    int n0 = 0;
    for (int i = 0; i < s->count; i++) if (s->history[i] == csi_classes[0]) n0++;
    int n1 = s->count - n0;

    int majority_label = (n0 >= n1) ? csi_classes[0] : csi_classes[1];
    int majority_count = (n0 >= n1) ? n0 : n1;
    s->vote_fraction = (float)majority_count / (float)s->count;

    // Strict majority required, matching `majority_count > len/2`.
    if (majority_count * 2 > s->count) s->confirmed = majority_label;
    return s->confirmed;
}

// -------------------------------------------------- rolling recalibration --

void csi_roller_reset(csi_roller_t *r)
{
    r->count = 0;
    r->have_floor = false;
}

void csi_roller_set_floor(csi_roller_t *r, const csi_baseline_t *startup)
{
    r->floor_ref = *startup;
    r->have_floor = true;
}

bool csi_roller_observe(csi_roller_t *r, int confirmed_label,
                        const float *amps, float rssi, csi_baseline_t *baseline)
{
    // Any non-empty label discards the buffer, so a burst of motion can never
    // contaminate the baseline.
    if (confirmed_label != csi_classes[0]) {
        r->count = 0;
        return false;
    }
    if (r->count < CSI_CALIB_FRAMES) {
        memcpy(r->amp_buf[r->count], amps, sizeof(float) * CSI_NS);
        r->rssi_buf[r->count] = rssi;
        r->count++;
    }
    if (r->count < CSI_CALIB_FRAMES) return false;

    csi_baseline_t fresh;
    csi_compute_baseline((const float *)r->amp_buf, r->rssi_buf,
                         CSI_CALIB_FRAMES, &fresh);
    r->count = 0;

    const float a = CSI_BLEND_ALPHA;
    for (int i = 0; i < CSI_NS; i++) {
        baseline->amp_mean[i] = (1.0f - a) * baseline->amp_mean[i] + a * fresh.amp_mean[i];
        baseline->amp_std[i]  = (1.0f - a) * baseline->amp_std[i]  + a * fresh.amp_std[i];
    }
    baseline->energy_mean = (1.0f - a) * baseline->energy_mean + a * fresh.energy_mean;
    baseline->energy_std  = (1.0f - a) * baseline->energy_std  + a * fresh.energy_std;
    baseline->rssi_mean   = (1.0f - a) * baseline->rssi_mean   + a * fresh.rssi_mean;
    baseline->rssi_std    = (1.0f - a) * baseline->rssi_std    + a * fresh.rssi_std;

    // Never let blending make the system MORE sensitive than the deliberate
    // startup calibration ever was - that ratchet is what used to drift the
    // live system into reporting motion in an empty room.
    if (r->have_floor) {
        for (int i = 0; i < CSI_NS; i++) {
            float lo = r->floor_ref.amp_std[i] * CSI_MIN_STD_FRACTION;
            if (baseline->amp_std[i] < lo) baseline->amp_std[i] = lo;
        }
        float lo_r = r->floor_ref.rssi_std * CSI_MIN_STD_FRACTION;
        if (baseline->rssi_std < lo_r) baseline->rssi_std = lo_r;
        float lo_e = r->floor_ref.energy_std * CSI_MIN_STD_FRACTION;
        if (baseline->energy_std < lo_e) baseline->energy_std = lo_e;
    }

    float med = median_f(baseline->amp_std, CSI_NS);
    float f1 = med * 0.1f;
    baseline->amp_std_floor = f1 > CSI_EPS ? f1 : CSI_EPS;
    float f2 = baseline->rssi_std * 0.1f;
    baseline->rssi_std_floor = f2 > CSI_EPS ? f2 : CSI_EPS;
    return true;
}

// ------------------------------------------------------------ noise floor --

csi_noise_level_t csi_assess_noise(const csi_baseline_t *b, float *relative_out)
{
    // Average amplitude over ACTIVE subcarriers only: the ~20 dead guard-band
    // ones sit at exactly zero and would deflate the scale, inflating the ratio.
    float sum = 0.0f;
    int active = 0;
    for (int i = 0; i < CSI_NS; i++) {
        if (b->amp_mean[i] > 0.0f) { sum += b->amp_mean[i]; active++; }
    }
    float scale = active ? (sum / (float)active) : 0.0f;
    float rel = scale > 0.0f ? (b->energy_mean / scale) : 0.0f;
    if (relative_out) *relative_out = rel;

    if (rel <= CSI_NOISE_REL_QUIET) return CSI_NOISE_QUIET;
    if (rel < CSI_NOISE_REL_LOUD) return CSI_NOISE_MODERATE;
    return CSI_NOISE_LOUD;
}
