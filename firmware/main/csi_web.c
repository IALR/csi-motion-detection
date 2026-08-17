// HTTP + WebSocket server: serves the project's own dashboard from the board.
//
// The page is csi_dashboard.html verbatim - the same file the PC setup uses,
// embedded into flash - and this speaks the SAME WebSocket message protocol
// csi_live_server.py speaks. That was the whole reason not to write a second
// dashboard: the protocol was the hard design work and it already exists.
//
// The page finds its server from where it was loaded (see WS_URL in the
// dashboard), so served over http:// it connects back here, while opening the
// same file from disk still reaches the Python server.

#include "csi_web.h"

#include <stdio.h>
#include <string.h>

#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "csi_standalone.h"

static const char *TAG = "csi_web";

// The dashboard, embedded at build time (see main/CMakeLists.txt EMBED_FILES).
extern const uint8_t dashboard_start[] asm("_binary_csi_dashboard_html_start");
extern const uint8_t dashboard_end[]   asm("_binary_csi_dashboard_html_end");

#define MAX_WS_CLIENTS 4
#define NODE_ID "A"

static httpd_handle_t s_server = NULL;
static int  s_clients[MAX_WS_CLIENTS];
static int  s_client_count = 0;

// One shared buffer: a frame message carries ~108 numbers, and building it on
// a task stack would be another needless kilobyte or two of stack pressure.
// Only the web task writes it.
static char s_json[3072];
static float s_row[CSI_NS];

static void client_add(int fd)
{
    for (int i = 0; i < s_client_count; i++) if (s_clients[i] == fd) return;
    if (s_client_count < MAX_WS_CLIENTS) {
        s_clients[s_client_count++] = fd;
        ESP_LOGI(TAG, "client connected (fd %d, %d total)", fd, s_client_count);
    } else {
        ESP_LOGW(TAG, "client limit (%d) reached, refusing fd %d", MAX_WS_CLIENTS, fd);
    }
}

static void client_remove(int fd)
{
    for (int i = 0; i < s_client_count; i++) {
        if (s_clients[i] == fd) {
            s_clients[i] = s_clients[--s_client_count];
            ESP_LOGI(TAG, "client gone (fd %d, %d left)", fd, s_client_count);
            return;
        }
    }
}

static void ws_broadcast(const char *json)
{
    if (!s_server || s_client_count == 0) return;
    httpd_ws_frame_t frame = {
        .final = true,
        .fragmented = false,
        .type = HTTPD_WS_TYPE_TEXT,
        .payload = (uint8_t *)json,
        .len = strlen(json),
    };
    // Iterate backwards: a failed send removes that client, which swaps the
    // last entry into its slot, and a forward loop would skip the moved one.
    for (int i = s_client_count - 1; i >= 0; i--) {
        int fd = s_clients[i];
        if (httpd_ws_send_frame_async(s_server, fd, &frame) != ESP_OK) {
            client_remove(fd);
        }
    }
}

// ---------------------------------------------------------------- handlers --

static esp_err_t page_get_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, (const char *)dashboard_start,
                           dashboard_end - dashboard_start - 1);
}

static esp_err_t ws_handler(httpd_req_t *req)
{
    if (req->method == HTTP_GET) {
        // Handshake. Nothing to send yet - the periodic push loop will bring
        // this client up to date on its next tick.
        client_add(httpd_req_to_sockfd(req));
        return ESP_OK;
    }

    httpd_ws_frame_t frame = { .type = HTTPD_WS_TYPE_TEXT };
    uint8_t buf[256] = {0};
    frame.payload = buf;
    esp_err_t ret = httpd_ws_recv_frame(req, &frame, sizeof(buf) - 1);
    if (ret != ESP_OK) return ret;

    if (frame.type == HTTPD_WS_TYPE_CLOSE) {
        client_remove(httpd_req_to_sockfd(req));
        return ESP_OK;
    }
    if (frame.type != HTTPD_WS_TYPE_TEXT) return ESP_OK;

    // Only one command exists, and a substring match is enough rather than
    // pulling in a JSON parser for it: {"type":"recalibrate","node":"A"}
    if (strstr((char *)buf, "\"recalibrate\"")) {
        ESP_LOGI(TAG, "recalibration requested by a client");
        csi_standalone_request_recalibration();
    }
    return ESP_OK;
}

// ------------------------------------------------------------ push messages --

static const char *level_name(csi_noise_level_t l)
{
    switch (l) {
    case CSI_NOISE_QUIET:    return "quiet";
    case CSI_NOISE_MODERATE: return "moderate";
    default:                 return "loud";
    }
}

static void send_init(void)
{
    int active = csi_standalone_active_count();
    if (active <= 0) return;
    snprintf(s_json, sizeof(s_json),
             "{\"type\":\"init\",\"node\":\"" NODE_ID "\","
             "\"n_subcarriers\":%d,\"n_active\":%d}", CSI_NS, active);
    ws_broadcast(s_json);
}

static void send_calibrating(const csi_sa_status_t *st)
{
    snprintf(s_json, sizeof(s_json),
             "{\"type\":\"calibrating\",\"node\":\"" NODE_ID "\","
             "\"buffered\":%d,\"calib_frames\":%d,\"remaining_seconds\":%.1f}",
             st->calib_buffered, st->calib_needed, st->calib_remaining_s);
    ws_broadcast(s_json);
}

static void send_calibrated(const csi_sa_status_t *st, bool rolling)
{
    // The dashboard grades and colours the noise tile from these fields, and
    // shows the detail text on hover, so the wording matches csi_common.py's
    // assess_noise_floor() - a loud room is a risk regime, not a failure.
    const char *detail =
        st->noise_level == CSI_NOISE_LOUD
            ? "An empty room here is nearly as disturbed as an occupied one, so the "
              "aggregate motion signal is weak. Sessions in this regime scored 85-95% "
              "held out, versus 95-99% in quiet rooms - results get more variable, "
              "with false MOVING readings the likelier error. It does not mean "
              "detection will fail."
        : st->noise_level == CSI_NOISE_MODERATE
            ? "Above the quiet sessions but the classes still separate well."
            : "Matches this project's most reliable recording sessions.";

    snprintf(s_json, sizeof(s_json),
             "{\"type\":\"%s\",\"node\":\"" NODE_ID "\",\"at_unix_ms\":%lld,"
             "\"noise_floor\":%.4f,\"noise_level\":\"%s\","
             "\"noise_headline\":\"%s environment (noise %.3f)\","
             "\"noise_detail\":\"%s\"}",
             rolling ? "recalibrated" : "calibrated",
             (long long)(esp_timer_get_time() / 1000),
             st->noise_relative, level_name(st->noise_level),
             level_name(st->noise_level), st->noise_relative, detail);
    ws_broadcast(s_json);
}

static void send_frame(const csi_sa_status_t *st, int active)
{
    int off = snprintf(s_json, sizeof(s_json),
                       "{\"type\":\"frame\",\"node\":\"" NODE_ID "\",\"row\":[");
    for (int i = 0; i < active && off < (int)sizeof(s_json) - 64; i++) {
        off += snprintf(s_json + off, sizeof(s_json) - off, "%s%.0f",
                        i ? "," : "", s_row[i]);
    }
    const char *pred = (st->confirmed < 0) ? "null"
                     : (st->confirmed == 2 ? "\"MOVING\"" : "\"EMPTY\"");
    off += snprintf(s_json + off, sizeof(s_json) - off,
                    "],\"rssi\":%.0f,\"energy\":%.3f,"
                    "\"prediction\":%s,\"confidence\":%.3f,"
                    "\"raw_prediction\":\"%s\",\"raw_confidence\":%.3f,"
                    "\"buffered\":%d,\"window_frames\":%d}",
                    st->rssi, st->energy, pred, st->confidence,
                    st->raw_prediction == 2 ? "MOVING" : "EMPTY",
                    st->raw_confidence, CSI_WINDOW_FRAMES, CSI_WINDOW_FRAMES);
    ws_broadcast(s_json);
}

static void send_combined(const csi_sa_status_t *st)
{
    // Single node for now. The dashboard's hero badge is driven by this
    // message, so it has to be sent even with one node; ESP-NOW will later
    // fold node B's state in here rather than changing the page.
    const char *node = (st->confirmed < 0) ? "null"
                     : (st->confirmed == 2 ? "\"MOVING\"" : "\"EMPTY\"");
    snprintf(s_json, sizeof(s_json),
             "{\"type\":\"combined\",\"prediction\":%s,"
             "\"nodes\":{\"" NODE_ID "\":%s},\"muted\":[]}", node, node);
    ws_broadcast(s_json);
}

// The page is only fed while someone is watching; with no clients this loop
// costs a poll every 100ms and nothing else.
static void web_push_task(void *arg)
{
    csi_sa_state_t last_state = (csi_sa_state_t)-1;
    int last_confirmed = -2;
    uint32_t last_seq = 0, last_recal = 0;
    bool init_sent = false;
    int active = 0;

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(100));
        if (s_client_count == 0) { init_sent = false; last_state = (csi_sa_state_t)-1; continue; }

        csi_sa_status_t st;
        csi_standalone_get_status(&st);
        active = csi_standalone_active_count();

        if (!init_sent && active > 0) {
            send_init();
            init_sent = true;
            last_state = (csi_sa_state_t)-1;   // force a state message next
        }
        if (!init_sent) continue;

        if (st.state == CSI_SA_CALIBRATING) {
            send_calibrating(&st);
            last_state = st.state;
            continue;
        }
        if (st.state == CSI_SA_RUNNING && last_state != CSI_SA_RUNNING) {
            send_calibrated(&st, false);
            last_state = st.state;
            last_recal = st.recal_count;
        }
        if (st.recal_count != last_recal) {
            last_recal = st.recal_count;
            send_calibrated(&st, true);
        }
        if (st.state != CSI_SA_RUNNING) { last_state = st.state; continue; }

        uint32_t seq = csi_standalone_get_row(s_row, active);
        if (seq != last_seq) {
            last_seq = seq;
            send_frame(&st, active);
        }
        if (st.confirmed != last_confirmed) {
            last_confirmed = st.confirmed;
            send_combined(&st);
        }
    }
}

// ------------------------------------------------------------------- start --

esp_err_t csi_web_start(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.lru_purge_enable = true;
    config.max_open_sockets = MAX_WS_CLIENTS + 2;   // + listener and headroom
    config.stack_size = 8192;

    esp_err_t err = httpd_start(&s_server, &config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "httpd_start failed: %s", esp_err_to_name(err));
        return err;
    }

    static const httpd_uri_t page = {
        .uri = "/", .method = HTTP_GET, .handler = page_get_handler,
    };
    static const httpd_uri_t ws = {
        .uri = "/ws", .method = HTTP_GET, .handler = ws_handler,
        .is_websocket = true,
    };
    httpd_register_uri_handler(s_server, &page);
    httpd_register_uri_handler(s_server, &ws);

    xTaskCreate(web_push_task, "csi_web_push", 4096, NULL, 3, NULL);
    ESP_LOGW(TAG, "dashboard served on port 80 (%d bytes) - open the board's IP "
                  "in a browser", (int)(dashboard_end - dashboard_start - 1));
    return ESP_OK;
}
