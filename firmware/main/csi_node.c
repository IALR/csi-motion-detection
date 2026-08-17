#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/uart.h"
#include "ping/ping_sock.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

// ---- Wi-Fi credentials ----
// Real values live in wifi_secrets.h, which is gitignored and never
// committed. Copy wifi_secrets.h.example to wifi_secrets.h and fill in
// your own network's SSID/password/router IP before building.
#include "wifi_secrets.h"
#include "csi_selftest.h"
#include "csi_standalone.h"
#include "csi_web.h"

// ---- CSI plumbing ----
#define MAX_CSI_BYTES   512   // covers HT40 (384 bytes) with margin
#define CSI_QUEUE_LEN   32

static const char *TAG = "csi_node";
static volatile int current_label = 0;  // 0=empty, 1=still, 2=moving

static QueueHandle_t csi_queue = NULL;
static volatile uint32_t frames_dropped = 0;  // queue-full drops

// AP MAC, captured on connect, used to keep only CSI from our AP
static uint8_t s_ap_bssid[6];
static volatile bool s_have_bssid = false;

typedef struct {
    int64_t ts_us;
    int     label;
    int8_t  rssi;
    int     len;
    int8_t  buf[MAX_CSI_BYTES];
} csi_item_t;

// ---------------- Wi-Fi ----------------
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_have_bssid = false;
        ESP_LOGI(TAG, "Disconnected, retrying...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGW(TAG, ">>> open http://" IPSTR "/ in a browser for the dashboard <<<",
                 IP2STR(&event->ip_info.ip));
        // Started here rather than in app_main because the HTTP server needs a
        // live network interface. Guarded so a Wi-Fi reconnect does not try to
        // start a second one.
        static bool web_started = false;
        if (!web_started && csi_web_start() == ESP_OK) {
            web_started = true;
        }

        // Capture the AP's BSSID so we can filter CSI to just our AP.
        wifi_ap_record_t ap;
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            memcpy(s_ap_bssid, ap.bssid, 6);
            s_have_bssid = true;
            ESP_LOGI(TAG, "AP BSSID %02x:%02x:%02x:%02x:%02x:%02x, ch %d",
                     ap.bssid[0], ap.bssid[1], ap.bssid[2],
                     ap.bssid[3], ap.bssid[4], ap.bssid[5], ap.primary);
        }
    }
}

// ---------------- Ping (triggers CSI; do NOT log every reply) ----------------
static void start_ping(void)
{
    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ping_config.count = 0;               // run forever
    ping_config.interval_ms = 100;       // ~10 Hz CSI trigger
    ping_config.task_stack_size = 4096;
    ipaddr_aton(ROUTER_IP, &ping_config.target_addr);

    // No per-reply callback: logging every ping floods the UART and
    // competes with the CSI stream for bandwidth.
    esp_ping_callbacks_t cbs = {
        .cb_args = NULL,
        .on_ping_success = NULL,
        .on_ping_timeout = NULL,
        .on_ping_end = NULL,
    };

    esp_ping_handle_t ping;
    esp_ping_new_session(&ping_config, &cbs, &ping);
    esp_ping_start(ping);
    ESP_LOGI(TAG, "Ping started (%d ms interval)", ping_config.interval_ms);
}

// ---------------- CSI callback: copy and hand off, nothing heavy ----------------
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || info->len <= 0 || info->len > MAX_CSI_BYTES) {
        return;
    }
    // Keep only CSI from our AP (ping replies + beacons), drop everything else.
    if (s_have_bssid && memcmp(info->mac, s_ap_bssid, 6) != 0) {
        return;
    }

    csi_item_t item;
    item.ts_us = esp_timer_get_time();
    item.label = current_label;
    item.rssi  = info->rx_ctrl.rssi;
    item.len   = info->len;
    memcpy(item.buf, info->buf, info->len);

    // Non-blocking: if the printer can't keep up, drop and count it.
    // This runs in the Wi-Fi task, so it must return fast.
    if (xQueueSend(csi_queue, &item, 0) != pdTRUE) {
        frames_dropped++;
    }
}

// ---------------- Printer task: formatting + UART happen here ----------------
static void csi_print_task(void *arg)
{
    static char line[2048];
    csi_item_t item;
    uint32_t last_report = 0;

    while (1) {
        if (xQueueReceive(csi_queue, &item, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        int n = item.len / 2;
        int off = 0;
        // Format: CSI_AMP,<ts_us>,<label>,<num_subcarriers>,<rssi>,<amp0>,<amp1>,...
        // Run the Python collector with --rssi-field to store the rssi column.
        off += snprintf(line + off, sizeof(line) - off,
                        "CSI_AMP,%lld,%d,%d,%d",
                        (long long)item.ts_us, item.label, n, (int)item.rssi);

        // Amplitudes are kept as floats alongside the printed integers so the
        // on-device detector scores exactly the values that go out over serial
        // - the SAME numbers the PC pipeline receives, not a separately
        // computed variant that could drift from them.
        static float amp_f[MAX_CSI_BYTES / 2];
        int kept = 0;
        for (int i = 0; i < n; i++) {
            int8_t imag = item.buf[i * 2];
            int8_t real = item.buf[i * 2 + 1];
            int amp = (int)lroundf(sqrtf((float)(imag * imag + real * real)));
            int w = snprintf(line + off, sizeof(line) - off, ",%d", amp);
            if (w < 0 || off + w >= (int)sizeof(line)) {
                break;
            }
            off += w;
            if (kept < (int)(sizeof(amp_f) / sizeof(amp_f[0]))) {
                amp_f[kept++] = (float)amp;
            }
        }

        // Raw printf (no ESP_LOG prefix/color) = fewer UART bytes per frame.
        // This stays FIRST and unchanged: the serial stream is what
        // csi_live_server.py and the collector consume, and standalone
        // detection must never be able to disturb or delay it.
        printf("%s\n", line);

        // Then score the same frame on-device. Done here rather than in a
        // separate task because it is cheap next to the UART write: ~200 trees
        // x ~20 levels is well under a millisecond, against ~65ms to shift a
        // 750-byte line out at 115200 baud. Only fully-formatted frames are
        // scored, so a truncated line never reaches the model.
        if (kept == n) {
            csi_standalone_on_frame(amp_f, n, (float)item.rssi);
        }

        // Occasionally report dropped-frame count so you can see if the
        // UART is the bottleneck (raise the console baud if drops climb).
        uint32_t now = (uint32_t)(esp_timer_get_time() / 1000000);
        if (frames_dropped && now != last_report && now % 10 == 0) {
            last_report = now;
            ESP_LOGW(TAG, "CSI frames dropped (queue full): %lu",
                     (unsigned long)frames_dropped);
        }
    }
}

static void start_csi(void)
{
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    ESP_LOGI(TAG, "CSI enabled");
}

// ---------------- Label input (only reached when a terminal is attached) ----
// During Python collection the host owns the label; this is a fallback for
// the IDF monitor. Harmless to keep.
static void label_input_task(void *arg)
{
    while (1) {
        int ch = getchar();
        if (ch != EOF) {
            if (ch == 'e' || ch == 'E') {
                current_label = 0;
                ESP_LOGI(TAG, "=== LABEL 0 (EMPTY) ===");
            } else if (ch == 'p' || ch == 'P' || ch == 's' || ch == 'S') {
                current_label = 1;
                ESP_LOGI(TAG, "=== LABEL 1 (STILL) ===");
            } else if (ch == 'm' || ch == 'M') {
                current_label = 2;
                ESP_LOGI(TAG, "=== LABEL 2 (MOVING) ===");
            }
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

// ---------------- Main ----------------
void app_main(void)
{
    // Parity check first, before Wi-Fi brings any noise into the log. This
    // replays real recorded windows through the on-device feature extraction
    // and forest, and compares against what scikit-learn produced for those
    // exact frames on the PC. If it fails, the port is wrong and nothing
    // downstream can be trusted - so it is loud about it rather than
    // continuing quietly into a subtly broken detector.
    if (!csi_selftest_run()) {
        ESP_LOGE(TAG, "Self-test FAILED - on-device inference is not trustworthy. "
                      "Continuing to stream CSI so the PC pipeline still works.");
    }

    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                               &wifi_event_handler, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Force this station's link to 20MHz.
    //
    // The number of CSI subcarriers is set by the channel width: 20MHz gives
    // 128, 40MHz gives 192. The model is built on 128, so a 40MHz link makes
    // every feature vector the wrong shape and detection stops dead. This
    // project has been broken by exactly that three times, always because the
    // access point renegotiated width on its own - a phone hotspot offers no
    // setting to prevent it, and previously the only suggested fix was to buy
    // a different router.
    //
    // Pinning it here is the real fix: the STA advertises 20MHz only, so the
    // AP cannot pull the link up to 40MHz whatever it decides to do. Non-fatal
    // if it fails - the mismatch is still detected and reported downstream.
    esp_err_t bw = esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW20);
    if (bw == ESP_OK) {
        ESP_LOGI(TAG, "link pinned to 20MHz (keeps CSI at 128 subcarriers)");
    } else {
        ESP_LOGW(TAG, "could not pin 20MHz: %s - the AP may still negotiate "
                      "40MHz, which yields 192 subcarriers and disables detection",
                 esp_err_to_name(bw));
    }

    ESP_LOGI(TAG, "WiFi init done, connecting...");

    // Queue + printer task BEFORE enabling CSI so no frame is lost.
    csi_queue = xQueueCreate(CSI_QUEUE_LEN, sizeof(csi_item_t));
    if (!csi_queue) {
        ESP_LOGE(TAG, "Failed to create CSI queue");
        return;
    }
    // 6144 rather than 4096: this task now also runs feature extraction and
    // the forest. The heavy scratch buffers in csi_features.c are static so
    // most of that cost is gone, but the earlier boot-loop was exactly this
    // mistake made on the main task, so leave headroom here too.
    xTaskCreate(csi_print_task, "csi_print", 6144, NULL, 4, NULL);

    vTaskDelay(pdMS_TO_TICKS(5000));
    start_ping();

    // Arm detection BEFORE frames can arrive. csi_standalone_init() zeroes its
    // whole state, and the printer task calls into that same state from
    // another task - initialising after start_csi() left a window where a
    // frame could be mid-flight through it while it was being wiped.
    // It discards the first WARMUP_FRAMES regardless, because packets right
    // after association are unrepresentative (rate adaptation and AGC still
    // settling) and would bias the baseline everything is compared against.
    csi_standalone_init();

    vTaskDelay(pdMS_TO_TICKS(1000));
    start_csi();

    xTaskCreate(label_input_task, "label_input", 4096, NULL, 5, NULL);
}