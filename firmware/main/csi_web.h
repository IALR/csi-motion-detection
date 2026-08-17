// HTTP + WebSocket server serving the project's own dashboard from the board.
//
// Serves csi_dashboard.html verbatim (embedded into flash) and speaks the same
// WebSocket protocol csi_live_server.py speaks, so the identical page works
// whether the laptop or the ESP32 is feeding it.
//
// Call after Wi-Fi has an IP.
#pragma once

#include "esp_err.h"

esp_err_t csi_web_start(void);
