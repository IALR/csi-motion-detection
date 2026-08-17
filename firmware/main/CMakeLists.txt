# The dashboard is embedded from the repo root rather than copied into the
# firmware tree, so the board serves the SAME file the PC setup uses and the
# two can never drift apart.
#
# NOTE on REQUIRES: the `main` component normally gets an implicit dependency
# on every component, but declaring REQUIRES at all turns that off - so every
# component this file uses has to be listed, not just the newly added one.
# Omitting them fails with a confusing "esp_timer.h: No such file" rather than
# anything mentioning dependencies.
idf_component_register(SRCS "csi_node.c"
                            "csi_features.c"
                            "csi_selftest.c"
                            "csi_standalone.c"
                            "csi_web.c"
                    INCLUDE_DIRS "."
                    EMBED_FILES "../../csi_dashboard.html"
                    REQUIRES esp_wifi
                             esp_event
                             esp_timer
                             esp_netif
                             nvs_flash
                             lwip
                             esp_driver_uart
                             esp_http_server)
