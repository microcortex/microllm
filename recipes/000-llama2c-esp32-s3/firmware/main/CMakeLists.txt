idf_component_register(SRCS "main.c" "johnny.c" "oled_ssd1306.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp_driver_i2c esp_timer)
