/*
 * oled_ssd1306.h — minimal SSD1306 128x64 text display over I2C (ESP-IDF 5.x)
 * MIT — 2026 Emmanuel Vazquez (@Vazquezelectronics) — MicroCortex
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OLED_W 128
#define OLED_H 64
#define OLED_COLS (OLED_W / 6) /* 6 px per glyph (5 px + 1 space) */
#define OLED_ROWS (OLED_H / 8) /* 8 text lines */

int oled_init(int sda_gpio, int scl_gpio, uint8_t i2c_addr);
void oled_clear(void);
void oled_text(int row, const char *text); /* row 0..7, clipped to OLED_COLS */
void oled_flush(void);

#ifdef __cplusplus
}
#endif
