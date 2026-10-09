/*
 * main.c — MicroCortex recipe 000: a tiny LLM generating on a $5 chip.
 *
 * Boots the `johnny` runtime from the "model" partition, generates from a
 * fixed prompt, streams tokens to the serial console and (optionally) to a
 * 128x64 SSD1306 OLED with a live tokens/second counter.
 *
 * MIT — 2026 The MicroCortex Authors
 */
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "johnny.h"
#if CONFIG_JOHNNY_USE_OLED
#include "oled_ssd1306.h"
#define OLED_ON 1
#else
#define OLED_ON 0
#endif

static jny_model_t s_model;
static float s_last_tps = 0.0f;

#if OLED_ON
static char s_roll[OLED_COLS + 1];
static int s_roll_len = 0;

static void oled_push_char(char c)
{
    if (c == '\n' || s_roll_len >= OLED_COLS) {
        s_roll[s_roll_len] = 0;
        oled_text(4, s_roll);
        s_roll_len = 0;
    }
    if (c != '\n') s_roll[s_roll_len++] = c;
    s_roll[s_roll_len] = 0;
    oled_text(4, s_roll);
}
#endif

static void on_token(char c, float tps, void *user)
{
    (void)user;
    s_last_tps = tps;
    printf("%c", c);
    fflush(stdout);
#if OLED_ON
    oled_push_char(c);
    static int counter = 0;
    if (++counter % 4 == 0) {
        char line[OLED_COLS + 1];
        snprintf(line, sizeof(line), "%.1f tok/s", (double)tps);
        oled_text(6, line);
        oled_flush();
    }
#else
    (void)tps;
#endif
}

void app_main(void)
{
    printf("\n=== MicroCortex :: johnny runtime (recipe 000) ===\n");

#if OLED_ON
    if (oled_init(CONFIG_JOHNNY_OLED_SDA, CONFIG_JOHNNY_OLED_SCL,
                  CONFIG_JOHNNY_OLED_ADDR) != 0) {
        printf("[oled] not found — continuing on serial only\n");
    } else {
        oled_text(0, "MicroCortex");
        oled_text(1, "johnny loading...");
        oled_flush();
    }
#endif

    const int rc = jny_load_from_partition(CONFIG_JOHNNY_MODEL_PARTITION, &s_model);
    if (rc != 0) {
        printf("[johnny] load failed rc=%d — flash the model first:\n", rc);
        printf("  esptool.py --chip esp32s3 --port PORT write_flash 0x190000 model_Q8.jny\n");
#if OLED_ON
        if (rc == JNY_ERR_PART || rc == JNY_ERR_MAGIC) {
            oled_text(2, "model missing!");
            oled_text(3, "flash model.bin");
            oled_flush();
        }
#endif
        for (;;) vTaskDelay(portMAX_DELAY);
    }

    printf("[johnny] dim=%d layers=%d heads=%d vocab=%d seq=%d quant=%s\n",
           (int)s_model.hdr.dim, (int)s_model.hdr.n_layers, (int)s_model.hdr.n_heads,
           (int)s_model.hdr.vocab_size, (int)s_model.hdr.seq_len,
           s_model.hdr.quant == JNY_QUANT_Q8 ? "q8" : "fp32");
    printf("[johnny] prompt: \"%s\"\n", CONFIG_JOHNNY_PROMPT);
    printf("[johnny] generating %d tokens:\n", CONFIG_JOHNNY_MAX_NEW_TOKENS);
    printf("----------------------------------------------------------------\n");

#if OLED_ON
    oled_text(0, "johnny: generating");
    oled_flush();
    s_roll_len = 0;
#endif

    /* continuous rounds: whoever opens the serial monitor (or resets the chip)
     * always catches the next generation within a few seconds */
    for (int round = 1;; round++) {
        printf("\n=== round %d | prompt: \"%s\" ===\n", round, CONFIG_JOHNNY_PROMPT);
        const int n = jny_generate(&s_model, CONFIG_JOHNNY_PROMPT,
                                   CONFIG_JOHNNY_MAX_NEW_TOKENS,
                                   CONFIG_JOHNNY_TEMPERATURE / 100.0f, on_token, NULL);
        printf("\n[johnny] round %d done: %d tokens at %.2f tok/s (EMA). blob: %d KB\n",
               round, n, (double)s_last_tps, (int)(s_model.blob_size / 1024));
        fflush(stdout);
#if OLED_ON
        oled_text(2, "done.");
        oled_flush();
#endif
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
