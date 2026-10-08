/*
 * johnny.h — MicroCortex tiny-LLM runtime for ESP32-S3 (recipe 000)
 *
 * Loads a "johnny v1" .jny model (see johnny_format.h) from a flash partition,
 * keeps it in PSRAM, and generates text autoregressively with a KV cache.
 *
 * MIT — 2026 The MicroCortex Authors. Inspired by llama2.c (MIT).
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include "johnny_format.h"

#ifdef __cplusplus
extern "C" {
#endif

#define JNY_MAX_LAYERS 8
#define JNY_N_PER_LAYER 9
#define JNY_TOKENIZER_MAX 512
#define JNY_ERR_MAGIC -1
#define JNY_ERR_MEM -2
#define JNY_ERR_PART -3
#define JNY_ERR_CTX -4

typedef struct {
    const void *data; /* fp32* if is_fp32, else int8* */
    float scale;      /* 1.0f for fp32 tensors */
    int rows;         /* out features */
    int cols;         /* in features */
    int is_fp32;
} jny_tensor_t;

typedef struct {
    jny_header_t hdr;
    char tokenizer[JNY_TOKENIZER_MAX]; /* char-level vocab string */
    uint8_t *blob;                     /* PSRAM copy of the weights blob */
    size_t blob_size;
    jny_tensor_t t[2 + 9 * JNY_MAX_LAYERS];

    float *rope_cos, *rope_sin;  /* [seq_len * head_dim/2] */
    float *kv;                   /* [n_layers][2][seq_len][kv_dim] */
    float *x, *xb, *q, *k, *v, *att, *h1, *hb, *tmp, *logits;
    int kv_dim, head_dim;
    int pos;
} jny_model_t;

/**
 * Load a .jny model from the flash partition `label` into PSRAM.
 * Returns 0 on success, negative JNY_ERR_* on failure.
 */
int jny_load_from_partition(const char *label, jny_model_t *m);

void jny_free(jny_model_t *m);

/**
 * Generate up to `max_new` tokens after `prompt`.
 * Calls `on_token` per generated char with the running tokens/sec rate.
 * Returns the number of tokens generated, or a negative JNY_ERR_*.
 */
int jny_generate(jny_model_t *m, const char *prompt, int max_new, float temperature,
                 void (*on_token)(char c, float tps, void *user), void *user);

#ifdef __cplusplus
}
#endif
