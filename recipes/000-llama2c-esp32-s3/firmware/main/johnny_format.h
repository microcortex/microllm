/*
 * johnny_format.h — MicroCortex model file format "johnny v1"
 *
 * A .jny file is a flat binary, little-endian, written by recipes/*/train
 * and consumed by the johnny runtime (firmware/main/johnny.c).
 *
 * Layout:
 *   [header]    14 x u32/i32  (56 bytes)
 *   [tokenizer] tokenizer_len bytes of UTF-8 (char-level: sorted unique chars)
 *   [weights]   at byte offset data_offset
 *
 * Header (in order):
 *   u32 magic        = 0x31594E4A ("JNY1" little-endian)
 *   u32 version      = 1
 *   u32 quant        = 0 (fp32) | 1 (q8 per-tensor)
 *   i32 dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
 *   u32 tokenizer_len
 *   u32 data_offset  = byte offset of the weights blob
 *   u32 reserved[2]  = 0
 *
 * Weights blob order (row-major, W[out][in]):
 *   token_emb      [vocab_size, dim]
 *   for each layer (n_layers):
 *     rms_attn_w   [dim]                     (fp32, always)
 *     wq           [dim, dim]
 *     wk           [kv_dim, dim]             kv_dim = n_kv_heads * head_dim
 *     wv           [kv_dim, dim]
 *     wo           [dim, dim]
 *     rms_ffn_w    [dim]                     (fp32, always)
 *     w1           [hidden_dim, dim]
 *     w2           [dim, hidden_dim]
 *     w3           [hidden_dim, dim]
 *   rms_final_w    [dim]                     (fp32, always)
 *   wcls           [vocab_size, dim]         (untied)
 *
 * Quantization:
 *   quant == 0: every tensor is fp32 LE.
 *   quant == 1: every 2D matrix is stored as [f32 scale][int8 data] where
 *               scale = max|w|/127 and data = round(w / scale) clamped to
 *               [-127, 127]. 1-D tensors (rmsnorm weights) stay fp32.
 *
 * 2026 — The MicroCortex Authors (MIT). Format inspired by llama2.c (MIT).
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define JNY_MAGIC 0x31594E4Au /* "JNY1" */
#define JNY_VERSION 1u
#define JNY_QUANT_FP32 0u
#define JNY_QUANT_Q8 1u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t quant;
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
    uint32_t tokenizer_len;
    uint32_t data_offset;
    uint32_t reserved[2];
} jny_header_t; /* 56 bytes */

#ifdef __cplusplus
}
#endif
