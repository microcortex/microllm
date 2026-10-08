/*
 * johnny.c — MicroCortex tiny-LLM runtime for ESP32-S3 (recipe 000)
 *
 * MIT — 2026 The MicroCortex Authors. Inspired by llama2.c (Karpathy, MIT).
 *
 * v1 design notes:
 *  - The whole weights blob is copied from the flash partition into PSRAM
 *    (N8R8 has 8 MB; a ~6.1 MB Q8 blob + fp32 KV cache for seq=96 fits).
 *  - Q8 tensors are per-tensor int8 with an fp32 scale; dequantization happens
 *    in the accumulator (later recipes swap this for esp-nn SIMD kernels).
 *  - RoPE tables and the KV cache are fp32; activations are fp32.
 */
#include "johnny.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"

static const char *TAG = "johnny";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

static void matmul_f32(float *out, const float *w, const float *x, int n_out, int n_in)
{
    for (int o = 0; o < n_out; o++) {
        const float *row = w + (size_t)o * n_in;
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) acc += row[i] * x[i];
        out[o] = acc;
    }
}

static void matmul_q8(float *out, const int8_t *w, float scale, const float *x, int n_out, int n_in)
{
    for (int o = 0; o < n_out; o++) {
        const int8_t *row = w + (size_t)o * n_in;
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) acc += (float)row[i] * x[i];
        out[o] = acc * scale;
    }
}

static void matmul_any(float *out, const jny_tensor_t *t, const float *x)
{
    if (t->is_fp32) matmul_f32(out, (const float *)t->data, x, t->rows, t->cols);
    else matmul_q8(out, (const int8_t *)t->data, t->scale, x, t->rows, t->cols);
}

static void vec_rmsnorm(float *out, const float *x, const float *w, int n)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}

/* Apply RoPE to n_heads contiguous heads of head_dim elements. */
static void rope_apply(float *vec, const jny_model_t *m, int n_heads, int pos)
{
    const int hd = m->head_dim;
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * hd;
        for (int j = 0; j < hd / 2; j++) {
            const float c = m->rope_cos[pos * (hd / 2) + j];
            const float s = m->rope_sin[pos * (hd / 2) + j];
            const float a = v[2 * j], b = v[2 * j + 1];
            v[2 * j]     = a * c - b * s;
            v[2 * j + 1] = a * s + b * c;
        }
    }
}

/* Dequantize one row of a 2-D tensor into out (fp32). */
static void dequant_row(float *out, const jny_tensor_t *t, int row)
{
    const int cols = t->cols;
    if (t->is_fp32) {
        memcpy(out, (const float *)t->data + (size_t)row * cols, cols * sizeof(float));
    } else {
        const int8_t *w = (const int8_t *)t->data + (size_t)row * cols;
        for (int i = 0; i < cols; i++) out[i] = (float)w[i] * t->scale;
    }
}

/* ------------------------------------------------------------------ */
/* Layout                                                              */
/* ------------------------------------------------------------------ */

/* Byte size of one tensor inside the blob (must match train/johnny_train.py). */
static size_t jny_tensor_bytes(int n_elems, uint32_t quant)
{
    if (quant == JNY_QUANT_FP32) return (size_t)n_elems * 4;
    return 4 + (size_t)(((n_elems + 3) & ~3)); /* f32 scale + int8 data + pad to 4 */
}

static size_t jny_expected_blob_size(const jny_header_t *h)
{
    const int dim = h->dim, hidden = h->hidden_dim;
    const int kv_dim = h->n_kv_heads * (dim / h->n_heads);
    const uint32_t q = h->quant;
    size_t total = jny_tensor_bytes((size_t)h->vocab_size * dim, q); /* tok_emb */
    for (int l = 0; l < h->n_layers; l++) {
        total += 4 * dim;                                              /* rms_attn */
        total += jny_tensor_bytes((size_t)dim * dim, q);               /* wq */
        total += jny_tensor_bytes((size_t)kv_dim * dim, q);            /* wk */
        total += jny_tensor_bytes((size_t)kv_dim * dim, q);            /* wv */
        total += jny_tensor_bytes((size_t)dim * dim, q);               /* wo */
        total += 4 * dim;                                              /* rms_ffn */
        total += jny_tensor_bytes((size_t)hidden * dim, q);            /* w1 */
        total += jny_tensor_bytes((size_t)dim * hidden, q);            /* w2 */
        total += jny_tensor_bytes((size_t)hidden * dim, q);            /* w3 */
    }
    total += 4 * dim;                                                  /* rms_final */
    total += jny_tensor_bytes((size_t)h->vocab_size * dim, q);         /* wcls */
    return total;
}

static void set_shape(jny_model_t *m, int idx, int rows, int cols)
{
    m->t[idx].rows = rows;
    m->t[idx].cols = cols;
}

static int jny_parse_tensors(jny_model_t *m)
{
    const jny_header_t *h = &m->hdr;
    const int dim = h->dim, hidden = h->hidden_dim;
    m->head_dim = dim / h->n_heads;
    m->kv_dim = h->n_kv_heads * m->head_dim;
    const int kv_dim = m->kv_dim;

    /* declare shapes in blob order */
    int idx = 0;
    set_shape(m, idx++, h->vocab_size, dim); /* tok_emb */
    for (int l = 0; l < h->n_layers; l++) {
        const int b = 1 + 9 * l;
        set_shape(m, b + 0, dim, 1);         /* rms_attn (1-D, fp32) */
        set_shape(m, b + 1, dim, dim);       /* wq */
        set_shape(m, b + 2, kv_dim, dim);    /* wk */
        set_shape(m, b + 3, kv_dim, dim);    /* wv */
        set_shape(m, b + 4, dim, dim);       /* wo */
        set_shape(m, b + 5, dim, 1);         /* rms_ffn (1-D, fp32) */
        set_shape(m, b + 6, hidden, dim);    /* w1 */
        set_shape(m, b + 7, dim, hidden);    /* w2 */
        set_shape(m, b + 8, hidden, dim);    /* w3 */
    }
    set_shape(m, idx++, dim, 1);             /* rms_final (1-D, fp32) */
    set_shape(m, idx++, h->vocab_size, dim); /* wcls */
    const int n_tensors = idx;

    /* walk the blob */
    const uint8_t *p = m->blob;
    for (int i = 0; i < n_tensors; i++) {
        jny_tensor_t *t = &m->t[i];
        const int n = t->rows * t->cols;
        if (t->cols == 1 || h->quant == JNY_QUANT_FP32) {
            t->data = p;
            t->scale = 1.0f;
            t->is_fp32 = 1;
            p += (size_t)n * 4;
        } else {
            memcpy(&t->scale, p, sizeof(float));
            p += 4;
            t->data = p;
            t->is_fp32 = 0;
            p += n;
            p = (const uint8_t *)(((uintptr_t)p + 3) & ~(uintptr_t)3); /* pad to 4 */
        }
        if (t->data == NULL) {
            ESP_LOGE(TAG, "tensor %d (%dx%d) got NULL data", i, t->rows, t->cols);
            return JNY_ERR_MAGIC;
        }
    }
    if ((size_t)(p - m->blob) != m->blob_size) {
        ESP_LOGE(TAG, "blob size mismatch: walked %d, header math says %d",
                 (int)(p - m->blob), (int)m->blob_size);
        return JNY_ERR_MAGIC;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Load / free                                                         */
/* ------------------------------------------------------------------ */

int jny_load_from_partition(const char *label, jny_model_t *m)
{
    memset(m, 0, sizeof(*m));

    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, label);
    if (part == NULL) {
        ESP_LOGE(TAG, "partition '%s' not found (type data, subtype 0x40)", label);
        return JNY_ERR_PART;
    }

    esp_partition_read(part, 0, &m->hdr, sizeof(m->hdr));
    if (m->hdr.magic != JNY_MAGIC) {
        ESP_LOGE(TAG, "bad magic 0x%08" PRIx32 " — is the model flashed?", m->hdr.magic);
        return JNY_ERR_MAGIC;
    }
    if (m->hdr.version != JNY_VERSION) {
        ESP_LOGE(TAG, "unsupported format version %" PRIu32, m->hdr.version);
        return JNY_ERR_MAGIC;
    }
    if (m->hdr.n_layers > JNY_MAX_LAYERS || m->hdr.tokenizer_len > JNY_TOKENIZER_MAX) {
        return JNY_ERR_MAGIC;
    }

    ESP_LOGI(TAG, "header in flash: dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d "
             "seq=%d quant=%u tokenizer=%u data_offset=%u",
             (int)m->hdr.dim, (int)m->hdr.hidden_dim, (int)m->hdr.n_layers,
             (int)m->hdr.n_heads, (int)m->hdr.n_kv_heads, (int)m->hdr.vocab_size,
             (int)m->hdr.seq_len, m->hdr.quant, m->hdr.tokenizer_len, m->hdr.data_offset);

    esp_partition_read(part, sizeof(jny_header_t), m->tokenizer,
                       m->hdr.tokenizer_len < JNY_TOKENIZER_MAX ? m->hdr.tokenizer_len
                                                                : JNY_TOKENIZER_MAX - 1);

    m->blob_size = jny_expected_blob_size(&m->hdr);
    const size_t avail = part->size - m->hdr.data_offset;
    if (m->blob_size > avail) {
        ESP_LOGE(TAG, "partition too small: need %d KB, have %d KB",
                 (int)(m->blob_size / 1024), (int)(avail / 1024));
        return JNY_ERR_MEM;
    }

    m->blob = heap_caps_malloc(m->blob_size, MALLOC_CAP_SPIRAM);
    if (m->blob == NULL) {
        ESP_LOGE(TAG, "PSRAM alloc failed for %d KB blob", (int)(m->blob_size / 1024));
        return JNY_ERR_MEM;
    }
    esp_err_t rd = esp_partition_read(part, m->hdr.data_offset, m->blob, m->blob_size);
    if (rd != ESP_OK) {
        ESP_LOGE(TAG, "partition read failed: %d", (int)rd);
        return JNY_ERR_PART;
    }

    int rc = jny_parse_tensors(m);
    if (rc != 0) return rc;

    const int dim = m->hdr.dim, seq = m->hdr.seq_len;
    const size_t kv_floats = (size_t)m->hdr.n_layers * 2 * seq * m->kv_dim;
    m->kv = heap_caps_malloc(kv_floats * sizeof(float), MALLOC_CAP_SPIRAM);
    m->rope_cos = malloc((size_t)seq * (m->head_dim / 2) * sizeof(float));
    m->rope_sin = malloc((size_t)seq * (m->head_dim / 2) * sizeof(float));
    m->x = malloc(dim * sizeof(float));
    m->xb = malloc(dim * sizeof(float));
    m->q = malloc(dim * sizeof(float));
    m->k = malloc(m->kv_dim * sizeof(float));
    m->v = malloc(m->kv_dim * sizeof(float));
    m->att = malloc(dim * sizeof(float));
    m->h1 = malloc(m->hdr.hidden_dim * sizeof(float));
    m->hb = malloc(m->hdr.hidden_dim * sizeof(float));
    m->tmp = malloc(dim * sizeof(float));
    m->logits = malloc(m->hdr.vocab_size * sizeof(float));
    if (!m->kv || !m->rope_cos || !m->rope_sin || !m->x || !m->xb || !m->q ||
        !m->k || !m->v || !m->att || !m->h1 || !m->hb || !m->tmp || !m->logits) {
        ESP_LOGE(TAG, "scratch allocation failed");
        return JNY_ERR_MEM;
    }

    for (int pos_i = 0; pos_i < seq; pos_i++) {
        for (int j = 0; j < m->head_dim / 2; j++) {
            const float theta = powf(10000.0f, -2.0f * j / (float)m->head_dim);
            m->rope_cos[pos_i * (m->head_dim / 2) + j] = cosf(pos_i * theta);
            m->rope_sin[pos_i * (m->head_dim / 2) + j] = sinf(pos_i * theta);
        }
    }

    ESP_LOGI(TAG, "model '%s': dim=%d layers=%d heads=%d kv=%d vocab=%d seq=%d"
                  " quant=%s (blob %d KB, kv %d KB)",
             label, (int)m->hdr.dim, (int)m->hdr.n_layers, (int)m->hdr.n_heads,
             (int)m->hdr.n_kv_heads, (int)m->hdr.vocab_size, (int)seq,
             m->hdr.quant == JNY_QUANT_Q8 ? "q8" : "fp32",
             (int)(m->blob_size / 1024), (int)(kv_floats * 4 / 1024));
    return 0;
}

void jny_free(jny_model_t *m)
{
    free(m->blob); free(m->kv); free(m->rope_cos); free(m->rope_sin);
    free(m->x); free(m->xb); free(m->q); free(m->k); free(m->v);
    free(m->att); free(m->h1); free(m->hb); free(m->tmp); free(m->logits);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------------------------------------------ */
/* Forward (one token, KV-cached)                                      */
/* ------------------------------------------------------------------ */

static void jny_forward(jny_model_t *m, int token)
{
    const jny_header_t *h = &m->hdr;
    const int dim = h->dim;
    const int seq = h->seq_len;
    const float attn_scale = 1.0f / sqrtf((float)m->head_dim);

    dequant_row(m->x, &m->t[0], token); /* token embedding */

    for (int l = 0; l < h->n_layers; l++) {
        const int base = 1 + 9 * l;
        float *K = m->kv + (size_t)l * 2 * seq * m->kv_dim;
        float *V = K + (size_t)seq * m->kv_dim;

        /* --- attention --- */
        vec_rmsnorm(m->xb, m->x, (const float *)m->t[base + 0].data, dim);
        matmul_any(m->q, &m->t[base + 1], m->xb); /* wq */
        matmul_any(m->k, &m->t[base + 2], m->xb); /* wk */
        matmul_any(m->v, &m->t[base + 3], m->xb); /* wv */
        rope_apply(m->q, m, h->n_heads, m->pos);
        rope_apply(m->k, m, h->n_kv_heads, m->pos);

        float *Kl = K + (size_t)m->pos * m->kv_dim;
        float *Vl = V + (size_t)m->pos * m->kv_dim;
        memcpy(Kl, m->k, m->kv_dim * sizeof(float));
        memcpy(Vl, m->v, m->kv_dim * sizeof(float));

        const int groups = h->n_heads / h->n_kv_heads;
        for (int hh = 0; hh < h->n_heads; hh++) {
            const float *qh = m->q + hh * m->head_dim;
            const int g = hh / groups;
            float maxs = -INFINITY;
            for (int t = 0; t <= m->pos; t++) {
                const float *kt = K + (size_t)t * m->kv_dim + g * m->head_dim;
                float s = 0.0f;
                for (int d = 0; d < m->head_dim; d++) s += qh[d] * kt[d];
                m->att[t] = s * attn_scale;
                if (m->att[t] > maxs) maxs = m->att[t];
            }
            float sum = 0.0f;
            for (int t = 0; t <= m->pos; t++) {
                m->att[t] = expf(m->att[t] - maxs);
                sum += m->att[t];
            }
            float *yh = m->tmp + hh * m->head_dim;
            memset(yh, 0, m->head_dim * sizeof(float));
            for (int t = 0; t <= m->pos; t++) {
                const float w = m->att[t];
                const float *vt = V + (size_t)t * m->kv_dim + g * m->head_dim;
                for (int d = 0; d < m->head_dim; d++) yh[d] += w * vt[d];
            }
        }
        matmul_any(m->xb, &m->t[base + 4], m->tmp); /* wo (out of place!) */
        for (int i = 0; i < dim; i++) m->x[i] += m->xb[i];

        /* --- ffn --- */
        vec_rmsnorm(m->xb, m->x, (const float *)m->t[base + 5].data, dim);
        matmul_any(m->h1, &m->t[base + 6], m->xb); /* w1 */
        matmul_any(m->hb, &m->t[base + 8], m->xb); /* w3 */
        for (int i = 0; i < h->hidden_dim; i++) {
            const float g = m->h1[i];
            m->h1[i] = g / (1.0f + expf(-g)) * m->hb[i]; /* silu(w1x) * (w3x) */
        }
        matmul_any(m->tmp, &m->t[base + 7], m->h1); /* w2 */
        for (int i = 0; i < dim; i++) m->x[i] += m->tmp[i];
    }

    vec_rmsnorm(m->xb, m->x, (const float *)m->t[1 + 9 * h->n_layers].data, dim);
    matmul_any(m->logits, &m->t[2 + 9 * h->n_layers], m->xb); /* wcls */
    m->pos++;
}

static int jny_sample(const jny_model_t *m, float temperature)
{
    float maxl = -INFINITY;
    for (int i = 0; i < m->hdr.vocab_size; i++)
        if (m->logits[i] > maxl) maxl = m->logits[i];
    float sum = 0.0f;
    for (int i = 0; i < m->hdr.vocab_size; i++) {
        m->logits[i] = expf((m->logits[i] - maxl) / temperature);
        sum += m->logits[i];
    }
    float r = (float)rand() / ((float)RAND_MAX + 1.0f) * sum;
    for (int i = 0; i < m->hdr.vocab_size; i++) {
        r -= m->logits[i];
        if (r <= 0.0f) return i;
    }
    return m->hdr.vocab_size - 1;
}

/* ------------------------------------------------------------------ */
/* Generate                                                            */
/* ------------------------------------------------------------------ */

int jny_generate(jny_model_t *m, const char *prompt, int max_new, float temperature,
                 void (*on_token)(char c, float tps, void *user), void *user)
{
    if (m->hdr.magic != JNY_MAGIC) return JNY_ERR_MAGIC;
    m->pos = 0;

    int ids[JNY_TOKENIZER_MAX];
    int n_prompt = 0;
    for (const char *p = prompt; *p; ++p) {
        const char *hit = memchr(m->tokenizer, *p, m->hdr.tokenizer_len);
        if (hit) ids[n_prompt++] = (int)(hit - m->tokenizer);
        if (n_prompt >= m->hdr.seq_len - 1) break;
    }
    if (n_prompt == 0) return 0;

    for (int i = 0; i < n_prompt; i++) jny_forward(m, ids[i]); /* prefill */

    double ema_tps = 0.0;
    int generated = 0;
    for (int g = 0; g < max_new; g++) {
        const int next = jny_sample(m, temperature);
        const char c = m->tokenizer[next];
        if (on_token) on_token(c, (float)ema_tps, user);
        generated++;

        if (m->pos >= m->hdr.seq_len) break; /* context full */
        const int64_t t0 = esp_timer_get_time();
        jny_forward(m, next);
        const int64_t dt = esp_timer_get_time() - t0;
        const float tps = 1e6f / (float)(dt > 0 ? dt : 1);
        ema_tps = (ema_tps == 0.0) ? tps : 0.9 * ema_tps + 0.1 * tps;
    }
    return generated;
}
