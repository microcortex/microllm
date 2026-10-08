#!/usr/bin/env python3
"""
johnny_train.py — MicroCortex Recipe 000 trainer & exporter.

Self-contained (no llama2.c dependency): char-level tokenizer, tiny Llama-style
transformer, fp32/bf16 training, and export to the "johnny v1" format (.jny)
consumed by firmware/main/johnny.c — including per-tensor Q8.

Recipe 000 config (micro-6M):
    dim=288 hidden=768 layers=6 heads=6 kv_heads=6 vocab<=256 seq=128
    ~6.1M params -> ~6.1 MB as Q8, fits ESP32-S3 8MB flash/PSRAM.

Usage:
    python johnny_train.py --data input.txt --steps 5000 --out out/
    python johnny_train.py --steps 0 --out out/        # re-export from checkpoint

MIT — 2026 The MicroCortex Authors. Architecture inspired by llama2.c (MIT) and
"Attention Is All You Need". Format spec: firmware/main/johnny_format.h
"""
import argparse
import json
import math
import os
import struct
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

JNY_MAGIC = 0x31594E4A  # "JNY1"
JNY_QUANT_FP32, JNY_QUANT_Q8 = 0, 1

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    dim: int = 288
    hidden_dim: int = 768
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    vocab_size: int = 256  # char-level (auto-set from corpus)
    seq_len: int = 96  # RAM budget on ESP32-S3: blob 6.1MB + KV 1.33MB ≈ 7.6/8 MB

    @property
    def head_dim(self):
        assert self.dim % self.n_heads == 0
        return self.dim // self.n_heads

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim


# --------------------------------------------------------------------------- #
# Tokenizer (char-level)
# --------------------------------------------------------------------------- #
class CharTokenizer:
    def __init__(self, text: str):
        self.itos = sorted(set(text))
        assert len(self.itos) <= 256, f"corpus has {len(self.itos)} unique chars (>256)"
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


# --------------------------------------------------------------------------- #
# Model (llama-style: RMSNorm, RoPE, SwiGLU, GQA)
# --------------------------------------------------------------------------- #
def rms_norm(x, w, eps=1e-5):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def apply_rope(x, cos, sin):
    # x: (bsz, n_heads, seq, head_dim); cos/sin: (seq, head_dim/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = 0
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.kv_dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin, cache=None):
        b, s, _ = x.shape
        cfg = self.cfg
        q = self.wq(x).view(b, s, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:  # incremental decode
            lc = cache[self.layer_idx]
            pos = lc["pos"]
            lc["k"][:, pos : pos + s] = k
            lc["v"][:, pos : pos + s] = v
            k = lc["k"][:, : pos + s]
            v = lc["v"][:, : pos + s]
            lc["pos"] = pos + s

        if cfg.n_kv_heads != cfg.n_heads:  # GQA
            rep = cfg.n_heads // cfg.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        if cache is None:  # full-sequence forward: causal mask
            mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), 1)
            att = att.masked_fill(mask, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.wo(y.transpose(1, 2).reshape(b, s, cfg.dim))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.layer_idx = 0
        self.rms_attn = nn.Parameter(torch.ones(cfg.dim))
        self.attn = Attention(cfg)
        self.rms_ffn = nn.Parameter(torch.ones(cfg.dim))
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(rms_norm(x, self.rms_attn), cos, sin, cache)
        xn = rms_norm(x, self.rms_ffn)
        return x + self.w2(F.silu(self.w1(xn)) * self.w3(xn))


class MicroLlama(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        for i, b in enumerate(self.blocks):
            b.layer_idx = i
            b.attn.layer_idx = i
        self.rms_final = nn.Parameter(torch.ones(cfg.dim))
        self.wcls = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        theta = 1.0 / (10000 ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
        t = torch.arange(cfg.seq_len).float()
        self.register_buffer("rope_cos", t.outer(theta).cos(), persistent=False)
        self.register_buffer("rope_sin", t.outer(theta).sin(), persistent=False)

    def rope(self, seq_len, device):
        return self.rope_cos[:seq_len].to(device), self.rope_sin[:seq_len].to(device)

    def forward(self, tokens, cache=None):
        x = self.tok_emb(tokens)
        cos, sin = self.rope(x.shape[1], x.device)
        for b in self.blocks:
            x = b(x, cos, sin, cache)
        return self.wcls(rms_norm(x, self.rms_final))


# --------------------------------------------------------------------------- #
# Export — must match firmware/main/johnny_format.h exactly
# --------------------------------------------------------------------------- #
def build_tensor_list(cfg: Config, sd: dict):
    tensors = [("tok_emb", sd["tok_emb.weight"].detach())]
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        tensors += [
            ("rms_attn_w", sd[p + "rms_attn"].detach()),
            ("wq", sd[p + "attn.wq.weight"].detach()),
            ("wk", sd[p + "attn.wk.weight"].detach()),
            ("wv", sd[p + "attn.wv.weight"].detach()),
            ("wo", sd[p + "attn.wo.weight"].detach()),
            ("rms_ffn_w", sd[p + "rms_ffn"].detach()),
            ("w1", sd[p + "w1.weight"].detach()),
            ("w2", sd[p + "w2.weight"].detach()),
            ("w3", sd[p + "w3.weight"].detach()),
        ]
    tensors += [("rms_final_w", sd["rms_final"].detach()), ("wcls", sd["wcls.weight"].detach())]
    return tensors


def pack_tensor(t: torch.Tensor, quant: int) -> bytes:
    """Serialize one tensor: fp32; or [f32 scale][int8] when quant==Q8 and 2-D."""
    if t.dim() == 1 or quant == JNY_QUANT_FP32:
        return t.to(torch.float32).numpy().astype("<f4").tobytes()
    w = t.to(torch.float32)
    scale = w.abs().max().clamp(min=1e-8).item() / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return struct.pack("<f", scale) + q.numpy().tobytes()


def export_jny(path, cfg: Config, sd: dict, tokenizer: CharTokenizer, quant: int):
    tensors = build_tensor_list(cfg, sd)
    vocab_str = "".join(tokenizer.itos).encode("utf-8")
    body = b"".join(pack_tensor(t, quant) for _, t in tensors)
    data_offset = 56 + len(vocab_str)
    header = struct.pack(
        "<IIi7iII2I",
        JNY_MAGIC, 1, quant,
        cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
        cfg.vocab_size, cfg.seq_len,
        len(vocab_str), data_offset, 0, 0,
    )
    assert len(header) == 56, len(header)
    with open(path, "wb") as f:
        f.write(header + vocab_str + body)
    n_params = sum(t.numel() for _, t in tensors)
    mb = os.path.getsize(path) / 1e6
    print(f"[export] {path}: {n_params/1e6:.2f}M params -> {mb:.2f} MB (quant={'q8' if quant else 'fp32'})")


# --------------------------------------------------------------------------- #
# Data & sampling
# --------------------------------------------------------------------------- #
def load_corpus(path: str) -> str:
    if not os.path.exists(path):
        if os.path.basename(path) == "input.txt":
            print(f"[data] downloading tiny shakespeare -> {path}")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        else:
            raise FileNotFoundError(path)
    return open(path, encoding="utf-8", errors="ignore").read()


def get_batch(ids, batch_size, seq_len, device):
    ix = torch.randint(len(ids) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(ids[i : i + seq_len]) for i in ix])
    y = torch.stack([torch.tensor(ids[i + 1 : i + 1 + seq_len]) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def sample(model, tokenizer, prompt, max_new=128, temperature=0.8, device="cpu"):
    cfg = model.cfg
    cache = [
        {
            "k": torch.zeros(1, cfg.n_kv_heads, cfg.seq_len, cfg.head_dim),
            "v": torch.zeros(1, cfg.n_kv_heads, cfg.seq_len, cfg.head_dim),
            "pos": 0,
        }
        for _ in range(cfg.n_layers)
    ]
    ids = tokenizer.encode(prompt)[-cfg.seq_len :]
    out = list(ids)
    was_training = model.training
    model.eval()
    for i in range(max_new):
        window = ids if i == 0 else ids[-1:]
        logits = model(torch.tensor([window], device=device), cache=cache)
        logits = logits[0, -1] / temperature
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        out.append(nxt)
        ids = (ids + [nxt])[-cfg.seq_len :]
    if was_training:
        model.train()
    return tokenizer.decode(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--out", default="out/")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--quant", choices=["fp32", "q8"], default="q8")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    text = load_corpus(args.data)
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    cfg = Config(vocab_size=len(tok.itos))
    model = MicroLlama(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params | {device} | vocab {cfg.vocab_size} | seq {cfg.seq_len}")

    if args.steps > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        use_amp = device == "cuda"
        model.train()
        ckpt_path = os.path.join(args.out, "model_ckpt.pt")
        start_step = 0
        if os.path.exists(ckpt_path):  # resume
            sd = torch.load(ckpt_path, weights_only=True)
            model.load_state_dict(sd)
            start_step = int(input("[resume] checkpoint found — enter step to resume from (e.g. 1500): ") or 0)
            print(f"[resume] continuing from step {start_step}")
        for step in range(start_step + 1, args.steps + 1):
            x, y = get_batch(ids, args.batch_size, cfg.seq_len, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(0, 1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 50 == 0 or step == 1:
                print(f"[step {step:5d}] loss {loss.item():.4f} | lr {sched.get_last_lr()[0]:.2e}")
            if step % 500 == 0 or step == args.steps:  # periodic ckpt: CPU training is long
                torch.save(model.state_dict(), os.path.join(args.out, "model_ckpt.pt"))
            if step % args.eval_every == 0 or step == args.steps:
                print("[sample]", repr(sample(model, tok, "JULIET:", 96, args.temperature, device)[:120]))
        torch.save(model.state_dict(), os.path.join(args.out, "model_ckpt.pt"))

    sd = torch.load(os.path.join(args.out, "model_ckpt.pt"), weights_only=True)
    quant = JNY_QUANT_Q8 if args.quant == "q8" else JNY_QUANT_FP32
    export_jny(os.path.join(args.out, f"model_{args.quant.upper()}.jny"), cfg, sd, tok, quant)
    json.dump({"config": vars(cfg), "tokenizer": tok.itos}, open(os.path.join(args.out, "meta.json"), "w"))
    print("[done] flash the Q8 file to the 'model' partition (offset 0x190000):")
    print(f"  esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 write_flash 0x190000 {args.out}/model_{args.quant.upper()}.jny")


if __name__ == "__main__":
    main()
