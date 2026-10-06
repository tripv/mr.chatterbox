#!/usr/bin/env python3
"""
llama_probe_configs.py — LlamaConfig definitions for the scaling-probe family + the final target.
Tied embeddings (Llama-3.2-1B/3B style), SwiGLU, MHA (set kv<heads for GQA), 64k vocab + the 124
reserved tag/chat tokens already in the tokenizer.

PROBES are the validation/scaling-probe sizes (345M / 500M / 1.2B); FINAL is the ~2.7B Chinchilla-
optimal target at the locked 64k tokenizer (~54.8B tokens / 20:1).
"""
from transformers import LlamaConfig
import tag_schema as T

VOCAB = 65536
EOS_TOKEN = T.EOS

PROBES = {
    "345M": dict(hidden=1024, layers=22, heads=8,  kv=8,  inter=2816),
    "500M": dict(hidden=1280, layers=21, heads=10, kv=10, inter=3328),
    "1.2B": dict(hidden=2048, layers=21, heads=16, kv=16, inter=5376),
}
FINAL = {"2.7B": dict(hidden=2560, layers=32, heads=20, kv=20, inter=6912)}

# Great Exhibition: single-pass Chinchilla-optimal size for the ~13.4B trainable tokens that exist
# for year<=1850 (see tag_schema.GREAT_EXHIBITION_YEAR_MAX / victoriana_data's year-cap filter).
# 667M actual params * 20 ~= 13.3B, matching the available pre-1851 corpus almost exactly.
GREAT_EXHIBITION = {"Great-Exhibition": dict(hidden=1536, layers=20, heads=12, kv=12, inter=4096)}

def build_config(spec, eos_id, seq_len=2048):
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=spec["hidden"],
        num_hidden_layers=spec["layers"],
        num_attention_heads=spec["heads"],
        num_key_value_heads=spec["kv"],
        intermediate_size=spec["inter"],
        max_position_embeddings=seq_len,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10000.0,
        eos_token_id=eos_id,
        bos_token_id=None,
        pad_token_id=None,
    )

if __name__ == "__main__":
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("tokenizer/victoriana-base/tokenizer.json")
    eos = tok.token_to_id(EOS_TOKEN)
    from transformers import LlamaForCausalLM
    print(f"{'name':6}{'hidden':>7}{'layers':>7}{'heads':>6}{'kv':>4}{'inter':>7}{'params':>10}")
    for name, spec in {**PROBES, **FINAL, **GREAT_EXHIBITION}.items():
        cfg = build_config(spec, eos)
        m = LlamaForCausalLM(cfg)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:6}{spec['hidden']:>7}{spec['layers']:>7}{spec['heads']:>6}"
              f"{spec['kv']:>4}{spec['inter']:>7}{n/1e6:>9.0f}M")
        del m
