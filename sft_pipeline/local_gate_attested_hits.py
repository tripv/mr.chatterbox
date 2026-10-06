#!/usr/bin/env python3
"""local_gate_attested_hits.py — run Stage 1 (the passage gate) over the RAW attested-figures
corpus hits using a LOCAL model (mlx_lm/Qwen), not Sonnet. Per Trip (2026-09-03): unlike the
fact_op REWRITE task (where local Qwen was explicitly distrusted after measured failures --
language contamination, template collapse), a gate is a constrained accept/reject
classification over FIXED text, much closer to the RERANK/judge role 03_retrieve_and_judge.py
already trusts Qwen for -- so this reuses gen_teacher_source4.py's own original design choice
(local model for the gate) rather than assuming Sonnet is required for every LLM call in this
project.

Purpose: measure the RAW-HIT -> GATE-PASS yield rate, per entity and in aggregate, over all
~10,000 raw hits from the partial corpus scan -- this tells us how much raw scanning the FULL
corpus needs before Stage 2 (specific question generation, still Sonnet-only) has enough
usable passages per person to work with, rather than guessing.

Uses the EXACT SAME ENTITY_GATE_TMPL as sonnet_attested_figure_pipeline.py (imported, not
duplicated) -- same criteria, only the backend differs. Updated 2026-09-03 to the entity-aware
gate (adds the "not_about_entity" criterion) alongside the sonnet pipeline -- both call sites
must pass the entity name through, or a name/word collision like "Leopold II of Belgium" vs. a
Belgium-treaty passage, or "H. G. Wells" vs. water wells, sails right through as "ok" because
the base gate has no way to know which entity it's even checking against.

  python3 local_gate_attested_hits.py --in probes/attested_corpus_hits.jsonl --limit 200
  python3 local_gate_attested_hits.py --in probes/attested_corpus_hits.jsonl
"""
from __future__ import annotations
import json, re, sys, time, argparse, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
from sonnet_attested_figure_pipeline import ENTITY_GATE_TMPL  # noqa: E402 -- prompt text only, not the backend

DEFAULT_MLX = "mlx-community/Qwen2.5-32B-Instruct-4bit"  # same judge model 03_retrieve_and_judge.py uses


def make_local_gate(model_name: str):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    print(f"loading {model_name} ...")
    m, tok = load(model_name)
    sampler = make_sampler(temp=0.0)

    def gate(entity: str, passage: str) -> dict | None:
        prompt = ENTITY_GATE_TMPL.format(entity=entity, passage=passage)
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        raw = generate(m, tok, prompt=text, max_tokens=200, sampler=sampler, verbose=False)
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return gate


def clean_passage(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=HERE / "probes/attested_corpus_hits.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "probes/attested_corpus_hits_gated.jsonl")
    ap.add_argument("--model", default=DEFAULT_MLX)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} raw hits to gate")

    gate = make_local_gate(args.model)

    stats = {"keep": 0, "reject": collections.Counter(), "fail": 0}
    per_entity = collections.defaultdict(lambda: {"total": 0, "keep": 0})
    t0 = time.time()

    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            passage = clean_passage(row["window"])
            v = gate(row["entity"], passage)
            per_entity[row["entity"]]["total"] += 1
            if v is None:
                stats["fail"] += 1
            elif v.get("passage_ok"):
                stats["keep"] += 1
                per_entity[row["entity"]]["keep"] += 1
                f.write(json.dumps({**row, "window": passage, "gate_note": v.get("note", "")},
                                    ensure_ascii=False) + "\n")
                f.flush()
            else:
                stats["reject"][v.get("reason", "unknown")] += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(rows)}] {stats} | {rate:.1f}/s | "
                      f"eta {(len(rows)-i-1)/rate/60:.1f}min", flush=True)

    print(f"\nFINAL: {stats}")
    total = sum(v["total"] for v in per_entity.values())
    kept = sum(v["keep"] for v in per_entity.values())
    print(f"overall yield: {kept}/{total} = {100*kept/total:.1f}%")

    print("\nper-entity yield (entity: kept/total):")
    for name, v in sorted(per_entity.items(), key=lambda kv: -kv[1]["keep"]):
        print(f"  {v['keep']:2d}/{v['total']:<3d}  {name}")

    zero_yield = [name for name, v in per_entity.items() if v["keep"] == 0]
    print(f"\n{len(zero_yield)}/{len(per_entity)} entities had ZERO passages survive the gate "
          f"(out of those that had >=1 raw hit)")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
