#!/usr/bin/env python3
"""
judge_short_passage_pairs.py -- Stage 4 (round-trip consistency) of the short-answer coverage fix.

Stage 3's judge picks the BEST of up to 15 retrieved candidates on relevance + voice -- a relative,
best-of-K selection. This stage independently re-judges each ACCEPTED (real question, chosen
passage) pair on its own absolute merits, with a fresh judge call that never saw the other 14
candidates -- catching cases where "best of a weak field" still wasn't good enough, the same
failure mode judge_ceiling_trims.py's adequacy check targets. A pair must pass BOTH stage 3 and
this stage to make the final cut.

  python3 judge_short_passage_pairs.py --backend mlx --limit 20   # calibration
  python3 judge_short_passage_pairs.py --backend mlx              # full run
"""
from __future__ import annotations
import json, re, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_PATH = HERE / "retrieved_short_passages.jsonl"
OUT_PATH = HERE / "short_passage_pair_judgments.jsonl"
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"

JUDGE_TMPL = """A training pair for a Victorian-era conversational model is being independently
double-checked (this is a second, separate judgment -- do not assume a prior check was correct).

A modern user said:
  "{prompt}"

The Victorian character's reply (real 19th-century text, selected verbatim):
  "{victorian}"

Judging this pair ALONE, with no other candidates to compare against: does the reply genuinely and
coherently address what the user asked, in a way that would read naturally as a real exchange --
not just topically adjacent, but an actual fitting response? Reject if the connection feels forced,
if the reply ignores the specific thing asked, or if it only works when read charitably.

Reply with ONLY JSON: {{"verdict": "PASS" or "FAIL", "why": "<one short phrase>"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    ap.add_argument("--backend", choices=["mlx"], default="mlx")
    ap.add_argument("--model", default=DEFAULT_MLX_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} pairs to re-judge")

    from mlx_lm import load, generate
    print(f"loading judge {args.model} ...")
    jm, jt = load(args.model)

    def judge(prompt_text):
        out = generate(jm, jt, prompt=jt.apply_chat_template(
            [{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False
        ), max_tokens=80, verbose=False)
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
            return str(d.get("verdict", "")).upper().strip(), str(d.get("why", ""))[:120]
        except Exception:
            return "", "parse-fail"

    n_pass = n_fail_v = n_fail_parse = 0
    results = []
    for i, r in enumerate(rows):
        prompt = JUDGE_TMPL.format(prompt=r["prompt"], victorian=r["victorian"])
        verdict, why = judge(prompt)
        if verdict not in ("PASS", "FAIL"):
            n_fail_parse += 1
            verdict = "FAIL"
        elif verdict == "PASS":
            n_pass += 1
        else:
            n_fail_v += 1
        results.append({**r, "verdict": verdict, "why": why})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}  pass={n_pass} fail={n_fail_v} parse_fail={n_fail_parse}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone: {n_pass} pass, {n_fail_v} fail, {n_fail_parse} parse-failures (treated as fail)")
    print(f"wrote -> {args.output}")


if __name__ == "__main__":
    main()
