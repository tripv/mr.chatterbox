#!/usr/bin/env python3
"""
tokenizer_sample.py — build the REUSABLE train/eval text samples for the tokenizer fertility
experiment. One streaming pass over all 15 active corpora:
  • in-band (1800-1899) only, per tag_schema; OOB docs skipped.
  • holdout books (tag_schema.is_holdout(htid)) → eval.jsonl; the rest → train.jsonl.
  • per-source char CAP so Hathi (69% of corpus) doesn't monopolise BPE merges — balanced morphology
    coverage. Shards are already shuffled at build time, so first-N-until-cap is a fair sample.

Materialising the sample once guarantees the 32k/48k/64k tokenizers train on IDENTICAL data (fair
fertility comparison). Output: tokenizer/sample/{train,eval}.jsonl ({"text","source"} per line).

Run: python tokenizer_sample.py                       # defaults: 400M train + 4M eval chars/source
     python tokenizer_sample.py --train-cap 600_000_000
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import pyarrow.parquet as pq
import tag_schema as T

ROOT = "training_data/pretraining"
OUT = "tokenizer/sample"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cap", type=int, default=400_000_000, help="max chars/source for train")
    ap.add_argument("--eval-cap", type=int, default=4_000_000, help="max chars/source for eval")
    a = ap.parse_args()

    dirs = sorted(d for d in os.listdir(ROOT)
                  if os.path.isdir(f"{ROOT}/{d}") and not d.startswith("_"))
    os.makedirs(OUT, exist_ok=True)
    ftr = open(f"{OUT}/train.jsonl", "w"); fev = open(f"{OUT}/eval.jsonl", "w")
    tr_chars = defaultdict(int); ev_chars = defaultdict(int)
    tr_docs = defaultdict(int); ev_docs = defaultdict(int); oob = 0

    def saturated(s):
        return tr_chars[s] >= a.train_cap and ev_chars[s] >= a.eval_cap

    for d in dirs:
        shards = sorted(glob.glob(f"{ROOT}/{d}/*.parquet"))
        seen = set()
        for shard in shards:
            try:
                rows = pq.read_table(shard, columns=["text", "year", "source", "htid"]).to_pylist()
            except Exception as e:
                print(f"  skip {shard} ({e})", file=sys.stderr); continue
            for r in rows:
                src = r["source"]; seen.add(src)
                if not T.in_band(r["year"]):
                    oob += 1; continue
                txt = r["text"]
                if not txt: continue
                if T.is_holdout(r["htid"]):
                    if ev_chars[src] < a.eval_cap:
                        fev.write(json.dumps({"text": txt, "source": src}) + "\n")
                        ev_chars[src] += len(txt); ev_docs[src] += 1
                elif tr_chars[src] < a.train_cap:
                    ftr.write(json.dumps({"text": txt, "source": src}) + "\n")
                    tr_chars[src] += len(txt); tr_docs[src] += 1
            # stop reading this corpus once every source it contains is saturated
            if seen and all(saturated(s) for s in seen):
                break
        print(f"  {d:30} done (sources {sorted(seen)})", file=sys.stderr)
    ftr.close(); fev.close()
    print(f"\n=== sample built (OOB skipped {oob:,}) ===")
    print(f"{'source':16}{'train docs':>12}{'train Mchars':>14}{'eval docs':>11}{'eval Mchars':>13}")
    tt = te = 0
    for s in sorted(tr_chars, key=lambda x: -tr_chars[x]):
        tt += tr_chars[s]; te += ev_chars[s]
        print(f"{s:16}{tr_docs[s]:>12,}{tr_chars[s]/1e6:>14.1f}{ev_docs[s]:>11,}{ev_chars[s]/1e6:>13.2f}")
    print(f"{'TOTAL':16}{sum(tr_docs.values()):>12,}{tt/1e6:>14.1f}{sum(ev_docs.values()):>11,}{te/1e6:>13.2f}")
    print(f"train ≈ {tt/1e9:.2f}B chars, eval ≈ {te/1e6:.0f}M chars → {OUT}/")

if __name__ == "__main__":
    main()
