import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
HERE=Path(__file__).resolve().parent
meta=[json.loads(l) for l in open(HERE/"index_meta.jsonl")]
resps=[m["resp"] for m in meta]
print(f"embedding {len(resps)} responses ...")
m=SentenceTransformer("all-MiniLM-L6-v2")
emb=m.encode(resps,batch_size=1024,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True).astype("float32")
np.save(HERE/"index_resp.npy",emb)
print("saved index_resp.npy",emb.shape)
