# minimal example to run a query against produced index
from sentence_transformers import SentenceTransformer
import numpy as np
import json

from faiss_module import FaissIndexStore

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


embedder = SentenceTransformer(EMBED_MODEL_NAME)
embed_dim = embedder.get_sentence_embedding_dimension()
store = FaissIndexStore.load("./data/vector/rag_index", 
                             dim=embed_dim)

# load metadata
meta = {}
with open("./data/vector/metadata.jsonl", "r", encoding="utf-8") as fh:
    for line in fh:
        item = json.loads(line)
        meta[item["id"]] = item

q = "What is BMW's total group revenue in 2021 ?"
qvec = embedder.encode([q], convert_to_numpy=True)
hits = store.search(qvec, k=10)
for hid, dist in hits:
    item = meta[hid]
    print("SCORE", dist, "PAGE", item["meta"]["page"])
    print(item["text"][:400])
    print("-----")
