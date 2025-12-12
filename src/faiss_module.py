import os
import json
from typing import List,  Tuple, Optional


from pdf2image import convert_from_path
import numpy as np

import faiss

from utils import ensure_dir

# -------- Embedding & FAISS --------
class FaissIndexStore:
    def __init__(self, dim: int, index_path: Optional[str] = None):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)  # simple flat L2 index
        self.ids = []  # parallel list of ids (metadata index mapping)

    def add(self, vectors: np.ndarray, ids: List[str]):
        assert vectors.shape[1] == self.dim
        self.index.add(vectors.astype('float32'))
        self.ids.extend(ids)

    def save(self, path_prefix: str):
        ensure_dir(os.path.dirname(path_prefix) or ".")
        faiss.write_index(self.index, path_prefix + ".index")
        # save ids
        with open(path_prefix + ".ids.json", "w", encoding="utf-8") as fh:
            json.dump(self.ids, fh)

    @classmethod
    def load(cls, path_prefix: str, dim: int):
        obj = cls(dim, index_path=path_prefix)
        obj.index = faiss.read_index(path_prefix + ".index")
        with open(path_prefix + ".ids.json", "r", encoding="utf-8") as fh:
            obj.ids = json.load(fh)
        return obj

    def search(self, qvec: np.ndarray, k: int = 4) -> List[Tuple[str, float]]:
        D, I = self.index.search(qvec.astype('float32'), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((self.ids[idx], float(dist)))
        return results