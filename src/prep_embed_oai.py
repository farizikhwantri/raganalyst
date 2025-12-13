import os
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
import numpy as np

from langchain_classic.docstore.document import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

from faiss_module import FaissIndexStore

# Load OPENAI_API_KEY (export in shell or use .env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Config
EMBED_MODEL = "text-embedding-3-large"  # 3072-dim
EMBED_DIM = 3072
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SUPPORTED_EXT = [".pdf"]

def extract_text_pymupdf(pdf_path: str) -> str:
    """Local text extraction using PyMuPDF (per OpenAI cookbook)."""
    doc = fitz.open(pdf_path)
    texts = []
    for i in range(len(doc)):
        page = doc[i]
        t = page.get_text("text")
        if t and t.strip():
            texts.append(t.strip())
    doc.close()
    return "\n\n".join(texts)

def chunk_documents(extracted_docs: List[Dict]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    chunks: List[Document] = []
    for doc in extracted_docs:
        for chunk in splitter.split_text(doc["text"] or ""):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": doc["filename"],
                        "file": doc["file"],
                        "chunk_id": uuid.uuid4().hex,
                    }
                )
            )
    return chunks

def embed_texts_openai(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts using OpenAI embeddings with batching."""
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for item in resp.data:
            all_vecs.append(item.embedding)
    return np.array(all_vecs, dtype=np.float32) if all_vecs else np.zeros((0, EMBED_DIM), dtype=np.float32)

def build_faiss(chunks: List[Document], outdir: str):
    """
    Save artifacts mirroring prep_embed:
    - rag_index.index
    - rag_index.ids.json
    - metadata.jsonl
    - embeddings.npy
    - images.json (empty)
    """
    os.makedirs(outdir, exist_ok=True)

    texts = [c.page_content for c in chunks]
    vectors = embed_texts_openai(texts, batch_size=64)

    dim = vectors.shape[1] if vectors.size else EMBED_DIM
    store = FaissIndexStore(dim)

    ids = []
    meta_lines = []
    for i, doc in enumerate(chunks):
        cid = doc.metadata.get("chunk_id") or uuid.uuid4().hex
        ids.append(cid)
        meta_lines.append({
            "id": cid,
            "text": doc.page_content,
            "meta": {
                "id": cid,
                "page": None,
                "source_type": "pdf_text",
                "chunk_index": i,
                "source_meta": {"filename": doc.metadata.get("filename")},
                "file": doc.metadata.get("file"),
            },
        })

    if vectors.shape[0] > 0:
        store.add(vectors, ids)

    prefix = os.path.join(outdir, "rag_index")
    store.save(prefix)

    with open(os.path.join(outdir, "metadata.jsonl"), "w", encoding="utf-8") as fh:
        for item in meta_lines:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    np.save(os.path.join(outdir, "embeddings.npy"), vectors)
    with open(os.path.join(outdir, "images.json"), "w", encoding="utf-8") as fh:
        json.dump([], fh, indent=2)

    manifest = {
        "embedding_provider": "openai",
        "embedding_model": EMBED_MODEL,
        "embedding_dim": int(dim),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[Saved] FAISS index and metadata to {outdir}")

def load_and_extract(folder: str) -> List[Dict]:
    """Walk folder, parse PDFs locally with PyMuPDF."""
    results: List[Dict] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_EXT:
                continue
            fpath = str(Path(root) / fname)
            print(f"[Parsing PDF] {fpath}")
            text = extract_text_pymupdf(fpath)
            results.append({"filename": fname, "file": fpath, "text": text})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder containing PDFs")
    parser.add_argument("--outdir", type=str, default="./vectorstore_oai")
    args = parser.parse_args()

    extracted = load_and_extract(args.folder)
    print(f"[Extracted] {len(extracted)} PDFs")

    chunks = chunk_documents(extracted)
    print(f"[Chunked] {len(chunks)} chunks")

    build_faiss(chunks, args.outdir)
    print("[Done] Ingestion complete")

if __name__ == "__main__":
    main()