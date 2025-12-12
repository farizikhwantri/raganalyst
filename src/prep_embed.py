import os
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from faiss_module import FaissIndexStore

from utils import ensure_dir
from utils import save_jsonl
from utils import extract_images_from_page
from utils import extract_text_blocks_with_layout
from utils import extract_tables_pdfplumber
from utils import ocr_page_image

# -------- Configuration --------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # for all-MiniLM-L6-v2
CHUNK_SIZE = 800     # characters per chunk (tune for your use)
CHUNK_OVERLAP = 150  # overlap between chunks
OCR_DPI = 300        # DPI for pdf2image OCR conversion
OCR_LANG = "eng"     # change to 'deu' or others as needed

# # -------- Chunking --------
# def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
#     """Simple character-based chunker preserving sentence boundaries where possible."""
#     text = text.strip()
#     if not text:
#         return []
#     # naive split by paragraphs first
#     paras = [p.strip() for p in text.split("\n\n") if p.strip()]
#     chunks = []
#     current = ""
#     for p in paras:
#         if len(current) + len(p) + 1 <= chunk_size:
#             current = (current + "\n\n" + p).strip() if current else p
#         else:
#             if current:
#                 chunks.append(current)
#             # if paragraph itself too big, split it into sentence-ish pieces
#             if len(p) <= chunk_size:
#                 current = p
#             else:
#                 # fall back to sliding window
#                 start = 0
#                 while start < len(p):
#                     end = min(start + chunk_size, len(p))
#                     chunks.append(p[start:end])
#                     start = end - overlap if end - overlap > start else end
#                 current = ""
#     if current:
#         chunks.append(current)
#     # apply overlap merge
#     merged = []
#     for i, c in enumerate(chunks):
#         if i == 0:
#             merged.append(c)
#         else:
#             prev = merged[-1]
#             if len(prev) + len(c) <= chunk_size + overlap:
#                 merged[-1] = prev + "\n\n" + c
#             else:
#                 merged.append(c)
#     return merged
# -------- Chunking --------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter.
    Tries larger separators first to preserve structure, then falls back.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # You can tune separators order; this works well for PDF text
        separators=[
            "\n\n",        # paragraphs
            "\n",          # lines
            ". ",          # sentences
            " ",           # words
            ""             # characters
        ],
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)



# -------- Ingestion Main --------
def ingest_pdf_to_faiss(pdf_path: str, outdir: str, use_ocr_threshold: float = 0.85):
    """
    Main ingestion flow:
    - Extract text blocks (PyMuPDF)
    - Extract tables (pdfplumber)
    - Extract images (PyMuPDF)
    - Run OCR if page contains very little extractable text
    - Chunk text + tables + OCR outputs
    - Embed chunks and store them in FAISS with metadata
    """
    ensure_dir(outdir)
    image_dir = os.path.join(outdir, "extracted_images")
    ensure_dir(image_dir)

    print("Loading embedding model:", EMBED_MODEL_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # 1) text blocks via PyMuPDF
    print("Extracting text blocks (PyMuPDF)...")
    text_blocks = extract_text_blocks_with_layout(pdf_path)

    # 2) tables via pdfplumber
    print("Extracting tables (pdfplumber)...")
    tables = extract_tables_pdfplumber(pdf_path)

    # 3) extract images (and save)
    print("Extracting images (PyMuPDF)...")
    pdf = fitz.open(pdf_path)
    all_images_meta = []
    for page_idx in range(len(pdf)):
        imgs = extract_images_from_page(pdf, page_idx, image_dir)
        all_images_meta.extend(imgs)
    pdf.close()

    # 4) decide OCR pages (low text density)
    print("Deciding whether OCR is required for pages...")
    # simple heuristic: if a page's extracted text length < threshold, run OCR on it
    pdf = fitz.open(pdf_path)
    pages_to_ocr = []
    for i in range(len(pdf)):
        txt = pdf[i].get_text("text").strip()
        # page area can be used for density but keep simple
        if len(txt) < 200:  # heuristics: tune as needed
            pages_to_ocr.append(i)
    pdf.close()
    print(f"Pages flagged for OCR: {pages_to_ocr[:30]} (count={len(pages_to_ocr)})")

    ocr_results = []
    if pages_to_ocr:
        print("Running OCR on flagged pages (this can take time)...")
        pil_pages = convert_from_path(pdf_path, dpi=OCR_DPI)
        for pidx in pages_to_ocr:
            pil = pil_pages[pidx]
            txt = ocr_page_image(pil, lang=OCR_LANG)
            if txt:
                ocr_results.append({"page": pidx, "type": "ocr", "text": txt})

    # 5) collect all textual sources to chunk
    print("Collecting textual sources for chunking...")
    sources = []
    for b in text_blocks:
        sources.append({
            "source_type": "text_block",
            "page": b["page"],
            "text": b["text"]
        })
    for t in tables:
        sources.append({
            "source_type": "table",
            "page": t["page"],
            "text": t["text"],
            "meta": {"nrows": t["nrows"], "ncols": t["ncols"]}
        })
    for o in ocr_results:
        sources.append({
            "source_type": "ocr",
            "page": o["page"],
            "text": o["text"]
        })

    # 6) chunk each source and attach metadata
    print("Chunking text...")
    chunk_items = []
    for s in sources:
        chunks = chunk_text(s["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, c in enumerate(chunks):
            chunk_id = uuid.uuid4().hex
            meta = {
                "id": chunk_id,
                "page": s.get("page"),
                "source_type": s.get("source_type"),
                "chunk_index": i,
                "source_meta": s.get("meta", {})
            }
            chunk_items.append({
                "id": chunk_id,
                "text": c,
                "meta": meta
            })

    print(f"Total chunks: {len(chunk_items)}")

    # 7) embed in batches
    print("Computing embeddings...")
    texts = [c["text"] for c in chunk_items]
    batch_size = 32
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        emb = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        vectors.append(emb)
    if vectors:
        vectors = np.vstack(vectors)
    else:
        vectors = np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)

    # 8) build FAISS
    print("Building FAISS index...")
    dim = vectors.shape[1] if vectors.size else EMBED_DIM
    store = FaissIndexStore(dim)
    ids = [c["id"] for c in chunk_items]
    if vectors.shape[0] > 0:
        store.add(vectors, ids)

    # 9) persist outputs
    print("Saving index and metadata...")
    prefix = os.path.join(outdir, "rag_index")
    store.save(prefix)
    # Save metadata
    metadata_path = os.path.join(outdir, "metadata.jsonl")
    meta_lines = []
    for c in chunk_items:
        meta_lines.append({
            "id": c["id"],
            "text": c["text"],
            "meta": c["meta"]
        })
    save_jsonl(meta_lines, metadata_path)

    # Save image metadata
    with open(os.path.join(outdir, "images.json"), "w", encoding="utf-8") as fh:
        json.dump(all_images_meta, fh, indent=2)

    # Optionally save raw embeddings
    emb_path = os.path.join(outdir, "embeddings.npy")
    np.save(emb_path, vectors)

    print("Ingestion finished. Outputs in:", outdir)
    return {
        "index_prefix": prefix,
        "metadata": metadata_path,
        "embeddings": emb_path,
        "images": os.path.join(outdir, "images.json")
    }



def iterate_pdfs_in_directory(dir_path: str) -> List[str]:
    """Return list of PDF file paths recursively found in a directory."""

    pdf_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

# -------- NEW: Combined ingestion for many PDFs into one index --------
def ingest_pdfs_to_faiss_combined(inputs: List[str], outdir: str):
    """
    Ingest multiple PDFs and produce a single FAISS index and single metadata.jsonl.
    - Preserves file path and page in each chunk's metadata.
    - Stores images per-PDF under outdir/extracted_images/<pdf_stem>/.
    """
    ensure_dir(outdir)
    print("Loading embedding model:", EMBED_MODEL_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    combined_chunks: List[Dict] = []
    all_vectors_list: List[np.ndarray] = []
    all_ids: List[str] = []
    all_images_meta: List[Dict] = []

    for pdf_path in inputs:
        print(f"Processing: {pdf_path}")
        pdf_stem = Path(pdf_path).stem
        image_dir = os.path.join(outdir, "extracted_images", pdf_stem)
        ensure_dir(image_dir)

        # Text, tables
        text_blocks = extract_text_blocks_with_layout(pdf_path)
        tables = extract_tables_pdfplumber(pdf_path)

        # Images
        pdf_doc = fitz.open(pdf_path)
        for page_idx in range(len(pdf_doc)):
            imgs = extract_images_from_page(pdf_doc, page_idx, image_dir)
            for m in imgs:
                m["file"] = pdf_path
            all_images_meta.extend(imgs)
        pdf_doc.close()

        # OCR heuristic
        pdf_doc = fitz.open(pdf_path)
        pages_to_ocr = []
        for i in range(len(pdf_doc)):
            txt = pdf_doc[i].get_text("text").strip()
            if len(txt) < 200:
                pages_to_ocr.append(i)
        pdf_doc.close()

        ocr_results = []
        if pages_to_ocr:
            pil_pages = convert_from_path(pdf_path, dpi=OCR_DPI)
            for pidx in pages_to_ocr:
                pil = pil_pages[pidx]
                txt = ocr_page_image(pil, lang=OCR_LANG)
                if txt:
                    ocr_results.append({"page": pidx, "type": "ocr", "text": txt})

        # Collect sources
        sources = []
        for b in text_blocks:
            sources.append({"source_type": "text_block", "page": b["page"], "text": b["text"], "file": pdf_path})
        for t in tables:
            sources.append({
                "source_type": "table",
                "page": t["page"],
                "text": t["text"],
                "meta": {"nrows": t["nrows"], "ncols": t["ncols"]},
                "file": pdf_path
            })
        for o in ocr_results:
            sources.append({"source_type": "ocr", "page": o["page"], "text": o["text"], "file": pdf_path})

        # Chunk and embed per-PDF to keep memory bounded
        chunk_items = []
        for s in sources:
            chunks = chunk_text(s["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for i, c in enumerate(chunks):
                cid = uuid.uuid4().hex
                meta = {
                    "id": cid,
                    "page": s.get("page"),
                    "source_type": s.get("source_type"),
                    "chunk_index": i,
                    "source_meta": s.get("meta", {}),
                    "file": s.get("file"),
                }
                chunk_items.append({"id": cid, "text": c, "meta": meta})

        texts = [c["text"] for c in chunk_items]
        if texts:
            # batch embed
            batch_size = 32
            pdf_vectors = []
            for i in range(0, len(texts), batch_size):
                emb = embedder.encode(texts[i:i+batch_size], convert_to_numpy=True, show_progress_bar=False)
                pdf_vectors.append(emb)
            pdf_vectors = np.vstack(pdf_vectors)
            all_vectors_list.append(pdf_vectors)
            all_ids.extend([c["id"] for c in chunk_items])
            combined_chunks.extend(chunk_items)

    # Build single FAISS index
    if all_vectors_list:
        all_vectors = np.vstack(all_vectors_list).astype(np.float32)
    else:
        all_vectors = np.zeros((0, EMBED_DIM), dtype=np.float32)

    print(f"Total combined chunks: {len(combined_chunks)}")
    print("Building combined FAISS index...")
    dim = all_vectors.shape[1] if all_vectors.size else EMBED_DIM
    store = FaissIndexStore(dim)
    if all_vectors.shape[0] > 0:
        store.add(all_vectors, all_ids)

    # Save combined artifacts
    prefix = os.path.join(outdir, "rag_index")
    store.save(prefix)

    metadata_path = os.path.join(outdir, "metadata.jsonl")
    save_jsonl([{"id": c["id"], "text": c["text"], "meta": c["meta"]} for c in combined_chunks], metadata_path)

    with open(os.path.join(outdir, "images.json"), "w", encoding="utf-8") as fh:
        json.dump(all_images_meta, fh, indent=2)

    emb_path = os.path.join(outdir, "embeddings.npy")
    np.save(emb_path, all_vectors)

    print("Combined ingestion finished. Outputs in:", outdir)
    return {"index_prefix": prefix, "metadata": metadata_path, "embeddings": emb_path, "images": os.path.join(outdir, "images.json")}

# -------- Command-line interface --------
def main():
    parser = argparse.ArgumentParser(description="RAG ingestion for multimedia PDF")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file or directory")
    parser.add_argument("--output_dir", type=str, default="./output_rag", help="Output directory")
    parser.add_argument("--combine", action="store_true", help="Combine multiple PDFs into one FAISS index")
    args = parser.parse_args()

    if os.path.isdir(args.pdf_path):
        pdfs = iterate_pdfs_in_directory(args.pdf_path)
        if args.combine:
            print(f"Combining {len(pdfs)} PDFs into one index -> {args.output_dir}")
            ingest_pdfs_to_faiss_combined(pdfs, args.output_dir)
        else:
            for pdf_file in pdfs:
                out_dir = os.path.join(args.output_dir, Path(pdf_file).stem)
                print(f"Processing {pdf_file} -> {out_dir}")
                ingest_pdf_to_faiss(pdf_file, out_dir)
    elif os.path.isfile(args.pdf_path) and args.pdf_path.lower().endswith(".pdf"):
        if args.combine:
            # Combining single file still works and produces a single index
            ingest_pdfs_to_faiss_combined([args.pdf_path], args.output_dir)
        else:
            ingest_pdf_to_faiss(args.pdf_path, args.output_dir)

if __name__ == "__main__":
    main()