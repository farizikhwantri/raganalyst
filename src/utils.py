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

from config import CHUNK_SIZE, CHUNK_OVERLAP, OCR_DPI, OCR_LANG


# -------- Utilities --------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_jsonl(lines: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as fh:
        for item in lines:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

# -------- Extraction Functions --------
def extract_images_from_page(pdf: fitz.Document, page_num: int, image_output_dir: str) -> List[Dict]:
    """Extract images on a page and save them to disk. Return metadata list."""
    page = pdf[page_num]
    images = page.get_images(full=True)
    saved = []
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = pdf.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image.get("ext", "png")
        img_id = f"p{page_num}_img{img_index}_{uuid.uuid4().hex[:8]}.{ext}"
        out_path = os.path.join(image_output_dir, img_id)
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        saved.append({
            "page": page_num,
            "img_index": img_index,
            "path": out_path,
            "width": base_image.get("width"),
            "height": base_image.get("height"),
            "ext": ext
        })
    return saved

def extract_text_blocks_with_layout(pdf_path: str) -> List[Dict]:
    """Extract text blocks with page numbers and basic bbox info using PyMuPDF."""
    doc = fitz.open(pdf_path)
    blocks = []
    for i in range(len(doc)):
        page = doc[i]
        # text as plain (keeps newlines) and blocks for layout
        text = page.get_text("text").strip()
        if text:
            blocks.append({"page": i, "type": "text", "text": text})
        # Also capture "blocks" with bbox if needed:
        # blocks_with_bbox = page.get_text("dict")['blocks'] ...
    doc.close()
    return blocks

def extract_tables_pdfplumber(pdf_path: str) -> List[Dict]:
    """Use pdfplumber to extract tables (returns CSV-like lists)."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for t_idx, table in enumerate(page_tables):
                # Normalize to CSV-like string or keep as list
                csv_lines = []
                for row in table:
                    row = [("" if c is None else str(c)).strip() for c in row]
                    csv_lines.append("|".join(row))
                content = "\n".join(csv_lines).strip()
                if content:
                    tables.append({
                        "page": i,
                        "table_index": t_idx,
                        "text": content,
                        "nrows": len(table),
                        "ncols": max(len(r) for r in table) if table else 0
                    })
    return tables

def ocr_page_image(page_image: Image.Image, lang: str = OCR_LANG) -> str:
    """Run Tesseract OCR on a PIL Image and return text."""
    text = pytesseract.image_to_string(page_image, lang=lang)
    return text.strip()

def extract_ocr_for_scanned_pages(pdf_path: str, pages: Optional[List[int]] = None) -> List[Dict]:
    """
    Convert selected pages to images and run OCR.
    If pages=None, process all pages.
    """
    pil_pages = convert_from_path(pdf_path, dpi=OCR_DPI)
    ocr_results = []
    for i, pil in enumerate(pil_pages):
        if pages is not None and i not in pages:
            continue
        text = ocr_page_image(pil, lang=OCR_LANG)
        if text:
            ocr_results.append({"page": i, "type": "ocr", "text": text})
    return ocr_results

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