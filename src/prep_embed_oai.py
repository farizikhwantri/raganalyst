"""
Multi-file RAG ingestion using LangChain + GPT-4o multimodal extraction.

Supports:
- PDFs
- Images (PNG/JPG)
- Text files
- Word documents

Pipeline:
1. Load files
2. GPT-4o extracts structured text from each file
3. LangChain chunks text
4. Embeddings (HuggingFace or OpenAI)
5. Save FAISS index + metadata

Run:
    python multi_ingest.py ./documents --outdir ./vectorstore
"""

import os
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any

from langchain_classic.docstore.document import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from openai import OpenAI
client = OpenAI()

# ---------------------------
# Config
# ---------------------------

GPT_MODEL = "gpt-4o"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

SUPPORTED_EXT = [".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"]


# ---------------------------
# Step 1: GPT-4o Universal Extractor
# ---------------------------

def extract_text_gpt(file_bytes: bytes, filename: str) -> str:
    """
    Sends any file (PDF, image, doc) to GPT-4o for multimodal text extraction.
    """

    resp = client.responses.create(
        model=GPT_MODEL,
        input=[{
            "type": "input_text",
            "text": f"Extract all textual content from this file ({filename}). "
                    f"If it contains tables or charts, describe them clearly."
        }],
        attachments=[{
            "filename": filename,
            "data": file_bytes
        }]
    )

    return resp.output_text.strip()


# ---------------------------
# Step 2: Load and extract from multiple files
# ---------------------------

def load_and_extract(folder: str) -> List[Dict[str, Any]]:
    results = []

    for root, dirs, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() not in SUPPORTED_EXT:
                continue

            fpath = Path(root) / fname
            print(f"[Extracting] {fpath}")

            with open(fpath, "rb") as f:
                data = f.read()

            text = extract_text_gpt(data, fname)

            results.append({
                "filename": fname,
                "text": text
            })

    return results


# ---------------------------
# Step 3: Chunking
# ---------------------------

def chunk_documents(extracted_docs: List[Dict]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for doc in extracted_docs:
        for chunk in splitter.split_text(doc["text"]):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": doc["filename"],
                        "chunk_id": uuid.uuid4().hex,
                    }
                )
            )
    return chunks


# ---------------------------
# Step 4: Build FAISS index
# ---------------------------

def build_faiss(chunks: List[Document], outdir: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[Building FAISS index]")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    os.makedirs(outdir, exist_ok=True)
    vectorstore.save_local(outdir)

    print(f"[Saved] Vectorstore saved to {outdir}")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder containing documents to ingest")
    parser.add_argument("--outdir", type=str, default="./vectorstore")
    args = parser.parse_args()

    extracted = load_and_extract(args.folder)
    print(f"[Extracted] {len(extracted)} documents processed")

    chunks = chunk_documents(extracted)
    print(f"[Chunked] {len(chunks)} chunks")

    build_faiss(chunks, args.outdir)
    print("[Done] Multi-file ingestion complete")


if __name__ == "__main__":
    main()
