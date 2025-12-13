# RAGAnalyst

RAG specializing in the analysis of Annual Reports within the automotive sector

Setup

```bash
pip install pymupdf pdfplumber pdf2image pytesseract sentence-transformers faiss-cpu python-magic
```

You also need Tesseract OCR installed on your machine (for Linux/macOS via package manager or Windows installer).

Example:

macOS (Homebrew): ```brew install tesseract```

Ubuntu: ```sudo apt install tesseract-ocr```


Build (on macOS; add --platform if on Apple Silicon to get faiss-cpu wheels):

```bash
docker build -t raganalyst:latest -f dockerfile .
```

Apple Silicon note: ```docker build --platform=linux/amd64 -t raganalyst:latest -f dockerfile .```

Run the OpenAI app (oai_streamlit.py):

```bash
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=sk-... -v "$(pwd)/data:/app/data" raganalyst:latest bash -lc "streamlit run src/oai_streamlit.py port=8501"
```

Run the local HF app (rag_streamlit.py):

```bash
docker run --rm -p 8501:8501   -e HUGGINGFACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXX -v "$(pwd)/.cache:/root/.cache/huggingface" -v "$(pwd)/data:/app/data" raganalyst:latest bash -lc "streamlit run src/rag_streamlit.py port=8501"
```

Notes:

Mount your data directory so FAISS/metadata paths like ./data/vector_oai/... work inside the container at /app/data.
For the HF app, the model will download on first run. To persist the cache, mount a cache volume:
-v "$(pwd)/.cache:/root/.cache/huggingface"
If the HF model is large and you’re on Apple Silicon, keep --platform=linux/amd64 for build/run to ensure compatible wheels for faiss-cpu.
