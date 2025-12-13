# syntax=docker/dockerfile:1
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System deps: libmagic for python-magic, libgomp for faiss-cpu, tesseract + poppler (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 libgomp1 tesseract-ocr poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (avoids CUDA wheels)
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      torch torchvision torchaudio

# Then install the rest
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy app
COPY . /app

EXPOSE 8501

# Default to OpenAI Streamlit app; override CMD to run HF app
CMD ["bash", "-lc", "streamlit run src/oai_streamlit.py --server.address=0.0.0.0 --server.port=8501"]