# syntax=docker/dockerfile:1
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Minimal system deps; libmagic for python-magic, libgomp for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the repo
COPY . /app

# Streamlit runs on 8501 by default
EXPOSE 8501

# Default: run the OpenAI RAG app; override command to run rag_streamlit.py
CMD ["bash", "-lc", "streamlit run src/oai_streamlit.py --server.address=0.0.0.0 --server.port=8501"]