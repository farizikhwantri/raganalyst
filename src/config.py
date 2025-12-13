# -------- Configuration --------
CHUNK_SIZE = 800     # characters per chunk (tune for your use)
CHUNK_OVERLAP = 150  # overlap between chunks
OCR_DPI = 300        # DPI for pdf2image OCR conversion
OCR_LANG = "eng"     # change to 'deu' or others as needed

# Embedder model (HF)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # for all-MiniLM-L6-v2

# Generator LLM (HF)
GEN_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_NEW_TOKENS = 256


# faiss_vector index path
FAISS_INDEX_PATH = "./data/vector/rag_index"
METADATA_PATH = "./data/vector/metadata.jsonl"

# OpenAI config (if using OpenAI embeddings or generation)
# ---------------- Config ----------------
OAI_INDEX_PREFIX = "./data/vector_oai/rag_index"
OAI_METADATA_PATH = "./data/vector_oai/metadata.jsonl"

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-large"  # default; overridden by manifest