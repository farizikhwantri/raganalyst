import os
import json
import streamlit as st
from pathlib import Path

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from faiss_module import FaissIndexStore

from rag_module import HybridProvider, rewrite_question, retrieve_with_expansion, generate_answer

# ---------------- Config ----------------
HYBRID_INDEX_PREFIX = "./data/vector/rag_index"  # FAISS built with HF embeddings
HYBRID_METADATA_PATH = "./data/vector/metadata.jsonl"

OPENAI_MODEL = "gpt-4o-mini"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Helpers ----------------
@st.cache_data
def load_manifest(index_prefix: str):
    # Optional: if you saved a manifest with HF embedder info, load it
    outdir = os.path.dirname(index_prefix)
    mpath = os.path.join(outdir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"embedding_provider": "hf", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dim": 384}

@st.cache_resource
def load_openai_client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or put in .env.")
    return OpenAI(api_key=key)

@st.cache_resource
def load_hf_embedder(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_data
def load_index(prefix_path: str, emb_dim: int):
    return FaissIndexStore.load(prefix_path, dim=emb_dim)

@st.cache_data
def load_metadata(metadata_path: str):
    meta = {}
    with open(metadata_path, "r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            meta[item["id"]] = item
    return meta

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Analyst (Hybrid: HF embed + OpenAI gen)", layout="wide")
st.title("RAG Analyst (Hybrid: HF embeddings + OpenAI generation)")

st.sidebar.header("Index configuration")
index_prefix = st.sidebar.text_input("FAISS index prefix (HF-built)", HYBRID_INDEX_PREFIX)
metadata_path = st.sidebar.text_input("Metadata JSONL path", HYBRID_METADATA_PATH)
top_k = st.sidebar.slider("Top-K", 1, 10, 5, 1)

# Sidebar: choose HF embedder (must match index’s dimension)
embedder_name = st.sidebar.text_input("HF embedder", "sentence-transformers/all-MiniLM-L6-v2")

client = load_openai_client()
manifest = load_manifest(index_prefix)

# Load HF embedder and ensure index dim matches
hf_embedder = load_hf_embedder(embedder_name)
embed_dim = hf_embedder.get_sentence_embedding_dimension()
st.sidebar.write(f"HF embedder: {embedder_name}")
st.sidebar.write(f"Embedding dim: {embed_dim}")

# Load index/metadata
try:
    store = load_index(index_prefix, emb_dim=embed_dim)
except Exception as e:
    st.error(f"Failed to load FAISS index: {e}")
    st.stop()

try:
    meta = load_metadata(metadata_path)
except Exception as e:
    st.error(f"Failed to load metadata: {e}")
    st.stop()

# Create Hybrid provider: HF embeddings + OpenAI generation
provider = HybridProvider(oai_client=client, hf_embedder=hf_embedder, gen_model=OPENAI_MODEL)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_q" not in st.session_state:
    st.session_state.last_q = None

with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("Ask a question", "", key="user_q_input")
    submitted = st.form_submit_button("Send")

if submitted and user_q.strip():
    if st.session_state.last_q == user_q.strip():
        st.stop()

    with st.spinner("Rewriting follow-up question..."):
        standalone_q = rewrite_question(provider, st.session_state.history, user_q.strip())

    with st.spinner("Retrieving relevant chunks with query expansion..."):
        docs, sources = retrieve_with_expansion(provider, standalone_q, store, meta, top_k=top_k, history=st.session_state.history)

    with st.spinner("Generating answer..."):
        answer = generate_answer(provider, standalone_q, docs)

    st.session_state.history.append(("user", user_q.strip()))
    st.session_state.history.append(("assistant", answer))
    st.session_state.last_q = user_q.strip()

    st.subheader("User Question")
    st.write(user_q.strip())
    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Standalone query: {standalone_q}")

    with st.expander("Retrieved sources"):
        for s in sources:
            st.write(f"- id={s['id']} | file={s.get('file')} | page={s.get('page')} | type={s.get('source_type')} | company={s.get('company')} | score={s['score']:.4f}")

# History + actions
st.subheader("History")
for role, msg in st.session_state.history[-10:]:
    st.markdown(f"**{role}**: {msg}")

cols = st.columns(2)
if cols[0].button("Clear history"):
    st.session_state.history = []
    st.rerun()

if cols[1].button("Reload index"):
    store = load_index(index_prefix, emb_dim=embed_dim)
    meta = load_metadata(metadata_path)
    st.success("Index reloaded.")