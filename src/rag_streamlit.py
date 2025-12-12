import json
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from faiss_module import FaissIndexStore
from config import EMBED_MODEL_NAME, GEN_MODEL_NAME

# ---- Cache model loaders ----
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, gen_model

@st.cache_data
def load_index(prefix_path: str, emb_dim: int):
    store = FaissIndexStore.load(prefix_path, dim=emb_dim)
    return store

@st.cache_data
def load_metadata(metadata_path: str):
    meta = {}
    with open(metadata_path, "r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            meta[item["id"]] = item
    return meta

# ---- RAG components ----
def rewrite_question(tokenizer, gen_model, history: list, question: str) -> str:
    hist_text = "\n".join([f"{r}: {t}" for r, t in history][-8:])
    prompt = (
        "You are a question rewriter. Rewrite the user's follow-up question into a "
        "standalone query that can be understood without conversation history.\n\n"
        f"History:\n{hist_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Rewritten standalone question:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    output = gen_model.generate(**inputs, max_new_tokens=96, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    rewritten = text.split("Rewritten standalone question:")[-1].strip()
    return rewritten or question

def retrieve(query: str, store: FaissIndexStore, embedder, meta: dict, k: int = 5):
    qvec = embedder.encode([query], convert_to_numpy=True)
    hits = store.search(qvec, k=k)
    docs = []
    sources = []
    for hid, dist in hits:
        item = meta.get(hid)
        if not item:
            continue
        docs.append(item["text"])
        src = item.get("meta", {})
        sources.append({
            "id": hid,
            "file": src.get("file"),
            "page": src.get("page"),
            "source_type": src.get("source_type"),
            "score": dist,
        })
    return docs, sources

def generate_answer(tokenizer, gen_model, query: str, retrieved_docs: list) -> str:
    context = "\n\n".join(retrieved_docs)
    prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "If the answer cannot be found in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    output = gen_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = text.split("Answer:")[-1].strip()
    return answer

# ---- Streamlit UI ----
st.set_page_config(page_title="Follow-up-aware RAG", layout="wide")
st.title("Follow-up-aware RAG (HF + FAISS)")

# Sidebar: index paths
st.sidebar.header("Index configuration")
index_prefix = st.sidebar.text_input("FAISS index prefix", "./data/vector/rag_index")
metadata_path = st.sidebar.text_input("Metadata JSONL path", "./data/vector/metadata.jsonl")
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=6)

# Load models
embedder = load_embedder()
tokenizer, gen_model = load_generator()

# Load index + metadata when paths change
if "store" not in st.session_state or st.session_state.get("index_prefix") != index_prefix:
    try:
        st.session_state.store = load_index(index_prefix, emb_dim=embedder.get_sentence_embedding_dimension())
        st.session_state.index_prefix = index_prefix
    except Exception as e:
        st.warning(f"Failed to load FAISS index from {index_prefix}: {e}")
        st.stop()

if "meta" not in st.session_state or st.session_state.get("metadata_path") != metadata_path:
    try:
        st.session_state.meta = load_metadata(metadata_path)
        st.session_state.metadata_path = metadata_path
    except Exception as e:
        st.warning(f"Failed to load metadata from {metadata_path}: {e}")
        st.stop()

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("Ask a question", "")
    submitted = st.form_submit_button("Send")

if submitted and user_q.strip():
    # RAG pipeline
    standalone_q = rewrite_question(tokenizer, gen_model, st.session_state.history, user_q.strip())
    docs, sources = retrieve(standalone_q, st.session_state.store, embedder, st.session_state.meta, k=top_k)
    answer = generate_answer(tokenizer, gen_model, standalone_q, docs)

    # Update history
    st.session_state.history.append(("user", user_q.strip()))
    st.session_state.history.append(("assistant", answer))

    # Show results
    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Standalone query: {standalone_q}")

    with st.expander("Retrieved sources"):
        for s in sources:
            st.write(f"- id={s['id']} | file={s.get('file')} | page={s.get('page')} | type={s.get('source_type')} | score={s['score']:.4f}")

# Conversation history
st.subheader("History")
for role, msg in st.session_state.history[-10:]:
    st.markdown(f"**{role}**: {msg}")

# Footer actions
cols = st.columns(2)
if cols[0].button("Clear history"):
    st.session_state.history = []
    st.experimental_rerun()
if cols[1].button("Reload index"):
    # Re-load cached resources
    st.session_state.store = load_index(index_prefix, emb_dim=embedder.get_sentence_embedding_dimension())
    st.session_state.meta = load_metadata(metadata_path)
    st.success("Index reloaded.")

