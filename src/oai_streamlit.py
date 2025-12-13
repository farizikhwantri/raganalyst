import os
import json
import streamlit as st
import numpy as np

import re
from pathlib import Path

from openai import OpenAI

from faiss_module import FaissIndexStore

# ---------------- Config ----------------
DEFAULT_INDEX_PREFIX = "./data/vector_oai/rag_index"
DEFAULT_METADATA_PATH = "./data/vector_oai/metadata.jsonl"

# OpenAI generation + embeddings
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-large"  # 3072-dim

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Helpers ----------------
@st.cache_data
def load_manifest(index_prefix: str):
    """index_prefix -> dirname -> manifest.json"""
    outdir = os.path.dirname(index_prefix)
    mpath = os.path.join(outdir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"embedding_provider": "openai", 
            "embedding_model": OPENAI_EMBED_MODEL, "embedding_dim": 3072}

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


def rewrite_question_oai(client: OpenAI, history: list, question: str) -> str:
    hist_text = "\n".join([f"{r}: {t}" for r, t in history][-8:])
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content":
             "You are a question rewriter. Rewrite the user's follow-up question into a standalone query "
             "that can be understood without conversation history."},
            {"role": "user", "content":
             f"History:\n{hist_text}\n\nFollow-up question: {question}\n\nRewritten standalone question:"},
        ],
        max_output_tokens=96,
    )
    text = getattr(resp, "output_text", "").strip()
    if not text and getattr(resp, "output", None):
        text = "".join([c.text for c in resp.output[0].content if hasattr(c, "text")]).strip()
    rewritten = text.split("Rewritten standalone question:")[-1].strip() if "Rewritten standalone question:" in text else text
    return rewritten or question

def _extract_years(text: str):
    return re.findall(r"(20\d{2})", text)

def _company_from_path(p: str) -> str:
    if not p:
        return "unknown"
    parts = Path(p).parts
    for seg in reversed(parts):
        m = re.match(r"([A-Za-z0-9\-]+)_Annual_Report(_)?(\d{4})?", seg)
        if m:
            return m.group(1)
    stem = Path(p).stem
    return stem.split("_")[0] if "_" in stem else stem

def discover_companies(meta: dict) -> list:
    seen = set()
    for item in meta.values():
        comp = _company_from_path((item.get("meta") or {}).get("file"))
        if comp:
            seen.add(comp)
    return sorted(seen)

def expand_query_oai(client: OpenAI, question: str, history: list, companies: list) -> list:
    years = _extract_years(question)
    year_hint = years[0] if years else ""
    sys_msg = (
        "You expand search queries for annual report retrieval. "
        "Output 6–10 short queries (one per line) covering company names and profitability synonyms "
        "(profitability, operating margin, EBIT, EBITDA, net income, profit margin, return on sales, operating profit). "
        "Prefer adding the target year if present. No explanations—only queries."
    )
    usr_msg = f"Question: {question}\nCompanies: {', '.join(companies[:8])}\nYear: {year_hint}"
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": usr_msg},
        ],
        max_output_tokens=160,
    )
    text = getattr(resp, "output_text", "").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    synonyms = ["profitability", "operating margin", "EBIT", "EBITDA", "net income", "profit margin", "return on sales", "operating profit"]
    det = []
    for comp in companies:
        for syn in synonyms[:4]:
            if year_hint:
                det.append(f"{comp} {year_hint} {syn}")
            det.append(f"{comp} {syn}")

    queries = [question.strip()] + lines + det

    seen = set()
    final_q = []
    for q in queries:
        qn = q.strip()
        if not qn or len(qn) > 128:
            continue
        k = qn.lower()
        if k not in seen:
            seen.add(k)
            final_q.append(qn)
        if len(final_q) >= 10:
            break
    return final_q

def retrieve_with_expansion_oai(client: OpenAI, query: str, store: FaissIndexStore, meta: dict, top_k: int = 6):
    companies = discover_companies(meta)
    expanded = expand_query_oai(client, query, st.session_state.history, companies)

    candidates = {}
    for q in expanded:
        qv = embed_query_openai(client, q)
        hits = store.search(qv, k=max(6, top_k))
        for hid, dist in hits:
            item = meta.get(hid)
            if not item:
                continue
            prev = candidates.get(hid)
            if not prev or dist < prev["score"]:
                comp = _company_from_path((item.get("meta") or {}).get("file"))
                candidates[hid] = {
                    "id": hid,
                    "text": item["text"],
                    "meta": item["meta"],
                    "score": float(dist),
                    "company": comp,
                }

    years = _extract_years(query)
    if years:
        sy = years[0]
        candidates = {hid: c for hid, c in candidates.items() if sy in str(c["meta"].get("file", ""))}

    by_company = {}
    for c in candidates.values():
        by_company.setdefault(c["company"], []).append(c)
    for comp in by_company:
        by_company[comp].sort(key=lambda x: x["score"])

    per_cap = max(1, top_k // max(1, len(by_company)))
    selected = []
    round_idx = 0
    while len(selected) < top_k:
        progressed = False
        for comp, items in by_company.items():
            take_idx = round_idx * per_cap
            if take_idx < len(items):
                selected.append(items[take_idx])
                progressed = True
                if len(selected) >= top_k:
                    break
        if not progressed:
            break
        round_idx += 1

    if len(selected) < top_k:
        remaining = sorted(candidates.values(), key=lambda x: x["score"])
        used = {c["id"] for c in selected}
        for c in remaining:
            if c["id"] in used:
                continue
            selected.append(c)
            if len(selected) >= top_k:
                break

    docs = [c["text"] for c in selected]
    sources = [{
        "id": c["id"],
        "file": c["meta"].get("file"),
        "page": c["meta"].get("page"),
        "source_type": c["meta"].get("source_type"),
        "company": c["company"],
        "score": c["score"],
    } for c in selected]
    return docs, sources

def embed_query_openai(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[query])
    vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    return vec

def retrieve_with_openai_embed(query: str, store: FaissIndexStore, client: OpenAI, meta: dict, k: int = 5):
    qvec = embed_query_openai(client, query)
    hits = store.search(qvec, k=k)
    docs, sources = [], []
    for hid, dist in hits:
        item = meta.get(hid)
        if not item:
            continue
        docs.append(item["text"])
        m = item.get("meta", {})
        sources.append({
            "id": hid,
            "file": m.get("file"),
            "page": m.get("page"),
            "source_type": m.get("source_type"),
            "score": dist,
        })
    return docs, sources

def generate_answer_openai(client: OpenAI, query: str, retrieved_docs: list) -> str:
    context = "\n\n".join(retrieved_docs) if retrieved_docs else "No context retrieved."
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content":
              "You are a helpful analyst specializing in the analysis of Annual Reports within the automotive sector. "
              "Your daily tasks involve extracting key financial metrics such as revenue, EBITDA, growth numbers, and conducting comparative analyses across different car companies, as well as comparing them to other sectors. "
              "Use the provided context to answer the user's question. "
              "If the answer cannot be found in the context, say you don't know. "
              "Cite page numbers when possible."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
        ],
        max_output_tokens=300,
    )
    # try:
    return resp.output_text.strip()
    # except Exception:
    #     return "I don't know."

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Analyst (OpenAI Embeddings + FAISS)", layout="wide")
st.title("RAG Analyst (OpenAI Embeddings + FAISS)")

st.sidebar.header("Index configuration")
index_prefix = st.sidebar.text_input("FAISS index prefix", DEFAULT_INDEX_PREFIX)
metadata_path = st.sidebar.text_input("Metadata JSONL path", DEFAULT_METADATA_PATH)
embed_dim = st.sidebar.selectbox("Embedding dimension", [3072, 1536], index=0, help="Use 3072 for text-embedding-3-large, 1536 for text-embedding-3-small.")
top_k = st.sidebar.slider("Top-K", 1, 10, 5, 1)

client = load_openai_client()

# Read manifest and set embedding model/dim automatically
manifest = load_manifest(index_prefix)
OPENAI_EMBED_MODEL = manifest.get("embedding_model", OPENAI_EMBED_MODEL)
embed_dim = int(manifest.get("embedding_dim", 3072))

st.sidebar.write(f"Embedding model: {OPENAI_EMBED_MODEL}")
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

def embed_query_openai(client: OpenAI, query: str) -> np.ndarray:
    # Use the same model declared in the manifest
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[query])
    vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    return vec

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
        standalone_q = rewrite_question_oai(client, st.session_state.history, user_q.strip())

    with st.spinner("Retrieving relevant chunks with query expansion..."):
        docs, sources = retrieve_with_expansion_oai(client, standalone_q, store, meta, top_k=top_k)

    with st.spinner("Generating answer..."):
        answer = generate_answer_openai(client, standalone_q, docs)

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
