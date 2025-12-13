import json
import os
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from faiss_module import FaissIndexStore
from config import EMBED_MODEL_NAME, GEN_MODEL_NAME
from config import FAISS_INDEX_PATH, METADATA_PATH

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- Cache model loaders ----
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
def rewrite_question(tokenizer, gen_model, 
                     history: list, 
                     question: str) -> str:
    hist_text = "\n".join([f"{r}: {t}" for r, t in history][-8:])
    # prompt = (
    #     "You are a question rewriter. Rewrite the user's follow-up question into a "
    #     "standalone query that can be understood without conversation history.\n\n"
    #     f"History:\n{hist_text}\n\n"
    #     f"Follow-up question: {question}\n\n"
    #     "Rewritten standalone question:"
    # )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a question rewriter. Rewrite the user's follow-up question into a "
                "standalone query that can be understood without conversation history."
            ),
        },
        {
            "role": "user",
            "content": (
                f"History:\n{hist_text}\n\n"
                f"Follow-up question: {question}\n\n"
                "Rewritten standalone question:"
            ),
        },
    ]
    # inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = inputs.to(gen_model.device)
        # output = gen_model.generate(**inputs, max_new_tokens=96, do_sample=False)
    output = gen_model.generate(**inputs, max_new_tokens=96, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    rewritten = text.split("Rewritten standalone question:")[-1].strip()
    return rewritten or question

def retrieve(query: str, store: FaissIndexStore, 
             embedder, meta: dict, k: int = 5):
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


def _extract_years(text: str):
    return re.findall(r"(20\d{2})", text)

def _company_from_path(p: str) -> str:
    """
    Try to extract a company name from paths like:
      data/vector/Ford_Annual_Report_2022/...
      .../BMW_Annual_Report_2023/...
    """
    if not p:
        return "unknown"
    parts = Path(p).parts
    for seg in reversed(parts):
        m = re.match(r"([A-Za-z0-9\-]+)_Annual_Report(_)?(\d{4})?", seg)
        if m:
            return m.group(1)
    # fallback: use first token from stem
    stem = Path(p).stem
    return stem.split("_")[0] if "_" in stem else stem

def discover_companies(meta: dict) -> list:
    seen = set()
    for item in meta.values():
        comp = _company_from_path((item.get("meta") or {}).get("file"))
        if comp:
            seen.add(comp)
    return sorted(seen)

def expand_query(tokenizer, gen_model, question: str, history: list, companies: list) -> list:
    """
    LLM-driven query expansion plus deterministic expansions over companies and profitability synonyms.
    Returns up to 10 short queries.
    """
    year_hints = _extract_years(question)
    year_hint = year_hints[0] if year_hints else ""

    sys_msg = (
        "You expand search queries for annual report retrieval. "
        "Output 6–10 short queries (one per line) covering company names and profitability synonyms. "
        "Prefer adding the target year if present. No explanations—only queries."
    )
    usr_msg = f"Question: {question}\nCompanies: {', '.join(companies[:8])}\nYear: {year_hint}"

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": usr_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(gen_model.device)
    out = gen_model.generate(**inputs, max_new_tokens=160, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    llm_lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Deterministic expansions
    synonyms = [
        "profitability", "operating margin", "EBIT", "EBITDA",
        "net income", "profit margin", "return on sales", "operating profit"
    ]
    det = []
    yrs = [year_hint] if year_hint else []
    for comp in companies:
        for syn in synonyms[:4]:  # keep it concise
            if yrs:
                det.append(f"{comp} {yrs[0]} {syn}")
            det.append(f"{comp} {syn}")

    queries = [question.strip()]
    queries.extend(llm_lines)
    queries.extend(det)

    # Clean + dedupe + keep short
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

def retrieve_with_expansion(
    tokenizer, gen_model, query: str, store: FaissIndexStore, embedder, meta: dict, top_k: int = 6
):
    """
    Run expanded queries, gather candidates, then select a diverse top-k across companies.
    """
    companies = discover_companies(meta)
    expanded = expand_query(tokenizer, gen_model, query, st.session_state.history, companies)

    # Collect candidates across all expansions, keep best distance per id
    candidates = {}
    for q in expanded:
        qv = embedder.encode([q], convert_to_numpy=True)
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

    # Year-aware filter (if question mentions a year)
    years = _extract_years(query)
    if years:
        sy = years[0]
        candidates = {
            hid: c for hid, c in candidates.items()
            if sy in str(c["meta"].get("file", ""))
        }

    # Enforce company diversity: pick at most ceil(top_k / #companies) per company in score order
    by_company = {}
    for c in candidates.values():
        by_company.setdefault(c["company"], []).append(c)
    for comp in by_company:
        by_company[comp].sort(key=lambda x: x["score"])  # lower distance first

    per_cap = max(1, top_k // max(1, len(by_company)))
    selected = []
    # Round-robin across companies
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

    # If not enough, fill with remaining best regardless of company
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

def generate_answer(tokenizer, gen_model, query: str, retrieved_docs: list) -> str:
    context = "\n\n".join(retrieved_docs)

    # Build chat messages for HF chat template
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful analyst specializing in the analysis of Annual Reports within the automotive sector. "
                "Your daily tasks involve extracting key financial metrics such as revenue, EBITDA, growth numbers, and conducting comparative analyses across different car companies, as well as comparing them to other sectors. "
                "Use the provided context to answer the user's question. "
                "If the answer cannot be found in the context, say you don't know. "
                "Cite page numbers when possible."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        },
    ]

    # Render prompt using the model's chat template
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, 
                       return_tensors="pt")
    inputs = inputs.to(gen_model.device)
    output = gen_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Try to return only the assistant part after "Answer:" if present
    answer = text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()
    return answer

# ---- Streamlit UI ----
st.set_page_config(page_title="Follow-up-aware RAG Analyst", layout="wide")
st.title("Follow-up-aware RAG Analyst (HF + FAISS)")

# Sidebar: index paths
st.sidebar.header("Index configuration")
index_prefix = st.sidebar.text_input("FAISS index prefix", FAISS_INDEX_PATH)
metadata_path = st.sidebar.text_input("Metadata JSONL path", METADATA_PATH)
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=5, step=1)

# Load models
embedder = load_embedder()
tokenizer, gen_model = load_generator()

# Load index + metadata when paths change
if "store" not in st.session_state or st.session_state.get("index_prefix") != index_prefix:
    try:
        emb_dim = embedder.get_sentence_embedding_dimension()
        st.session_state.store = load_index(index_prefix, 
                                            emb_dim=emb_dim)
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
    # RAG pipeline with loading indicators
    with st.spinner("Rewriting follow-up question..."):
        standalone_q = rewrite_question(
            tokenizer, gen_model,
            st.session_state.history,
            user_q.strip()
        )

    # with st.spinner("Retrieving relevant chunks..."):
    #     docs, sources = retrieve(
    #         standalone_q,
    #         st.session_state.store,
    #         embedder, st.session_state.meta,
    #         k=top_k
    #     )
    with st.spinner("Retrieving relevant chunks with query expansion..."):
        docs, sources = retrieve_with_expansion(
            tokenizer, gen_model,
            standalone_q,
            st.session_state.store,
            embedder, st.session_state.meta,
            top_k=top_k
        )

    with st.spinner("Generating answer..."):
        answer = generate_answer(tokenizer, gen_model, standalone_q, docs)

    # Update history
    st.session_state.history.append(("user", user_q.strip()))
    st.session_state.history.append(("assistant", answer))

    # Show results
    st.subheader("User Question")
    st.write(user_q.strip())
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
    st.rerun()
if cols[1].button("Reload index"):
    # Re-load cached resources
    emb_dim = embedder.get_sentence_embedding_dimension()
    st.session_state.store = load_index(index_prefix, 
                                        emb_dim=emb_dim)
    st.session_state.meta = load_metadata(metadata_path)
    st.success("Index reloaded.")

