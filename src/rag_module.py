from __future__ import annotations
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

from faiss_module import FaissIndexStore

# Optional imports (loaded only when needed)
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- Utilities ----------
def _extract_years(text: str) -> List[str]:
    return re.findall(r"(20\d{2})", text or "")

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

def discover_companies(meta: Dict) -> List[str]:
    seen = set()
    for item in meta.values():
        comp = _company_from_path((item.get("meta") or {}).get("file"))
        if comp:
            seen.add(comp)
    return sorted(seen)


# ---------- Abstract Provider ----------
class Provider(ABC):
    @abstractmethod
    def rewrite_question(self, history: List[Tuple[str, str]], question: str) -> str: ...
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray: ...
    @abstractmethod
    def expand_queries(self, question: str, history: List[Tuple[str, str]], companies: List[str]) -> List[str]: ...
    @abstractmethod
    def generate_answer(self, query: str, retrieved_docs: List[str]) -> str: ...


# ---------- OpenAI Provider ----------
class OpenAIProvider(Provider):
    def __init__(self, client: OpenAI, gen_model: str = "gpt-4o-mini", embed_model: str = "text-embedding-3-large"):
        self.client = client
        self.gen_model = gen_model
        self.embed_model = embed_model

    def rewrite_question(self, history: List[Tuple[str, str]], question: str) -> str:
        hist_text = "\n".join([f"{r}: {t}" for r, t in history][-8:])
        resp = self.client.responses.create(
            model=self.gen_model,
            input=[
                {"role": "system", "content":
                 "Rewrite the user's follow-up question into a standalone query understandable without history."},
                {"role": "user", "content":
                 f"History:\n{hist_text}\n\nFollow-up question: {question}\n\nRewritten standalone question:"},
            ],
            max_output_tokens=96,
        )
        text = getattr(resp, "output_text", "").strip()
        return text or question

    def embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.embed_model, input=[query])
        return np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)

    def expand_queries(self, question: str, history: List[Tuple[str, str]], companies: List[str]) -> List[str]:
        years = _extract_years(question)
        year_hint = years[0] if years else ""
        sys_msg = (
            "Expand search queries for annual report retrieval. "
            "Output 6–10 short queries (one per line) with company names and profitability synonyms. "
            "Prefer adding the target year if present. No explanations."
        )
        usr_msg = f"Question: {question}\nCompanies: {', '.join(companies[:8])}\nYear: {year_hint}"
        resp = self.client.responses.create(
            model=self.gen_model,
            input=[{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}],
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
        # dedupe and cap
        seen, final_q = set(), []
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

    def generate_answer(self, query: str, retrieved_docs: List[str]) -> str:
        context = "\n\n".join(retrieved_docs) if retrieved_docs else "No context retrieved."
        resp = self.client.responses.create(
            model=self.gen_model,
            input=[
                {"role": "system", "content":
                    "You are a helpful analyst specializing in the analysis of Annual Reports within the automotive sector. "
                    "Your daily tasks involve extracting key financial metrics such as revenue, EBITDA, growth numbers, and conducting comparative analyses across different car companies, as well as comparing them to other sectors. "
                    "Use the provided context to answer the user's question. "
                    "If the answer cannot be found in the context, say you don't know. "
                    "Cite page numbers when possible."
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
            ],
            max_output_tokens=300,
        )
        return getattr(resp, "output_text", "").strip()


# ---------- Hugging Face Provider ----------
class HFProvider(Provider):
    def __init__(self, tokenizer: AutoTokenizer, gen_model: AutoModelForCausalLM, embedder: SentenceTransformer):
        self.tokenizer = tokenizer
        self.gen_model = gen_model
        self.embedder = embedder

    def rewrite_question(self, history: List[Tuple[str, str]], question: str) -> str:
        hist_text = "\n".join([f"{r}: {t}" for r, t in history][-8:])
        messages = [
            {"role": "system", "content":
             "Rewrite the user's follow-up question into a standalone query understandable without history."},
            {"role": "user", "content":
             f"History:\n{hist_text}\n\nFollow-up question: {question}\n\nRewritten standalone question:"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.gen_model.device)
        out = self.gen_model.generate(**inputs, max_new_tokens=96, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # Best-effort split
        return (text.split("Rewritten standalone question:")[-1].strip() or text or question)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query], convert_to_numpy=True)

    def expand_queries(self, question: str, history: List[Tuple[str, str]], companies: List[str]) -> List[str]:
        years = _extract_years(question)
        year_hint = years[0] if years else ""
        sys_msg = (
            "Expand search queries for annual report retrieval. "
            "Output 6–10 short queries (one per line) with company names and profitability synonyms. "
            "Prefer adding the target year if present. No explanations."
        )
        usr_msg = f"Question: {question}\nCompanies: {', '.join(companies[:8])}\nYear: {year_hint}"
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.gen_model.device)
        out = self.gen_model.generate(**inputs, max_new_tokens=160, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        synonyms = ["profitability", "operating margin", "EBIT", "EBITDA", "net income", "profit margin", "return on sales", "operating profit"]
        det = []
        for comp in companies:
            for syn in synonyms[:4]:
                if year_hint:
                    det.append(f"{comp} {year_hint} {syn}")
                det.append(f"{comp} {syn}")

        queries = [question.strip()] + lines + det
        seen, final_q = set(), []
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

    def generate_answer(self, query: str, retrieved_docs: List[str]) -> str:
        context = "\n\n".join(retrieved_docs) if retrieved_docs else "No context retrieved."
        messages = [
            {"role": "system", "content":
                "You are a helpful analyst specializing in the analysis of Annual Reports within the automotive sector. "
                "Your daily tasks involve extracting key financial metrics such as revenue, EBITDA, growth numbers, and conducting comparative analyses across different car companies, as well as comparing them to other sectors. "
                "Use the provided context to answer the user's question. "
                "If the answer cannot be found in the context, say you don't know. "
                "Cite page numbers when possible."
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.gen_model.device)
        out = self.gen_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()


# ---------- Hybrid Provider ----------
class HybridProvider(Provider):
    """
    Hybrid provider:
    - Embeddings: SentenceTransformers (HF)
    - Rewriting, query expansion, answer generation: OpenAI Responses
    """
    def __init__(self, oai_client: OpenAI, hf_embedder: SentenceTransformer, gen_model: str = "gpt-4o-mini"):
        self.client = oai_client
        self.embedder = hf_embedder
        self.gen_model = gen_model

    # Reuse OpenAI for rewriting
    def rewrite_question(self, history: List[Tuple[str, str]], question: str) -> str:
        hist_text = "\n".join([f"{r}: {t}" for r, t in history][-8:])
        resp = self.client.responses.create(
            model=self.gen_model,
            input=[
                {"role": "system", "content":
                 "Rewrite the user's follow-up question into a standalone query understandable without history."},
                {"role": "user", "content":
                 f"History:\n{hist_text}\n\nFollow-up question: {question}\n\nRewritten standalone question:"},
            ],
            max_output_tokens=96,
        )
        text = getattr(resp, "output_text", "").strip()
        return text or question

    # Embed query with HF SentenceTransformers
    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query], convert_to_numpy=True)

    # Use OpenAI for expansion
    def expand_queries(self, question: str, history: List[Tuple[str, str]], companies: List[str]) -> List[str]:
        years = _extract_years(question)
        year_hint = years[0] if years else ""
        sys_msg = (
            "Expand search queries for annual report retrieval. "
            "Output 6–10 short queries (one per line) with company names and profitability synonyms. "
            "Prefer adding the target year if present. No explanations."
        )
        usr_msg = f"Question: {question}\nCompanies: {', '.join(companies[:8])}\nYear: {year_hint}"
        resp = self.client.responses.create(
            model=self.gen_model,
            input=[{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}],
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
        seen, final_q = set(), []
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

    # Use OpenAI for generation
    def generate_answer(self, query: str, retrieved_docs: List[str]) -> str:
        context = "\n\n".join(retrieved_docs) if retrieved_docs else "No context retrieved."
        resp = self.client.responses.create(
            model=self.gen_model,
            input=[
                {"role": "system", "content":
                    "You are a helpful analyst specializing in the analysis of Annual Reports within the automotive sector. "
                    "Your daily tasks involve extracting key financial metrics such as revenue, EBITDA, growth numbers, and conducting comparative analyses across different car companies, as well as comparing them to other sectors. "
                    "Use the provided context to answer the user's question. "
                    "If the answer cannot be found in the context, say you don't know. "
                    "Cite page numbers when possible."
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
            ],
            max_output_tokens=300,
        )
        return getattr(resp, "output_text").strip()


# ---------- Pipeline functions (provider-agnostic) ----------
def rewrite_question(provider: Provider, history: List[Tuple[str, str]], question: str) -> str:
    return provider.rewrite_question(history, question)

def retrieve_with_expansion(
    provider: Provider,
    query: str,
    store: FaissIndexStore,
    meta: Dict,
    top_k: int = 6,
    history: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[str], List[Dict]]:
    history = history or []
    companies = discover_companies(meta)
    expanded = provider.expand_queries(query, history, companies)

    # Gather candidates across expansions, keep best distance per id
    candidates: Dict[str, Dict] = {}
    for q in expanded:
        qv = provider.embed_query(q)
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

    # Optional year filter
    years = _extract_years(query)
    if years:
        sy = years[0]
        candidates = {hid: c for hid, c in candidates.items() if sy in str(c["meta"].get("file", ""))}

    # Company-balanced selection
    by_company: Dict[str, List[Dict]] = {}
    for c in candidates.values():
        by_company.setdefault(c["company"], []).append(c)
    for comp in by_company:
        by_company[comp].sort(key=lambda x: x["score"])

    per_cap = max(1, top_k // max(1, len(by_company)))
    selected: List[Dict] = []
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

def generate_answer(provider: Provider, query: str, retrieved_docs: List[str]) -> str:
    return provider.generate_answer(query, retrieved_docs)
