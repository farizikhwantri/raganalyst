import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

from faiss_module import FaissIndexStore

from config import EMBED_MODEL_NAME
from config import GEN_MODEL_NAME
from config import FAISS_INDEX_PATH
from config import METADATA_PATH

def rewrite_question(history: list, question: str) -> str:
    """
    Rewrite follow-up question into a standalone query.
    history: list of (role, text), e.g., [("user","..."),("assistant","...")]
    """
    hist_text = "\n".join([f"{r}: {t}" for r, t in history][-8:])  # last 8 turns
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
    # keep the part after the marker
    rewritten = text.split("Rewritten standalone question:")[-1].strip()
    return rewritten or question

def retrieve(query: str, store: FaissIndexStore, embedder, meta: dict, k: int = 4):
    qvec = embedder.encode([query], convert_to_numpy=True)
    hits = store.search(qvec, k=k)
    # Return ordered chunks
    docs = []
    for hid, dist in hits:
        item = meta.get(hid)
        if not item:
            continue
        docs.append(item["text"])
    return docs

def generate_answer(query: str, retrieved_docs: list) -> str:
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

def rag_with_followup(question: str, history: list, store: FaissIndexStore, embedder, meta: dict, k: int = 5):
    standalone_q = rewrite_question(history, question)
    docs = retrieve(standalone_q, store, embedder, meta, k=k)
    answer = generate_answer(standalone_q, docs)
    return answer, standalone_q

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )


    # existing setup
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    emb_dim = embedder.get_sentence_embedding_dimension()
    store = FaissIndexStore.load(FAISS_INDEX_PATH, 
                                 dim=emb_dim)

    # load metadata map id -> item
    meta = {}
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            meta[item["id"]] = item

    # simple chat loop
    history = []
    print("Follow-up-aware RAG with Llama 3.2-1B Instruct. Type Ctrl+C to exit.")
    try:
        while True:
            q = input("User: ").strip()
            if not q:
                continue
            answer, standalone_q = rag_with_followup(q, history, store, embedder, meta, k=5)
            print(f"Standalone: {standalone_q}")
            print(f"Assistant: {answer}\n")
            history.append(("user", q))
            history.append(("assistant", answer))
    except KeyboardInterrupt:
        print("\nBye.")

