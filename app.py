import json
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "data" / "documents"
INDEX_PATH = APP_DIR / "data" / "index.npz"
CHUNKS_PATH = APP_DIR / "data" / "chunks.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

st.set_page_config(page_title="QROR Research Assistant", page_icon="📚", layout="wide")
st.title("QROR Research Assistant")
st.caption("Ask questions about your course PDFs. Answers are grounded in retrieved passages when available.")


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_index():
    pdf_paths = sorted(DOCS_DIR.rglob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDFs found. Add PDFs under: {DOCS_DIR}")

    records = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for chunk_number, chunk in enumerate(split_text(page_text), start=1):
                    records.append({
                        "text": chunk,
                        "source": str(pdf_path.relative_to(DOCS_DIR)),
                        "page": page_number,
                        "chunk": chunk_number,
                    })
        except Exception as exc:
            st.warning(f"Could not read {pdf_path.name}: {exc}")

    if not records:
        raise ValueError("No extractable text found. Scanned PDFs may need OCR first.")

    embedder = load_embedder()
    texts = [record["text"] for record in records]
    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDEX_PATH, vectors=vectors)
    CHUNKS_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records), len(pdf_paths)


def retrieve(question: str, top_k: int = TOP_K):
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError("Build the document index first using the sidebar.")

    vectors = np.load(INDEX_PATH)["vectors"]
    records = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    query_vector = load_embedder().encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )[0].astype("float32")
    scores = vectors @ query_vector
    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for source_id, idx in enumerate(best_indices, start=1):
        record = records[int(idx)].copy()
        record["score"] = float(scores[idx])
        record["source_id"] = source_id
        results.append(record)
    return results


def ask_ollama(question: str, passages: list[dict]):
    context = "\n\n".join(
        f"[S{p['source_id']}] File: {p['source']} | Page: {p['page']}\n{p['text']}"
        for p in passages
    )
    system_prompt = (
        "You are a careful academic assistant for an M.Tech student in QROR. "
        "Answer using the supplied source passages whenever relevant. Cite factual claims "
        "from the passages using [S1], [S2], etc. Do not invent page numbers or citations. "
        "If the passages do not contain enough evidence, say what is missing and clearly "
        "separate general knowledge from document-supported claims. For mathematical "
        "questions, explain assumptions and show derivations step by step."
    )
    user_prompt = f"Question:\n{question}\n\nRetrieved source passages:\n{context}"
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2},
        },
        timeout=240,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


with st.sidebar:
    st.header("Your course materials")
    st.write("Add PDF files to this folder:")
    st.code(str(DOCS_DIR), language=None)
    st.write("Then build or rebuild the search index.")
    if st.button("Build / refresh index", type="primary", use_container_width=True):
        with st.spinner("Reading PDFs and creating embeddings…"):
            try:
                chunk_count, pdf_count = build_index()
                st.success(f"Indexed {chunk_count} text chunks from {pdf_count} PDF(s).")
            except Exception as exc:
                st.error(str(exc))
    st.divider()
    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    st.caption(f"Answer model: {OLLAMA_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved source passages"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[S{source['source_id']}] {source['source']} — page {source['page']}**  \n"
                        f"Similarity score: {source['score']:.3f}"
                    )
                    st.write(source["text"])

question = st.chat_input("Ask about probability, statistics, OR, or your notes…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes…"):
                passages = retrieve(question)
            with st.spinner("Generating an answer with the local model…"):
                answer = ask_ollama(question, passages)
            st.markdown(answer)
            with st.expander("Retrieved source passages"):
                for source in passages:
                    st.markdown(
                        f"**[S{source['source_id']}] {source['source']} — page {source['page']}**  \n"
                        f"Similarity score: {source['score']:.3f}"
                    )
                    st.write(source["text"])
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": passages
            })
        except FileNotFoundError as exc:
            st.error(str(exc))
        except requests.ConnectionError:
            st.error("Could not connect to Ollama. Start Ollama and make sure the model is available.")
        except requests.HTTPError as exc:
            st.error(f"Ollama returned an error: {exc}")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
