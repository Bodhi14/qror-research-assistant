import json
import os
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "").rstrip("/")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

st.set_page_config(page_title="QROR Research Assistant", page_icon="📚", layout="wide")
st.title("QROR Research Assistant")
st.caption("Ask questions about course PDFs. Answers include retrieved source passages when available.")


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
        raise ValueError("No PDFs found. Upload a PDF in the sidebar or add one under data/documents/.")

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

    vectors = load_embedder().encode(
        [record["text"] for record in records],
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


def build_prompts(question: str, passages: list[dict]):
    context = "\n\n".join(
        f"[S{p['source_id']}] File: {p['source']} | Page: {p['page']}\n{p['text']}"
        for p in passages
    )
    system_prompt = (
        "You are a careful academic assistant for an M.Tech student in QROR. "
        "Use the supplied source passages whenever relevant. Cite claims supported by the passages "
        "using [S1], [S2], etc. Do not invent citations. If evidence is missing, say so and "
        "separate general knowledge from document-supported claims. For mathematical questions, "
        "explain assumptions and show derivations step by step."
    )
    user_prompt = f"Question:\n{question}\n\nRetrieved source passages:\n{context}"
    return system_prompt, user_prompt


def generate_answer(question: str, passages: list[dict]):
    system_prompt, user_prompt = build_prompts(question, passages)
    if LLM_PROVIDER == "ollama":
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
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

    if LLM_PROVIDER == "openai_compatible":
        if not INFERENCE_BASE_URL or not INFERENCE_API_KEY or not INFERENCE_MODEL:
            raise ValueError(
                "For openai_compatible inference, configure INFERENCE_BASE_URL, "
                "INFERENCE_API_KEY, and INFERENCE_MODEL in your environment."
            )
        endpoint = INFERENCE_BASE_URL
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint + "/chat/completions"
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {INFERENCE_API_KEY}"},
            json={
                "model": INFERENCE_MODEL,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai_compatible'.")


def render_sources(passages: list[dict]):
    with st.expander("Retrieved source passages"):
        for source in passages:
            st.markdown(
                f"**[S{source['source_id']}] {source['source']} — page {source['page']}**  \n"
                f"Similarity score: {source['score']:.3f}"
            )
            st.write(source["text"])


with st.sidebar:
    st.header("Course materials")
    st.caption("Upload PDFs for this running app session, or add permitted PDFs under data/documents/.")
    uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Save uploaded PDFs", use_container_width=True, disabled=not uploaded_pdfs):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        saved = 0
        for uploaded_pdf in uploaded_pdfs:
            safe_name = Path(uploaded_pdf.name).name
            (DOCS_DIR / safe_name).write_bytes(uploaded_pdf.getvalue())
            saved += 1
        st.success(f"Saved {saved} PDF(s). Now build or refresh the index.")
    if st.button("Build / refresh index", type="primary", use_container_width=True):
        with st.spinner("Reading PDFs and creating embeddings…"):
            try:
                chunk_count, pdf_count = build_index()
                st.success(f"Indexed {chunk_count} text chunks from {pdf_count} PDF(s).")
            except Exception as exc:
                st.error(str(exc))
    st.divider()
    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    if LLM_PROVIDER == "ollama":
        st.caption(f"Answer model: Ollama / {OLLAMA_MODEL}")
    else:
        st.caption(f"Answer model: {INFERENCE_MODEL or 'not configured'}")
    st.caption(f"Inference provider model: {LLM_PROVIDER}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask about probability, statistics, OR, or your notes…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes…"):
                passages = retrieve(question)
            with st.spinner("Generating an answer…"):
                answer = generate_answer(question, passages)
            st.markdown(answer)
            render_sources(passages)
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": passages
            })
        except FileNotFoundError as exc:
            st.error(str(exc))
        except requests.ConnectionError:
            if LLM_PROVIDER == "ollama":
                st.error("Could not connect to Ollama. Start Ollama and verify OLLAMA_BASE_URL and the model.")
            else:
                st.error("Could not connect to the inference provider. Check its URL and network availability.")
        except requests.HTTPError as exc:
            st.error(f"The inference provider returned an HTTP error: {exc}")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
