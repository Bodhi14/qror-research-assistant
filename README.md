# QROR Research Assistant

A Streamlit retrieval-augmented generation (RAG) assistant for QROR course PDFs. It extracts text from PDFs, embeds chunks with Sentence Transformers, retrieves relevant passages using cosine similarity, and generates answers with source references.

## Features

- Upload PDF files in the sidebar or place permitted PDFs in `data/documents/`.
- Build a local dense-vector index using `sentence-transformers/all-MiniLM-L6-v2`.
- Fall back to OCR for image-only PDF pages, including many scanned notes.
- Use Ollama locally, or configure an OpenAI-compatible inference endpoint for a cloud deployment.
- Inspect the retrieved passages and page numbers behind answers.

## Local setup (Ollama)

Use Python 3.11 if possible. Install and start Ollama, then pull the default model:

```powershell
ollama pull gemma4:e2b
```

Create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

The default configuration is `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://localhost:11434`, and `OLLAMA_MODEL=gemma4:e2b`.

### Environment configuration

For local development, copy `.env.example` to `.env` and edit the values:

```powershell
Copy-Item .env.example .env
```

The app loads `.env` at startup with `python-dotenv`. With the default Ollama
configuration, only these settings are needed:

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b
```

`INFERENCE_BASE_URL`, `INFERENCE_API_KEY`, and `INFERENCE_MODEL` are only needed
when `LLM_PROVIDER=openai_compatible`. They tell the app where to send the
grounded question and retrieved PDF passages, which API key authenticates that
request, and which hosted model should generate the answer. They are not needed
for local Ollama.

On Streamlit Community Cloud, there is no local `.env` file. Add the same
values under **App settings -> Secrets** instead; Streamlit exposes those
secrets to the app as `st.secrets`.

## Cloud deployment with an OpenAI-compatible inference API

This section is required only for Streamlit Cloud or another host where Ollama
is not running. Ollama at `http://localhost:11434` refers to the hosting server,
not your personal computer, so Streamlit Cloud needs a hosted model API.

### Where to configure it in Streamlit Cloud

1. Open your app in Streamlit Cloud.
2. Open **Manage app** or the app menu.
3. Select **Settings -> Secrets**.
4. Paste the following TOML configuration, replacing all placeholder values:

```toml
LLM_PROVIDER = "openai_compatible"
INFERENCE_BASE_URL = "https://YOUR_PROVIDER_BASE_URL/v1"
INFERENCE_API_KEY = "your-provider-api-key"
INFERENCE_MODEL = "exact-model-id"
```

5. Save the secrets and reboot or rerun the app.

These settings are read by `app.py` through `st.secrets`:

- `LLM_PROVIDER=openai_compatible`
- `INFERENCE_BASE_URL`: the provider's OpenAI-compatible base URL, usually ending in `/v1`; the app adds `/chat/completions`
- `INFERENCE_API_KEY`: the provider's secret API key, sent as a Bearer token
- `INFERENCE_MODEL`: the exact model identifier supported by the provider

Do not commit API keys, OAuth files, or `.env` files. Check the provider's terms, model license, pricing, and data retention before sending course content to it.

For DigitalOcean App Platform, connect the repository, configure a Python web service, and use a run command such as:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8080
```

Configure the service's HTTP port to match. Add the inference variables as encrypted secrets in App Platform. The default App Platform filesystem may be ephemeral: uploaded PDFs and indexes may not survive a redeploy or restart. For a hack-day demo, upload a small, permitted sample PDF and rebuild the index after deployment; for persistent multi-user storage, use object storage and a deliberate index lifecycle.

## How it works

1. Extracts text from PDF pages with `pypdf`.
2. If a page has no extractable text, renders it and runs Tesseract OCR.
3. Splits page text into overlapping chunks and stores file/page metadata.
4. Embeds chunks with `all-MiniLM-L6-v2`.
5. Retrieves the top four chunks by cosine similarity.
6. Sends the question and retrieved passages to the selected LLM backend and asks it to cite source labels such as `[S1]`.

### Handwritten and scanned notes

`pypdf` can read text embedded in a PDF, but it cannot read handwriting or
image-only scans. The app now automatically tries OCR for pages where `pypdf`
returns no text. Streamlit Cloud installs Tesseract from `packages.txt`; local
Windows development requires installing the Tesseract application separately
and adding it to `PATH`.

OCR is a best-effort fallback: Tesseract works substantially better on clear
printed text than on handwriting, equations, faint pencil, skewed pages, or
complex layouts. After uploading notes, click **Build / refresh index** again so
the OCR text is embedded. If the writing is still not recognized, use a
higher-resolution scan or a handwriting-capable vision/OCR service.

## End-to-end orchestration

```mermaid
flowchart TD
    A[QROR PDF arrives] --> B{How is it received?}

    B -->|Manual upload| C[Streamlit PDF uploader]
    B -->|Email attachment| D[Email automation service]

    D --> E[Read email and download PDF attachment]
    E --> F[Validate PDF and choose subject folder]
    F --> G[Save PDF under data/documents/]

    C --> G

    G --> H[User clicks Build / refresh index]
    H --> I[pypdf extracts text page by page]
    I --> J[Split text into overlapping chunks]
    J --> K[Sentence Transformers creates embeddings]
    K --> L[Save vector index to data/index.npz]
    K --> M[Save chunk metadata to data/chunks.json]

    N[Student asks a question] --> O[Embed the question]
    O --> P[NumPy cosine similarity search]
    P --> Q[Select top 4 passages]
    Q --> R[Build grounded prompt with S1, S2 citations]

    R --> S{Selected LLM provider}

    S -->|Ollama| T[Local Gemma or Qwen model]
    S -->|OpenAI-compatible API| U[Hosted inference model]

    T --> V[Generated answer]
    U --> V

    V --> W[Display answer in Streamlit]
    Q --> X[Display source files, pages, passages, and similarity scores]

    W --> Y[Student verifies answer against sources]
    X --> Y
```

### Why answers contain `[S1]` and `[S2]`

The retrieval step assigns each selected passage a source ID before the prompt is
sent to the model. For example:

```text
[S1] File: Probability/notes.pdf | Page: 3
[S2] File: Probability/lecture.pdf | Page: 7
```

The system prompt then tells the model to cite claims supported by those passages
using `[S1]`, `[S2]`, and so on. This is intentional: the labels connect an answer
back to the retrieved evidence shown in the Streamlit source expander. They are
not additional facts generated by the model.

If you prefer citations to be less prominent, you can change the instruction in
`build_prompts()` in `app.py`, but removing it makes answers harder to verify.

## Limitations and safety

- Scanned PDFs may require OCR. Equations, tables, and multi-column layouts may extract poorly.
- Similarity scores are not probabilities and do not prove that a passage answers a question. Verify important claims against displayed source text.
- This is a prototype, not a secure multi-tenant document platform. Do not upload confidential or restricted course materials to a public demo without permission.
- A hosted inference provider may process prompts and retrieved passages. Review its data policy before use.
- `gmail_importer.py` is a separate future/optional email-ingestion script; it should not contain the Streamlit app.

## License

Add an OSI-approved license before publishing the repository. Also review the model and dependency licenses.
