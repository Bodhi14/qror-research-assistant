"""Import PDF attachments from Gmail into QROR subject folders.

Uses Gmail API read-only OAuth access. It never sends, labels, or deletes email.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from email.utils import parseaddr
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader

APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "data" / "documents"
DB_PATH = APP_DIR / "data" / "gmail_imports.sqlite3"
CREDENTIALS_PATH = APP_DIR / "credentials.json"
TOKEN_PATH = APP_DIR / "token.json"
DEFAULT_QUERY = (
    "has:attachment filename:pdf "
    "{to:qror26_28@googlegroups.com list:qror26_28.googlegroups.com}"
)
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

SUBJECTS = [
    "Statistical_Methods_1",
    "Probability",
    "Operations_Research",
    "Quality_Management",
    "Data_Structure_and_Programming",
]

# Sender mappings are strong signals, but can be edited for your actual email headers.
SENDER_TO_SUBJECT = {
    "susantagauri@hotmail.com": "Statistical_Methods_1",
    "nandini@isical.ac.in": "Probability",
    "sahadeepjyoti01@gmail.com": "Operations_Research",
    "kaustav.kundu@isical.ac.in": "Quality_Management",
    "ujjwal.isical@gmail.com": "Data_Structure_and_Programming",
}

# Terms are intentionally broad; sender, subject, filename, and PDF text are scored separately.
KEYWORDS = {
    "Statistical_Methods_1": [
        "statistical methods", "statistics", "regression", "sampling", "chi-square",
        "hypothesis testing", "estimation", "confidence interval", "statistical inference",
        "goodness of fit", "correlation", "anova", "least squares",
    ],
    "Probability": [
        "probability", "random variable", "moment generating", "mgf", "distribution",
        "conditional probability", "binomial", "poisson", "normal distribution", "expectation",
        "variance", "markov", "bayes", "joint density", "cumulative distribution",
    ],
    "Operations_Research": [
        "operations research", "operation research", "linear programming", "simplex",
        "transportation problem", "assignment problem", "hungarian", "duality", "optimization",
        "queueing", "queuing", "inventory model", "decision theory", "big m", "northwest corner",
    ],
    "Quality_Management": [
        "quality management", "quality control", "control chart", "six sigma", "process capability",
        "acceptance sampling", "reliability", "iso 9001", "pareto", "cause and effect",
        "fishbone", "quality assurance", "defect rate",
    ],
    "Data_Structure_and_Programming": [
        "data structure", "programming", "algorithm", "python", "c++", "linked list", "binary tree",
        "graph traversal", "sorting", "time complexity", "recursion", "stack", "queue data structure",
        "hash table", "dynamic programming", "source code",
    ],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decode_b64url(value: str | None) -> bytes:
    if not value:
        return b""
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def safe_filename(name: str) -> str:
    name = Path(name or "attachment.pdf").name
    name = re.sub(r"[^A-Za-z0-9._() -]+", "_", name).strip(" .")
    if not name:
        name = "attachment.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:180]


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS imported_attachments (
                sha256 TEXT PRIMARY KEY,
                gmail_message_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                sender TEXT NOT NULL,
                email_subject TEXT NOT NULL,
                classified_subject TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_json TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                imported_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_imported_subject ON imported_attachments(classified_subject)")
        conn.commit()


def get_existing_hashes() -> set[str]:
    with sqlite3.connect(DB_PATH) as conn:
        return {row[0] for row in conn.execute("SELECT sha256 FROM imported_attachments")}


def pdf_text(data: bytes, max_chars: int = 12000) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        parts = []
        for page in reader.pages[:12]:
            parts.append(page.extract_text() or "")
            if sum(len(p) for p in parts) >= max_chars:
                break
        return " ".join(" ".join(parts).split())[:max_chars]
    except Exception:
        return ""


def classify_subject(sender: str, email_subject: str, email_body: str, filename: str, extracted_pdf_text: str):
    """Always returns the highest-scoring subject, plus confidence and evidence."""
    sender = parseaddr(sender)[1].lower().strip()
    if sender in SENDER_TO_SUBJECT:
        chosen = SENDER_TO_SUBJECT[sender]
        return chosen, 0.98, [f"sender mapping: {sender} -> {chosen}"]

    fields = [
        ("email subject", email_subject.lower(), 6.0),
        ("filename", filename.lower(), 4.0),
        ("email body", email_body.lower(), 1.5),
        ("PDF text", extracted_pdf_text.lower(), 1.0),
    ]
    scores = {subject: 0.0 for subject in SUBJECTS}
    evidence: dict[str, list[str]] = {subject: [] for subject in SUBJECTS}
    for subject, terms in KEYWORDS.items():
        for field_name, content, weight in fields:
            for term in terms:
                count = content.count(term)
                if count:
                    # Cap each term per field to prevent one repeated word dominating the classification.
                    points = weight * min(count, 3)
                    scores[subject] += points
                    evidence[subject].append(f"{field_name}: '{term}' (+{points:g})")

    # Stable tie-break order means even an uninformative attachment gets a destination.
    chosen = max(SUBJECTS, key=lambda subject: scores[subject])
    total = sum(scores.values())
    if total == 0:
        confidence = 0.05
        chosen_evidence = ["No subject-specific evidence found; assigned the first configured subject as a last-resort fallback."]
    else:
        ranked = sorted(scores.values(), reverse=True)
        confidence = min(0.95, max(0.10, scores[chosen] / total * (0.5 + 0.5 * (scores[chosen] / max(ranked[1] if len(ranked) > 1 else 0, 1.0)))))
        chosen_evidence = evidence[chosen][:12]
        chosen_evidence.insert(0, "Scores: " + ", ".join(f"{key}={value:g}" for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)))
    return chosen, round(confidence, 3), chosen_evidence


def extract_message_body(payload: dict[str, Any]) -> str:
    parts_to_check = [payload]
    chunks: list[str] = []
    while parts_to_check:
        part = parts_to_check.pop()
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        if mime == "text/plain" and body.get("data"):
            try:
                chunks.append(decode_b64url(body["data"]).decode("utf-8", errors="replace"))
            except Exception:
                pass
        parts_to_check.extend(part.get("parts", []) or [])
    return " ".join(" ".join(chunks).split())[:20000]


def find_pdf_parts(payload: dict[str, Any]):
    parts_to_check = [payload]
    while parts_to_check:
        part = parts_to_check.pop()
        filename = part.get("filename", "")
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        if filename and (filename.lower().endswith(".pdf") or mime == "application/pdf"):
            attachment_id = body.get("attachmentId")
            if attachment_id:
                yield attachment_id, filename or "attachment.pdf"
        parts_to_check.extend(part.get("parts", []) or [])


def gmail_service():
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CREDENTIALS_PATH.name}. Create an OAuth Desktop app credential in Google Cloud, "
            "download its JSON, and place it beside gmail_importer.py. See README.md."
        )
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Install requirements first: python -m pip install -r requirements.txt") from exc

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def header_map(payload: dict[str, Any]) -> dict[str, str]:
    return {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers", [])}


def store_attachment(data: bytes, filename: str, subject: str, dry_run: bool) -> str:
    folder = DOCS_DIR / subject
    folder.mkdir(parents=True, exist_ok=True)
    base = safe_filename(filename)
    target = folder / base
    if target.exists():
        suffix = hashlib.sha256(data).hexdigest()[:8]
        target = folder / f"{target.stem}_{suffix}{target.suffix}"
    if not dry_run:
        target.write_bytes(data)
    return str(target.relative_to(APP_DIR))


def import_pdfs(query: str, max_results: int, dry_run: bool = False) -> int:
    init_db()
    service = gmail_service()
    existing_hashes = get_existing_hashes()
    page_token = None
    processed_messages = 0
    imported_count = 0

    while True:
        result = service.users().messages().list(
            userId="me", q=query, maxResults=min(max_results, 500), pageToken=page_token
        ).execute()
        messages = result.get("messages", [])
        for item in messages:
            message = service.users().messages().get(userId="me", id=item["id"], format="full").execute()
            payload = message.get("payload", {})
            headers = header_map(payload)
            sender_header = headers.get("from", "")
            sender_email = parseaddr(sender_header)[1].lower()
            email_subject = headers.get("subject", "(no subject)")
            body = extract_message_body(payload)
            processed_messages += 1

            for attachment_id, filename in find_pdf_parts(payload):
                attachment = service.users().messages().attachments().get(
                    userId="me", messageId=item["id"], id=attachment_id
                ).execute()
                data = decode_b64url(attachment.get("data"))
                if not data:
                    print(f"SKIP empty attachment: {filename}")
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in existing_hashes:
                    print(f"DUPLICATE skipped: {filename}")
                    continue

                text = pdf_text(data)
                classified, confidence, evidence = classify_subject(
                    sender_header, email_subject, body, filename, text
                )
                relative_path = store_attachment(data, filename, classified, dry_run)
                print(f"{'DRY RUN ' if dry_run else ''}IMPORT: {filename} -> {relative_path}")
                print(f"  sender={sender_email or '(unknown)'} | confidence={confidence:.3f}")
                print(f"  evidence={'; '.join(evidence[:4])}")

                if not dry_run:
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("""
                            INSERT OR IGNORE INTO imported_attachments
                            (sha256, gmail_message_id, attachment_id, original_filename, sender,
                             email_subject, classified_subject, confidence, evidence_json, stored_path, imported_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            digest, item["id"], attachment_id, filename, sender_email,
                            email_subject, classified, confidence,
                            json.dumps(evidence, ensure_ascii=False), relative_path, now_iso(),
                        ))
                        conn.commit()
                    existing_hashes.add(digest)
                    imported_count += 1

        page_token = result.get("nextPageToken")
        if not page_token or processed_messages >= max_results:
            break

    print(f"\nFinished. Messages checked: {processed_messages}; new PDFs imported: {imported_count}.")
    if dry_run:
        print("Dry run: no PDFs or audit records were written.")
    print(f"Audit database: {DB_PATH.relative_to(APP_DIR)}")
    return imported_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Import PDF attachments from Gmail into QROR subject folders.")
    parser.add_argument("--query", default=os.environ.get("GMAIL_QUERY", DEFAULT_QUERY), help="Gmail search query")
    parser.add_argument("--max-results", type=int, default=100, help="Maximum Gmail messages to inspect (default: 100)")
    parser.add_argument("--dry-run", action="store_true", help="Classify and preview attachments without saving PDFs or audit records")
    args = parser.parse_args()
    if args.max_results < 1:
        parser.error("--max-results must be at least 1")
    try:
        import_pdfs(args.query, args.max_results, args.dry_run)
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
