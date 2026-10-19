"""The bench, Lesson 2's FieldOracle PDFs, the questions, and Lesson 2's reader and splitter."""

import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DATA = Path(__file__).resolve().parent.parent / "data"


def load_docs(name="bench"):
    """One Document per file, tagged with the metadata in catalog.json."""
    catalog = json.loads((DATA / name / "catalog.json").read_text())
    docs = []
    for path in sorted((DATA / name / "docs").glob("*.md")):
        metadata = {"source": path.name, "current": True, **catalog[path.name]}
        docs.append(Document(page_content=path.read_text(), metadata=metadata))
    return docs


def read_pdf(path, metadata):
    """Lesson 2's reader: one Document per page that has text, with its page number."""
    reader = PdfReader(path)
    return [
        Document(page_content=text, metadata={**metadata, "page": number})
        for number, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text() or "").strip()
    ]


def load_pdfs(name="fieldoracle"):
    """Every page of every PDF in data/<name>/docs, tagged the way Lesson 2's upload tagged it.

    The category comes from the file prefix (CG-, POL-, FR-), as in index_corpus.sh. The doc_id
    is Lesson 2's: the file name plus the first six hex digits of the file's SHA-256, so the
    chunk ids from split() match the ids in your Pinecone index.
    """
    categories = {"CG": "cultivation_guide", "POL": "policy", "FR": "farm_record"}
    pages = []
    for path in sorted((DATA / name / "docs").glob("*.pdf")):
        stem = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
        doc_id = f"{stem}-{hashlib.sha256(path.read_bytes()).hexdigest()[:6]}"
        category = categories[path.name.split("-")[0]]
        pages += read_pdf(path, {"doc_id": doc_id, "source": path.name, "category": category})
    return pages


def load_questions(name="bench", answerable=True):
    """Questions with evidence phrases. answerable=False returns the ones with no answer."""
    questions = json.loads((DATA / name / "questions.json").read_text())
    return [q for q in questions if bool(q["evidence"]) == answerable]


def split(docs, size=1000, overlap=150):
    """Lesson 2's splitter. Each chunk's id is its document id (or file name) plus its position.

    The pages of one PDF share a doc_id, so their chunks are numbered on from page to page.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks, count = [], {}
    for chunk in splitter.split_documents(docs):
        stem = chunk.metadata.get("doc_id") or chunk.metadata["source"].removesuffix(".md")
        chunk.id = f"{stem}-{count.get(stem, 0)}"
        count[stem] = count.get(stem, 0) + 1
        chunks.append(chunk)
    return chunks
