"""Supporting code for the retrieval bench: loading the corpus, building indexes,
and running a retriever over a question set. The lesson's ideas live in the notebooks."""

import json
import pathlib
import re
import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.metrics import precision_at_k, recall_at_k, reciprocal_rank

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"


# ---------- corpus and questions ----------

def load_corpus(name="bench"):
    """One Document per file, with the metadata from catalog.json."""
    folder = DATA / name
    catalog = json.loads((folder / "catalog.json").read_text())
    docs = []
    for path in sorted((folder / "docs").glob("*.md")):
        info = catalog[path.name]
        metadata = {"source": path.name, "current": True, **info}
        docs.append(Document(page_content=path.read_text(), metadata=metadata))
    return docs


def split(docs, size=1000, overlap=150):
    """Lesson 2's splitter. Each chunk gets a chunk_id: file name plus position."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = []
    for doc in docs:
        pieces = splitter.split_documents([doc])
        stem = doc.metadata["source"].removesuffix(".md")
        for i, piece in enumerate(pieces):
            piece.metadata["chunk_id"] = f"{stem}-{i}"
            chunks.append(piece)
    return chunks


def load_questions(name="bench", answerable_only=True):
    questions = json.loads((DATA / name / "questions.json").read_text())
    if answerable_only:
        questions = [q for q in questions if q["evidence"]]
    return questions


def question(qid, name="bench"):
    return next(q for q in load_questions(name, answerable_only=False) if q["id"] == qid)


# ---------- indexes ----------

@lru_cache(maxsize=1)
def embeddings():
    """A small local embedding model. Runs on CPU, no API key, same numbers for everyone."""
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, cache_dir=str(CACHE / "fastembed"))


def dense_index(chunks):
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(chunks, embeddings())


def tokenize(text):
    """Lowercase words and numbers, keeping codes like k-17, 25,000 and agr/fs/2025/07 whole."""
    return re.findall(r"[a-z0-9]+(?:[-/.,][a-z0-9]+)*", text.lower())


def bm25(chunks, k=5):
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenize)


def reranker(top_n=5):
    from flashrank import Ranker
    from langchain_community.document_compressors import FlashrankRerank

    client = Ranker(model_name=RERANK_MODEL, cache_dir=str(CACHE / "flashrank"))
    return FlashrankRerank(client=client, top_n=top_n)


# ---------- running a retriever over questions ----------

def evaluate(retriever, questions, k=5):
    """Run every question through the retriever and score the top k."""
    rows = []
    for q in questions:
        start = time.perf_counter()
        docs = retriever.invoke(q["question"])[:k]
        ms = (time.perf_counter() - start) * 1000
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "recall": recall_at_k(docs, q, k),
            "precision": precision_at_k(docs, q, k),
            "rr": reciprocal_rank(docs, q),
            "chars": sum(len(d.page_content) for d in docs),
            "ms": ms,
        })
    return rows


def mean(rows, key):
    return round(sum(r[key] for r in rows) / len(rows), 2) if rows else 0.0


def summary(name, rows, k=5):
    print(
        name,
        "| recall@%d" % k, mean(rows, "recall"),
        "| precision@%d" % k, mean(rows, "precision"),
        "| MRR", mean(rows, "rr"),
        "| chars", int(mean(rows, "chars")),
        "| ms", int(mean(rows, "ms")),
    )


def by_type(name, rows):
    for t in ["keyword", "paraphrase", "multi", "filter"]:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            print(name, t, "| recall", mean(sub, "recall"), "| MRR", mean(sub, "rr"))


def misses(rows):
    return [r["id"] for r in rows if r["recall"] < 1]


def show(docs, q, width=90):
    """Print a ranked list. A star marks a chunk that contains evidence."""
    from scripts.metrics import is_relevant

    print("Q:", q["question"])
    for position, doc in enumerate(docs, start=1):
        mark = "*" if is_relevant(doc, q) else " "
        text = " ".join(doc.page_content.split())[:width]
        print(position, mark, doc.metadata.get("chunk_id"), "|", text)


def rank_of_evidence(retriever, q):
    """Position of the first relevant chunk in whatever the retriever returns, or None."""
    from scripts.metrics import is_relevant

    for position, doc in enumerate(retriever.invoke(q["question"]), start=1):
        if is_relevant(doc, q):
            return position
    return None
