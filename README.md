# level4-lesson3 · Retrieval quality

STEMLink AI Engineer Bootcamp · Level 4 · Lesson 3

Lesson 2 built FieldOracle. This lesson measures how well it retrieves, and makes it better
one measured change at a time: hybrid search, cross-encoder reranking, chunking, filters,
and reading the failures.

Notebooks 00 to 05 and 07 run **offline** on a bench corpus with a small local embedding
model and FAISS. Everyone gets the same numbers and nothing costs money. Notebook 06 runs
the same measurements on your own Pinecone index.

## Setup

### macOS

```bash
git clone <repo-url> level4-lesson3
cd level4-lesson3
uv sync
uv run python -m scripts.warmup
```

### Windows (PowerShell)

```powershell
git clone <repo-url> level4-lesson3
cd level4-lesson3
uv sync
uv run python -m scripts.warmup
```

`scripts.warmup` downloads the embedding model (bge-small, about 70 MB) and the reranker
(about 35 MB) into `.cache/`. Do it before class.

For notebook 06 only: copy `.env.example` to `.env` and fill in your keys, then copy
`data/my_questions.example.json` to `data/my_questions.json` and add your ten questions.

## Running a notebook

```bash
uv run marimo edit notebooks/00_bench.py
```

## Notebooks

- **00_bench.py** · The bench: a fixed corpus, questions with evidence phrases, and a dense baseline
<!-- notebook-00 -->
- **01_metrics.py** · recall@k, precision@k and MRR, by hand and over the whole set; what k costs
<!-- notebook-01 -->
- **02_hybrid.py** · BM25 and its tokenizer, reciprocal rank fusion by hand, `EnsembleRetriever`
<!-- notebook-02 -->
- **03_rerank.py** · Cross-encoder versus bi-encoder scores, two-stage retrieval, candidate count versus latency
<!-- notebook-03 -->
- **04_tuning.py** · One knob at a time: chunk size at equal budget, overlap, a metadata filter
<!-- notebook-04 -->
- **05_failures.py** · Where the evidence went: query, chunking, index or corpus; unanswerable questions
<!-- notebook-05 -->
- **06_your_corpus.py** · The same measurements on your Pinecone index; writes `reports/retrieval_report.md`
<!-- notebook-06 -->
<!-- notebook-07 -->

## What is where

| Path | What it is |
|---|---|
| `data/bench/` | The FieldOracle-style bench: 22 documents in three categories, 36 questions with evidence phrases |
| `data/helpdesk/` | The breakout corpus: a fictional appliance company's manuals, policies and tickets |
| `scripts/metrics.py` | recall@k, precision@k and reciprocal rank, as written in notebook 01 |
| `scripts/bench.py` | Loading, splitting, indexing, and running a retriever over a question set |
| `scripts/pinecone_corpus.py` | Reads every chunk back out of your Pinecone index (notebook 06) |
| `reports/` | Where notebook 06 writes your retrieval report |
| `docs/retrieval-lab.html` | The Retrieval Lab explorable (concept map, metrics, fusion, rerank cost, chunk budget). Open it in a browser |

All documents in `data/` are a teaching corpus. Figures, circulars and records are
illustrative and fictional; they are not official recommendations.

`langchain-community`, which provides FAISS, BM25Retriever, FastEmbedEmbeddings and
FlashrankRerank, is in maintenance mode. The versions are pinned in `uv.lock`.
