# level4-lesson3 · Retrieval quality

STEMLink AI Engineer Bootcamp · Level 4 · Lesson 3

Lesson 2 built FieldOracle. This lesson measures how well it retrieves, and makes it better
one measured change at a time: hybrid search, cross-encoder reranking, chunking, filters,
and reading the failures.

Notebooks 00 to 05 and 07 run on a bench corpus of 22 documents, with the same
OpenAI embeddings and Pinecone account as Lessons 1 and 2. The vector store is held in
memory so it can be rebuilt in seconds when we change the chunk size. Notebook 06 runs the
same measurements on your own Pinecone index, reading Lesson 2's FieldOracle PDFs from
`data/fieldoracle` for the keyword side. Metrics come from
[ranx](https://amenra.github.io/ranx/), a standard library for ranking evaluation.

## Setup

### macOS

```bash
git clone git@github.com:STEM-Link-AI-Engineer-3-0-Workspace/level4-lesson3.git
cd level4-lesson3
uv sync
cp .env.example .env    # then paste in your OpenAI and Pinecone keys
```

### Windows (PowerShell)

```powershell
git clone git@github.com:STEM-Link-AI-Engineer-3-0-Workspace/level4-lesson3.git
cd level4-lesson3
uv sync
copy .env.example .env  # then paste in your OpenAI and Pinecone keys
```

For notebook 06, copy `data/my_questions.example.json` to `data/my_questions.json` and add
your ten Lesson 2 questions.

## Running a notebook

From the repo root:

```bash
uv run marimo edit notebooks/00_bench.py
```

## Notebooks

- **00_bench.py** · The bench: a fixed corpus, questions with evidence phrases, and a dense baseline
<!-- notebook-00 -->
<!-- notebook-01 -->
- **02_hybrid.py** · BM25 and its tokenizer, reciprocal rank fusion by hand, then as an `rrf` function
<!-- notebook-02 -->
<!-- notebook-03 -->
<!-- notebook-04 -->
<!-- notebook-05 -->
<!-- notebook-06 -->
<!-- notebook-07 -->

## What is where

| Path | What it is |
|---|---|
| `data/bench/` | The FieldOracle-style bench: 22 documents in three categories, 36 questions with evidence phrases |
| `data/fieldoracle/docs/` | Lesson 2's FieldOracle corpus: the 20 PDFs (`CG-` cultivation guides, `POL-` policies, `FR-` farm records) and `index_corpus.sh` to re-index them |
| `data/fieldoracle/questions.json` | 35 worked example questions with evidence phrases on those PDFs, and 3 they cannot answer |
| `data/helpdesk/` | The breakout corpus: a fictional appliance company's manuals, policies and tickets |
| `scripts/config.py` | Loads `.env`; model and index names |
| `scripts/corpus.py` | Loads the bench, the FieldOracle PDFs (Lesson 2's `pypdf` reader and chunk ids) and the questions; Lesson 2's splitter |

Everything else, including scoring, fusion and reranking, is written out in the notebooks.
| `reports/` | Where notebook 06 writes your retrieval report |
| `docs/retrieval-lab.html` | The Retrieval Lab explorable (concept map, metrics, fusion, rerank cost, chunk budget). Open it in a browser |

All documents in `data/` are a teaching corpus. Figures, circulars and records are
illustrative and fictional; they are not official recommendations.

Reranking calls Pinecone's hosted `bge-reranker-v2-m3`, one request per question. The whole
lesson uses roughly 250 requests; check your plan's rerank quota.
