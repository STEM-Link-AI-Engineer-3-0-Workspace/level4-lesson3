import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    _repo_root = Path.cwd()
    while not (_repo_root / "pyproject.toml").exists():
        _repo_root = _repo_root.parent
    if str(_repo_root / "scripts") not in sys.path:
        sys.path.insert(0, str(_repo_root / "scripts"))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 · Your FieldOracle index

    The same measurements on **your** Pinecone index from Lesson 2 and **your** ten questions.
    The notebook writes `reports/retrieval_report.md`, which is this lesson's homework.

    Before running:

    1. `.env` has `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX` if your index is
       not called `fieldoracle`.
    2. Copy `data/my_questions.example.json` to `data/my_questions.json` and add your ten
       questions from the Lesson 2 homework. Give each answerable one at least one evidence
       phrase, copied exactly from the chunk that answers it. Leave `evidence` empty for the
       two your corpus cannot answer. `data/fieldoracle/questions.json` has 35 worked
       examples on the same PDFs if you want to see what good evidence phrases look like.
    """)
    return


@app.cell
def _():
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    from config import EMBED_MODEL, INDEX_NAME, RERANK_MODEL
    from corpus import load_pdfs, split

    pc = Pinecone()
    store = PineconeVectorStore(index_name=INDEX_NAME, embedding=OpenAIEmbeddings(model=EMBED_MODEL))

    # BM25 needs the text of every chunk. Read the same PDFs you uploaded in Lesson 2, with
    # Lesson 2's reader and splitter, so every chunk gets the id it has in Pinecone.
    chunks = split(load_pdfs(), size=1000, overlap=150)
    print("chunks from the PDFs:", len(chunks))

    # Check: the ids Pinecone returns should be ids we built here.
    _ids = {c.id for c in chunks}
    _found = store.similarity_search("fertiliser subsidy per hectare", k=5)
    print("Pinecone ids we also built:", sum(d.id in _ids for d in _found), "of", len(_found))
    return RERANK_MODEL, chunks, pc, store


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The twenty PDFs are in `data/fieldoracle/docs`, the same files you indexed in Lesson 2.
    `load_pdfs` reads them with Lesson 2's `pypdf` reader, and `split` gives every chunk Lesson
    2's id, `<doc_id>-<n>`, the id POST /search returned. The answer key and the citations use
    those ids.

    If the check above prints fewer than 5 of 5, your index was built with other files or
    another chunk size. Re-index with `bash data/fieldoracle/docs/index_corpus.sh` while your
    Lesson 2 app is running, or ask for help.
    """)
    return


@app.cell
def _(chunks):
    import json

    from ranx import Qrels

    with open("data/my_questions.json") as _f:
        questions = [q for q in json.load(_f) if q["evidence"]]

    key = Qrels({
        q["id"]: {c.id: 1 for c in chunks if any(p in c.page_content for p in q["evidence"])}
        for q in questions
    })
    for q in questions:
        if not key.qrels.get(q["id"]):
            print("NOT FOUND in any chunk:", q["id"], q["evidence"])
    print("answerable questions:", len(questions))
    return key, questions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fix every `NOT FOUND` line before going on. A question whose evidence matches nothing
    scores zero forever. Copying from a PDF often changes quotes, ligatures or spaces, so copy
    from the chunk text instead. `look` prints a chunk by its id.
    """)
    return


@app.cell
def _(chunks):
    def look(chunk_id):
        for c in chunks:
            if c.id == chunk_id:
                print(c.metadata.get("source"), "page", c.metadata.get("page"))
                print(c.page_content)

    # look("paste-an-id-from-your-lesson-2-homework")
    return


@app.cell
def _(RERANK_MODEL, chunks, key, pc, questions, store):
    import re
    import time

    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from ranx import Run, compare

    def tokenize(text):
        return re.findall(r"[a-z0-9]+(?:[-/.,][a-z0-9]+)*", text.lower())


    def rerank(query, docs, top_n=5):
        result = pc.inference.rerank(model=RERANK_MODEL, query=query, documents=[d.page_content for d in docs], top_n=top_n, return_documents=False)
        return [docs[r.index] for r in result.data]

    def run(search, name):
        ranking, chars, start = {}, 0, time.perf_counter()
        for q in questions:
            docs = search(q["question"])
            chars += sum(len(d.page_content) for d in docs)
            ranking[q["id"]] = {d.id: len(docs) - i for i, d in enumerate(docs)}
        ms = (time.perf_counter() - start) * 1000 / len(questions)
        print(name, "| chars per question:", chars // len(questions), "| ms per question:", int(ms))
        return Run(ranking, name=name)

    def hybrid(n):
        """Pinecone's top n and BM25's top n, fused with reciprocal rank fusion."""
        bm25 = BM25Retriever.from_documents(chunks, k=n, preprocess_func=tokenize)
        return EnsembleRetriever(retrievers=[store.as_retriever(search_kwargs={"k": n}), bm25], weights=[0.5, 0.5])

    hybrid_5, hybrid_20 = hybrid(5), hybrid(20)

    runs = [
        run(lambda q: store.similarity_search(q, k=5), "dense"),
        run(lambda q: hybrid_5.invoke(q)[:5], "hybrid"),
        run(lambda q: rerank(q, hybrid_20.invoke(q)), "hybrid + rerank"),
    ]
    report = compare(key, runs=runs, metrics=["recall@5", "precision@5", "mrr@5"])
    print(report)
    return (report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The report

    The cell below writes the table. The **Changes** and **Failures** sections are yours: make
    at least one change of your own (a filter, k, or the candidate count), measure it, and
    diagnose every miss with the four categories from notebook 05.
    """)
    return


@app.cell
def _(report):
    lines = [
        "# FieldOracle retrieval report",
        "",
        "```",
        str(report),
        "```",
        "",
        "## Changes",
        "",
        "| Change | Before | After | Kept? | Why |",
        "|---|---|---|---|---|",
        "| | | | | |",
        "",
        "## Failures",
        "",
        "| Question | Fault (query / chunking / index / corpus) | Evidence |",
        "|---|---|---|",
        "| | | |",
    ]
    with open("reports/retrieval_report.md", "w") as _f:
        _f.write("\n".join(lines) + "\n")
    print("written: reports/retrieval_report.md")
    return


if __name__ == "__main__":
    app.run()
