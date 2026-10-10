import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys
    import pathlib

    root = pathlib.Path.cwd()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    sys.path.insert(0, str(root))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 · Hybrid search

    Dense retrieval missed most of the **keyword** questions. An embedding model turns
    *buprofezin 25 SC* into a point near other pesticide talk; the exact product name and
    dose are blurred away.

    Search engines solved exact matching decades ago with **BM25**: score a chunk by how many
    of the query's words it contains, giving more weight to rare words and less to long chunks.
    It does not know that *drain* and *dry out* mean similar things, but it never confuses
    *25 SC* with *5 SC*.
    """)
    return


@app.cell
def _():
    from scripts.bench import (
        by_type,
        dense_index,
        evaluate,
        load_corpus,
        load_questions,
        misses,
        question,
        show,
        split,
        summary,
    )

    chunks = split(load_corpus(), size=1000, overlap=150)
    questions = load_questions()
    dense = dense_index(chunks).as_retriever(search_kwargs={"k": 5})
    dense_rows = evaluate(dense, questions)
    summary("dense", dense_rows)
    return (
        by_type,
        chunks,
        dense,
        dense_rows,
        evaluate,
        misses,
        question,
        questions,
        show,
        summary,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BM25, and why its tokenizer matters

    BM25 compares words, so first it has to decide what a word is. LangChain's
    `BM25Retriever` defaults to splitting on spaces. Watch what that does to a claim number
    at the end of a question.
    """)
    return


@app.cell
def _(chunks, question, show):
    from langchain_community.retrievers import BM25Retriever

    q03 = question("q03")
    print("default words:", q03["question"].split()[-3:])
    bm25_default = BM25Retriever.from_documents(chunks, k=5)
    show(bm25_default.invoke(q03["question"]), q03)
    return BM25Retriever, q03


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `CLM-2025-0412?` with the question mark attached matches nothing in the corpus, and
    neither does `What` versus `what`. A tokenizer that lowercases and keeps codes whole
    fixes both. It is in `scripts/bench.py`; here it is in action:
    """)
    return


@app.cell
def _(BM25Retriever, chunks, q03, show):
    from scripts.bench import tokenize

    print("our words:", tokenize(q03["question"])[-3:])
    bm25 = BM25Retriever.from_documents(chunks, k=5, preprocess_func=tokenize)
    show(bm25.invoke(q03["question"]), q03)
    return (bm25,)


@app.cell
def _(bm25, by_type, dense_rows, evaluate, misses, questions, summary):
    bm25_rows = evaluate(bm25, questions)
    summary("dense", dense_rows)
    summary("bm25 ", bm25_rows)
    by_type("dense", dense_rows)
    by_type("bm25 ", bm25_rows)
    print("dense missed:", misses(dense_rows))
    print("bm25 missed:", misses(bm25_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each one wins a different type of question and misses a different set. That is the
    whole argument for hybrid search: the two are wrong in different places.

    ## Combining two rankings

    Their scores cannot be added: a BM25 score of 9.3 and a cosine of 0.71 are on unrelated
    scales. **Reciprocal rank fusion** ignores the scores and uses only positions. Each list
    gives a chunk `1 / (60 + rank)`, and the chunk's fused score is the sum.

    Here it is by hand for one question.
    """)
    return


@app.cell
def _(bm25, dense, question):
    from scripts.metrics import is_relevant

    q11 = question("q11")
    dense_docs = dense.invoke(q11["question"])
    bm25_docs = bm25.invoke(q11["question"])
    print("dense:", [d.metadata["chunk_id"] for d in dense_docs])
    print("bm25:", [d.metadata["chunk_id"] for d in bm25_docs])

    fused = {}
    by_id = {}
    for ranking in [dense_docs, bm25_docs]:
        for rank, doc in enumerate(ranking, start=1):
            chunk_id = doc.metadata["chunk_id"]
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (60 + rank)
            by_id[chunk_id] = doc

    for chunk_id in sorted(fused, key=fused.get, reverse=True)[:5]:
        mark = "*" if is_relevant(by_id[chunk_id], q11) else " "
        print(round(fused[chunk_id], 4), mark, chunk_id)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A chunk that appears in **both** lists collects two shares and rises. A chunk only one
    retriever found keeps one share. The 60 stops the first place from dominating.

    LangChain's `EnsembleRetriever` does exactly this, with an optional weight per retriever.
    """)
    return


@app.cell
def _(bm25, by_type, dense, evaluate, misses, questions, summary):
    from langchain_classic.retrievers import EnsembleRetriever

    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5], id_key="chunk_id")
    hybrid_rows = evaluate(hybrid, questions)
    summary("hybrid", hybrid_rows)
    by_type("hybrid", hybrid_rows)
    print("hybrid missed:", misses(hybrid_rows))
    return (EnsembleRetriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall is the best of the three. Look at MRR, though: it is not. Fusion found the right
    chunk more often but does not always put it first, because a chunk that only one retriever
    liked has to compete with chunks both retrievers ranked in the middle. Notebook 03 deals
    with order.

    ## The weights are a dial

    Weights tilt the fusion toward one retriever. Try them and see which question types move.
    One result will look odd: work out why before reading the note below the cell.
    """)
    return


@app.cell
def _(EnsembleRetriever, bm25, dense, evaluate, questions, summary):
    for _w in [0.3, 0.5, 0.7]:
        _hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[_w, 1 - _w], id_key="chunk_id")
        summary("bm25 weight %.1f" % _w, evaluate(_hybrid, questions))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With weight 0.3 the result is identical to dense alone. When the two top-5 lists barely
    overlap, every dense chunk scores 0.7/(60+r) and every BM25 chunk 0.3/(60+r), so all the
    dense chunks sort first. With short lists, unequal weights act like a switch.

    ## In production

    Your FieldOracle index is in Pinecone. Pinecone can store a keyword (sparse or full-text)
    representation next to the dense vector, run both searches, and you fuse the two rankings
    with the same RRF formula you just wrote. The idea is identical; the BM25 index just
    lives on the server instead of in memory. Notebook 06 does it the simple way: BM25 in
    memory over the chunks fetched from your index.
    """)
    return


if __name__ == "__main__":
    app.run()
