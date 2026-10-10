import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Path setup: lets the notebook import from scripts/ wherever marimo was started.
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
    # 00 · The bench

    FieldOracle answers questions. Is its retrieval any good? If you change the chunk size,
    does it get better or worse? Right now the only way to tell is to ask a few questions and
    squint at the answers.

    A **bench** fixes that: a fixed corpus, a fixed set of questions, and for each question a
    record of what the right passage says. Every change we make today gets run against it.
    """)
    return


@app.cell
def _():
    from scripts.bench import load_corpus

    docs = load_corpus()
    print("documents:", len(docs))
    for doc in docs:
        print(doc.metadata["category"], "|", doc.metadata["title"])
    return (docs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Twenty-two documents in FieldOracle's three classes. Notice the two fertilizer circulars:
    one for 2024 and the one that replaced it in 2025. That is deliberate.

    Now split them exactly as Lesson 2 did: 1000 characters with 150 of overlap.
    """)
    return


@app.cell
def _(docs):
    from scripts.bench import split

    chunks = split(docs, size=1000, overlap=150)
    print("chunks:", len(chunks))
    print(chunks[20].metadata)
    print(chunks[20].page_content)
    return (chunks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The questions

    Each question has one or more **evidence phrases**: short pieces of text copied exactly
    from the document that answers it. A retrieved chunk counts as relevant if it contains
    one of them.

    In Lesson 2 you wrote down the chunk id that answers each of your questions. That breaks
    the moment you change the chunk size, because the ids change. Evidence phrases survive
    re-chunking, which is what lets us tune.
    """)
    return


@app.cell
def _():
    from scripts.bench import load_questions

    questions = load_questions()
    print("questions:", len(questions))
    for kind in ["keyword", "paraphrase", "multi", "filter"]:
        print(kind, len([q for q in questions if q["type"] == kind]))
    print(questions[10])
    return (questions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Four kinds of question:

    - **keyword**: the question names a code, a product or an id (Bg 403, CLM-2025-0412).
    - **paraphrase**: a farmer describing a problem in their own words.
    - **multi**: the answer needs two passages, often from two documents.
    - **filter**: a near-duplicate passage exists that answers a different version of the question.

    ## The baseline retriever

    A vector index over the chunks. We use a small local embedding model (bge-small, 384
    dimensions) and FAISS, so every run is free, offline, and gives everyone the same numbers.
    Your FieldOracle index uses OpenAI embeddings in Pinecone; notebook 06 runs the same
    measurements there.
    """)
    return


@app.cell
def _(chunks):
    from scripts.bench import dense_index

    store = dense_index(chunks)
    dense = store.as_retriever(search_kwargs={"k": 5})
    print("indexed:", store.index.ntotal)
    return (dense,)


@app.cell
def _(dense):
    from scripts.bench import question, show

    q11 = question("q11")
    show(dense.invoke(q11["question"]), q11)
    return question, show


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A star marks a chunk that contains evidence. Now a keyword question: the farmer names the
    exact product.
    """)
    return


@app.cell
def _(dense, question, show):
    q05 = question("q05")
    show(dense.invoke(q05["question"]), q05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No star. The chunk that says *buprofezin 25 SC at 1.5 ml per litre* exists, and the
    embedding model did not find it.

    One question at a time is anecdote. Notebook 01 turns this into numbers.
    """)
    return


if __name__ == "__main__":
    app.run()
