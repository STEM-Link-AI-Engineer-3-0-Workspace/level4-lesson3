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
    # 00 · The bench

    **Where we are.** In Lesson 2, FieldOracle answered a farmer's question in four steps:
    take the question, search the index for matching chunks, keep the top 5, and let the model
    write the answer from only those chunks. Today is only about step 2, the search.

    **One word for today: retriever.** A retriever is that search step written as a function:
    a question goes in, a ranked list of chunks comes out, best match first. You have used one
    already: `store.similarity_search(question, k=5)` from Lessons 1 and 2.

    If the right chunk is not in that list, the model cannot answer correctly, however good the
    prompt is. So: how good is the list, and how do we make it better? Asking a few questions
    and reading the answers cannot tell you.

    A **bench** can: a fixed set of documents, a fixed set of questions, and for each question
    a record of what the right passage says. In this notebook we load all three and run the
    retriever you already know.
    """)
    return


@app.cell
def _():
    from corpus import load_docs, load_questions, split

    docs = load_docs()
    print("documents:", len(docs))
    for doc in docs:
        print(doc.metadata["category"], "|", doc.metadata["title"])
    return docs, load_questions, split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Twenty-two documents in FieldOracle's three classes. Notice the two fertilizer
    circulars, one for 2024 and the one that replaced it in 2025. That is on purpose.

    Split them the way Lesson 2 did: 1000 characters with 150 of overlap.
    """)
    return


@app.cell
def _(docs, split):
    chunks = split(docs, size=1000, overlap=150)
    print("chunks:", len(chunks))
    print("id:", chunks[20].id)
    print(chunks[20].page_content)
    return (chunks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The questions

    Each question has one or more **evidence phrases**, copied exactly from the document
    that answers it. A chunk is relevant if it contains one of them. Chunk ids change every
    time you change the chunk size; the phrases do not.
    """)
    return


@app.cell
def _(load_questions):
    questions = load_questions()
    print("questions:", len(questions))
    for kind in ["keyword", "paraphrase", "multi", "filter"]:
        print(kind, len([q for q in questions if q["type"] == kind]))
    print(questions[10])
    return (questions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **keyword**: names a code or product (Bg 403, CLM-2025-0412)
    - **paraphrase**: a farmer describing a problem in their own words
    - **multi**: needs two passages
    - **filter**: a near-duplicate passage answers a different version of the question

    The **answer key** lists, for each question, the chunk ids that contain evidence.
    `ranx`, the library we score with in notebook 01, calls it `Qrels`.
    """)
    return


@app.cell
def _(chunks, questions):
    from ranx import Qrels

    key = Qrels({
        q["id"]: {c.id: 1 for c in chunks if any(p in c.page_content for p in q["evidence"])}
        for q in questions
    })
    print("q11 relevant chunks:", key.qrels["q11"])
    print("q21 relevant chunks:", key.qrels["q21"])
    return (key,)


@app.cell
def _(key):
    print(key)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The baseline: search by meaning

    The retriever you already know: embed the question, return the nearest chunks. This is
    called **dense retrieval**. Same embeddings as Lessons 1 and 2, in a vector store held in
    memory so it can be rebuilt in seconds when we change the chunk size. `InMemoryVectorStore` has the same
    methods as `PineconeVectorStore`.
    """)
    return


@app.cell
def _(chunks):
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    from config import EMBED_MODEL

    store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))
    print("stored:", len(store.store))
    return (store,)


@app.cell
def _(questions, store):
    def show(docs, q):
        """Print a ranked list. A star marks a chunk that contains evidence."""
        print("Q:", q["question"])
        for rank, doc in enumerate(docs, start=1):
            star = "*" if any(p in doc.page_content for p in q["evidence"]) else " "
            print(rank, star, doc.id, "|", " ".join(doc.page_content.split())[:80])

    q11 = next(q for q in questions if q["id"] == "q11")
    show(store.similarity_search(q11["question"], k=5), q11)
    return (show,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A star marks a chunk with evidence. Now a keyword question: the farmer names the exact
    product. Before running it, ask the room: will embedding search find it?
    """)
    return


@app.cell
def _(questions, show, store):
    q05 = next(q for q in questions if q["id"] == "q05")
    show(store.similarity_search(q05["question"], k=5), q05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at where the star is. With OpenAI's embeddings this one usually comes back first:
    the model has seen enough product names to place *buprofezin* near *buprofezin*. Other
    codes in the bench (claim numbers, plot ids, circular numbers) are rarer, and that is where
    embeddings start to slip.

    One question at a time is anecdote: q05 working tells you nothing about the other
    thirty-two. Notebook 01 counts.
    """)
    return


if __name__ == "__main__":
    app.run()
