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
    # 07 · Breakout: the Paravi help desk

    Paravi sells air conditioners, washing machines and refrigerators. Its support team wants
    retrieval over nine documents: three manuals, five policy pages and last month's ticket
    log. Twenty customer questions come with evidence phrases.

    Four teams use the same documents in four different products. **Your group has one.**
    Read only your row.

    | Group | Product | What counts | Limit |
    |---|---|---|---|
    | A | Hotline voice bot: reads one passage aloud | `mrr@1` (the first passage is right) | Under 80 ms per question |
    | B | Technician tablet: technicians type codes | `mrr@5`, keyword questions only | None |
    | C | Warranty disputes desk: a person reads everything | `recall@6` | At most 6 passages |
    | D | Shop phone app with a small on-device model | `recall` at your k | At most 1,200 characters per question |

    In 25 minutes: build at least three search setups, score each, pick one, and post in the
    chat: **group, setup, your metric, one sentence on why.**
    """)
    return


@app.cell
def _():
    GROUP = "A"
    return (GROUP,)


@app.cell
def _(GROUP):
    import re
    import time

    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from pinecone import Pinecone
    from ranx import Qrels, Run, compare

    from config import EMBED_MODEL, RERANK_MODEL
    from corpus import load_docs, load_questions, split

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    pc = Pinecone()
    docs = load_docs("helpdesk")
    questions = load_questions("helpdesk")
    if GROUP == "B":
        questions = [q for q in questions if q["type"] == "keyword"]

    def tokenize(text):
        return re.findall(r"[a-z0-9]+(?:[-/.,][a-z0-9]+)*", text.lower())

    def make_hybrid(store, chunks, k):
        """Dense top k and BM25 top k, fused with reciprocal rank fusion."""
        bm25 = BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenize)
        return EnsembleRetriever(retrievers=[store.as_retriever(search_kwargs={"k": k}), bm25], weights=[0.5, 0.5])

    def rerank(query, found, top_n):
        result = pc.inference.rerank(model=RERANK_MODEL, query=query, documents=[d.page_content for d in found], top_n=top_n, return_documents=False)
        return [found[r.index] for r in result.data]

    def score(search, name, k, chunks):
        """Run every question and score it by your group's rule."""
        ranking, chars, start = {}, 0, time.perf_counter()
        for q in questions:
            found = search(q["question"])
            chars += sum(len(d.page_content) for d in found)
            ranking[q["id"]] = {d.id: len(found) - i for i, d in enumerate(found)}
        ms = (time.perf_counter() - start) * 1000 / len(questions)
        key = Qrels({
            q["id"]: {c.id: 1 for c in chunks if any(p in c.page_content for p in q["evidence"])}
            for q in questions
        })
        metric = {"A": "mrr@1", "B": "mrr@5", "C": "recall@6", "D": f"recall@{k}"}[GROUP]
        print(name, "| chars per question:", chars // len(questions), "| ms per question:", int(ms))
        print(compare(key, runs=[Run(ranking, name=name)], metrics=[metric]))
    return (
        BM25Retriever,
        InMemoryVectorStore,
        docs,
        embeddings,
        make_hybrid,
        rerank,
        score,
        split,
        tokenize,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The parts

    - `split(docs, size, overlap)`: chunks
    - `InMemoryVectorStore.from_documents(chunks, embeddings)` then `store.similarity_search(q, k=k)`: dense
    - `BM25Retriever.from_documents(chunks, k=20, preprocess_func=tokenize)` then `bm25.invoke(q)[:k]`: keyword
    - `hybrid = make_hybrid(store, chunks, k)` once, then `hybrid.invoke(q)[:k]`: hybrid, LangChain's
      `EnsembleRetriever` doing the reciprocal rank fusion from notebook 02
    - `rerank(q, candidates, top_n=k)`: cross-encoder
    - `score(search_function, "a name", k, chunks)`: your group's number

    One setup is done for you. Add yours in new cells.
    """)
    return


@app.cell
def _(InMemoryVectorStore, docs, embeddings, make_hybrid, score, split):
    chunks = split(docs, size=400, overlap=60)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    hybrid = make_hybrid(store, chunks, 5)

    score(lambda q: store.similarity_search(q, k=5), "dense 400/60 k=5", 5, chunks)
    score(lambda q: hybrid.invoke(q)[:5], "hybrid 400/60 k=5", 5, chunks)
    return


if __name__ == "__main__":
    app.run()
