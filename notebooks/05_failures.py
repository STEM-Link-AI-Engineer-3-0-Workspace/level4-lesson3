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
    # 05 · Reading the failures

    An average tells you whether a change helped. It does not tell you what to change next.
    For that, read the questions the system still misses, one at a time, and decide where
    the fault is:

    | Fault | What you see | Typical fix |
    |---|---|---|
    | **Query** | Reworded, the question would find it | Rewrite or split the query |
    | **Chunking** | The evidence is cut in two, or buried in a chunk about something else | Size, overlap, splitting on headings |
    | **Index** | The chunk is whole and on topic and still ranks low | Tokenizer, embedding model, fusion |
    | **Corpus** | The answer is not there, is contradicted, or needs a calculation | Fix the documents. Not retrieval |
    """)
    return


@app.cell
def _():
    import re

    from langchain_community.retrievers import BM25Retriever
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from pinecone import Pinecone
    from ranx import Qrels, Run, evaluate

    from config import EMBED_MODEL, RERANK_MODEL
    from corpus import load_docs, load_questions, split

    pc = Pinecone()
    chunks = split(load_docs(), 1000, 150)
    questions = load_questions()
    key = Qrels({
        q["id"]: {c.id: 1 for c in chunks if any(p in c.page_content for p in q["evidence"])}
        for q in questions
    })
    store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))

    def tokenize(text):
        return re.findall(r"[a-z0-9]+(?:[-/.,][a-z0-9]+)*", text.lower())

    bm25 = BM25Retriever.from_documents([c for c in chunks if c.metadata["current"]], k=50, preprocess_func=tokenize)

    def rrf(rankings, c=60):
        scores, by_id = {}, {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                scores[doc.id] = scores.get(doc.id, 0) + 1 / (c + rank)
                by_id[doc.id] = doc
        return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]

    def dense_50(query):
        return store.similarity_search(query, k=50, filter=lambda d: d.metadata["current"])

    def candidates(query):
        return rrf([dense_50(query)[:20], bm25.invoke(query)[:20]])

    def best(query, top_n=4):
        """Notebook 04's winner: filtered hybrid, reranked. Returns chunks and scores."""
        docs = candidates(query)
        result = pc.inference.rerank(model=RERANK_MODEL, query=query, documents=[d.page_content for d in docs], top_n=top_n, return_documents=False)
        return [docs[r.index] for r in result.data], [r.score for r in result.data]
    return Run, best, bm25, candidates, chunks, dense_50, evaluate, key, load_questions, questions


@app.cell
def _(Run, best, evaluate, key, questions):
    best_run = Run({q["id"]: {d.id: 4 - i for i, d in enumerate(best(q["question"])[0])} for q in questions}, name="best")
    print(evaluate(key, best_run, ["recall@4", "mrr@4"]))
    misses = [q for q in questions if best_run.scores["recall@4"][q["id"]] < 1]
    print("missed:", [q["id"] for q in misses])
    return (misses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where did the evidence go?

    For each miss, the position of the first evidence chunk in four lists: dense alone and
    BM25 alone (50 deep), the fused candidates, and the final answer. `None` means it never
    appeared.
    """)
    return


@app.cell
def _(best, bm25, candidates, dense_50, misses):
    def position(docs, q):
        for rank, doc in enumerate(docs, start=1):
            if any(p in doc.page_content for p in q["evidence"]):
                return rank
        return None

    for q in misses:
        print(q["id"], q["type"], "|", q["question"])
        print(
            "   dense:", position(dense_50(q["question"]), q),
            "| bm25:", position(bm25.invoke(q["question"]), q),
            "| candidates:", position(candidates(q["question"]), q),
            "| final:", position(best(q["question"])[0], q),
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **High in one list, gone at the end:** fusion or reranking lost it. An index problem.
    - **Low everywhere:** read the chunk before blaming the retriever.
    - **A multi question with one piece found:** one search cannot serve two topics. A query
      problem; Lesson 1's agent searched twice for exactly this reason.

    Print the chunk behind any piece of evidence:
    """)
    return


@app.cell
def _(chunks):
    def look(phrase):
        for c in chunks:
            if phrase in c.page_content:
                print(c.id)
                print(c.page_content)

    look("of which 2.0 ha is paddy")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For each of your misses, post in the chat: **question id, fault, one sentence of evidence.**

    ## Questions with no answer

    A retriever always returns its top k, so for a question the corpus cannot answer it hands
    the model *something*. The reranker's score is the only signal that something is poor.
    """)
    return


@app.cell
def _(best, load_questions, questions):
    for _q in questions[:5] + load_questions(answerable=False):
        _docs, _scores = best(_q["question"], top_n=1)
        print(round(_scores[0], 3), "|", _q["question"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first five have answers; the last three do not. If the unanswerable ones score lower,
    a threshold lets FieldOracle say "the documents do not cover this".

    The Yala 2026 question is about a topic the corpus covers, for a season it does not. It
    probably scores like an answerable question, so no threshold catches it. That one is for
    the grounding prompt and the `found` flag from Lesson 2.
    """)
    return


if __name__ == "__main__":
    app.run()
