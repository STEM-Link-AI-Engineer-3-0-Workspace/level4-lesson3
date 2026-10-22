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
    # 02 · Hybrid search

    Look back at notebook 01's by-kind numbers. Dense search does worst on some kinds of
    question, usually the ones with codes in them: an embedding places *CLM-2025-0412* near
    other claim talk, and the exact number blurs away.

    The other way to search, from the start of the lesson, is **search by words**. Its standard
    version is called **BM25**, the classic keyword ranking: score a chunk by the question's words it contains,
    with rare words counting more. It does not know that *drain* and *dry out* are related, but
    it never confuses *25 SC* with *5 SC*.
    """)
    return


@app.cell
def _():
    import time

    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from ranx import Qrels, Run, compare

    from config import EMBED_MODEL
    from corpus import load_docs, load_questions, split

    chunks = split(load_docs(), size=1000, overlap=150)
    questions = load_questions()
    key = Qrels({
        q["id"]: {c.id: 1 for c in chunks if any(p in c.page_content for p in q["evidence"])}
        for q in questions
    })
    store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))

    def run(search, name):
        """Notebook 01's loop: every question through search, as a ranx Run."""
        ranking, chars, start = {}, 0, time.perf_counter()
        for q in questions:
            docs = search(q["question"])
            chars += sum(len(d.page_content) for d in docs)
            ranking[q["id"]] = {d.id: len(docs) - i for i, d in enumerate(docs)}
        ms = (time.perf_counter() - start) * 1000 / len(questions)
        print(name, "| chars per question:", chars // len(questions), "| ms per question:", int(ms))
        return Run(ranking, name=name)

    def show(docs, q):
        print("Q:", q["question"])
        for rank, doc in enumerate(docs, start=1):
            star = "*" if any(p in doc.page_content for p in q["evidence"]) else " "
            print(rank, star, doc.id, "|", " ".join(doc.page_content.split())[:80])

    dense = run(lambda q: store.similarity_search(q, k=5), "dense")
    return chunks, compare, dense, key, questions, run, show, store


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BM25, and why its tokenizer matters

    BM25 compares words, so it first has to decide what a word is. LangChain's
    `BM25Retriever` splits on spaces by default. Watch the claim number at the end of q03.
    """)
    return


@app.cell
def _(chunks, questions, show):
    from langchain_community.retrievers import BM25Retriever

    q03 = next(q for q in questions if q["id"] == "q03")
    print("words:", q03["question"].split()[-3:])
    show(BM25Retriever.from_documents(chunks, k=5).invoke(q03["question"]), q03)
    return BM25Retriever, q03


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `CLM-2025-0412?` keeps its question mark and matches nothing. A tokenizer that lowercases
    and keeps codes whole fixes it:
    """)
    return


@app.cell
def _(BM25Retriever, chunks, q03, show):
    import re

    def tokenize(text):
        """Lowercase words, keeping codes like k-17, 25,000 and agr/fs/2025/07 whole."""
        return re.findall(r"[a-z0-9]+(?:[-/.,][a-z0-9]+)*", text.lower())

    print("words:", tokenize(q03["question"])[-3:])
    bm25 = BM25Retriever.from_documents(chunks, k=5, preprocess_func=tokenize)
    show(bm25.invoke(q03["question"]), q03)
    return (bm25,)


@app.cell
def _(bm25, compare, dense, key, questions, run):
    from ranx import evaluate

    keyword = run(bm25.invoke, "bm25")
    print(compare(key, runs=[dense, keyword], metrics=["recall@5", "mrr@5"]))

    for _r in [dense, keyword]:
        evaluate(key, _r, ["recall@5"])
        for _kind in ["keyword", "paraphrase", "multi", "filter"]:
            _ids = [q["id"] for q in questions if q["type"] == _kind]
            print(_r.name, _kind, "| recall@5:", round(sum(_r.scores["recall@5"][i] for i in _ids) / len(_ids), 2))
    return (keyword,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back to the cold open

    On the first slides, A was search by meaning (dense) and B was search by words (BM25).
    Here are those two questions again, run through both.
    """)
    return


@app.cell
def _(bm25, questions, show, store):
    for _qid in ["q07", "q14"]:
        _q = next(q for q in questions if q["id"] == _qid)
        print("A, dense")
        show(store.similarity_search(_q["question"], k=3), _q)
        print("B, bm25")
        show(bm25.invoke(_q["question"])[:3], _q)
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the two blocks of rows. The kinds where one does well and the other does badly are
    the argument for using both: they are wrong in different places. (On this bench BM25
    usually leads on **keyword** and dense on **paraphrase**; check that it holds on yours.)

    ## Combining two rankings

    The scores cannot be added: BM25 gives 9.3, cosine gives 0.71, different scales.
    **Reciprocal rank fusion** uses positions only. Each list gives a chunk `1 / (60 + rank)`,
    and the chunk's fused score is the sum. Here it is for one question:
    """)
    return


@app.cell
def _(bm25, questions, show, store):
    q11 = next(q for q in questions if q["id"] == "q11")
    dense_docs = store.similarity_search(q11["question"], k=5)
    bm25_docs = bm25.invoke(q11["question"])

    fused = {}
    for ranking in [dense_docs, bm25_docs]:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc.id] = fused.get(doc.id, 0) + 1 / (60 + rank)

    for chunk_id in sorted(fused, key=fused.get, reverse=True)[:5]:
        print(round(fused[chunk_id], 4), chunk_id)
    print()
    show(bm25_docs, q11)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A chunk in **both** lists collects two shares and rises. A chunk only one list found keeps
    one share. Open the **Retrieval Lab, station 2** and move the depth slider from 5 to 6 to
    watch the answer jump to first place.

    The same loop as a function, so we can run it for every question:
    """)
    return


@app.cell
def _(bm25, compare, dense, key, keyword, run, store):
    def rrf(rankings, c=60):
        scores, by_id = {}, {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                scores[doc.id] = scores.get(doc.id, 0) + 1 / (c + rank)
                by_id[doc.id] = doc
        return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]

    hybrid = run(lambda q: rrf([store.similarity_search(q, k=5), bm25.invoke(q)])[:5], "hybrid")
    print(compare(key, runs=[dense, keyword, hybrid], metrics=["recall@5", "precision@5", "mrr@5"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall is the best of the three. MRR is not: fusion found the answer more often but does
    not always put it first. Notebook 03 fixes the order.

    In production, Pinecone can run a keyword search next to the dense one, and you fuse the
    two lists with this same formula.
    """)
    return


if __name__ == "__main__":
    app.run()
