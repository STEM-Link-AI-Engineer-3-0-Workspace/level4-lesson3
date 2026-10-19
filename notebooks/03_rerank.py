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
    # 03 · Reranking

    Embeddings turn the question and the chunk into vectors **separately**; the chunk's vector
    was made before anyone asked anything. A **cross-encoder** reads the question and one chunk
    **together** and returns one relevance score. It is more accurate and too slow for a whole
    corpus, so we use it on a shortlist: fetch 20 candidates cheaply, let the cross-encoder
    re-order them, keep the top 5. This is **two-stage retrieval**.

    Pinecone hosts a cross-encoder (`bge-reranker-v2-m3`), so this uses the same key as Lesson 1.
    """)
    return


@app.cell
def _():
    import re
    import time

    from langchain_community.retrievers import BM25Retriever
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from pinecone import Pinecone
    from ranx import Qrels, Run, compare

    from config import EMBED_MODEL, RERANK_MODEL
    from corpus import load_docs, load_questions, split

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    pc = Pinecone()
    chunks = split(load_docs(), size=1000, overlap=150)
    questions = load_questions()
    key = Qrels({
        q["id"]: {c.id: 1 for c in chunks if any(p in c.page_content for p in q["evidence"])}
        for q in questions
    })
    store = InMemoryVectorStore.from_documents(chunks, embeddings)

    def tokenize(text):
        return re.findall(r"[a-z0-9]+(?:[-/.,][a-z0-9]+)*", text.lower())

    bm25 = BM25Retriever.from_documents(chunks, k=20, preprocess_func=tokenize)

    def rrf(rankings, c=60):
        scores, by_id = {}, {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                scores[doc.id] = scores.get(doc.id, 0) + 1 / (c + rank)
                by_id[doc.id] = doc
        return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]

    def run(search, name):
        ranking, chars, start = {}, 0, time.perf_counter()
        for q in questions:
            docs = search(q["question"])
            chars += sum(len(d.page_content) for d in docs)
            ranking[q["id"]] = {d.id: len(docs) - i for i, d in enumerate(docs)}
        ms = (time.perf_counter() - start) * 1000 / len(questions)
        print(name, "| chars per question:", chars // len(questions), "| ms per question:", int(ms))
        return Run(ranking, name=name)
    return (
        RERANK_MODEL,
        bm25,
        compare,
        embeddings,
        key,
        pc,
        questions,
        rrf,
        run,
        store,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One question, four passages, two kinds of score
    """)
    return


@app.cell
def _(RERANK_MODEL, embeddings, pc):
    import numpy as np

    question = "What is the dose of buprofezin 25 SC?"
    passages = [
        "Where spraying is needed, use buprofezin 25 SC at 1.5 ml per litre directed at the base of the plants.",
        "Spray hexaconazole 5 SC at 2 ml per litre if more than 10 percent of tillers are affected.",
        "Avoid pyrethroid sprays, which kill the natural enemies of BPH and cause resurgence.",
        "Do not spray the same insecticide more than twice a season.",
    ]

    cosine = np.array(embeddings.embed_documents(passages)) @ np.array(embeddings.embed_query(question))

    result = pc.inference.rerank(
        model=RERANK_MODEL,
        query=question,
        documents=passages,
        top_n=len(passages),
        return_documents=False,
    )
    cross = {r.index: r.score for r in result.data}

    for i, p in enumerate(passages):
        print("cosine:", round(float(cosine[i]), 3), "| cross-encoder:", round(cross[i], 3), "|", p[:50])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cosine scores sit close together: all four passages are about spraying. The
    cross-encoder separates the one that answers the question.

    ## Two-stage retrieval

    Stage one: hybrid search from notebook 02, 20 candidates per side. Stage two: the
    cross-encoder re-orders them and we keep 5.
    """)
    return


@app.cell
def _(RERANK_MODEL, bm25, pc, rrf, store):
    def candidates(query, n=20):
        return rrf([store.similarity_search(query, k=n), bm25.invoke(query)[:n]])

    def rerank(query, docs, top_n=5):
        result = pc.inference.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[d.page_content for d in docs],
            top_n=top_n,
            return_documents=False,
        )
        return [docs[r.index] for r in result.data]
    return candidates, rerank


@app.cell
def _(candidates, compare, key, rerank, run):
    hybrid = run(lambda q: candidates(q, 5)[:5], "hybrid")
    reranked = run(lambda q: rerank(q, candidates(q, 20)), "hybrid + rerank")
    print(compare(key, runs=[hybrid, reranked], metrics=["recall@5", "precision@5", "mrr@5"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read **MRR** (did the right chunk move up?), **recall** (did the bigger pool find more?)
    and **ms** (what it cost, including a network call to Pinecone). Post your ms in the chat.

    ## How many candidates?

    The reranker can only promote what stage one handed it.
    """)
    return


@app.cell
def _(candidates, compare, key, rerank, run):
    runs = [run(lambda q, n=n: rerank(q, candidates(q, n)), f"rerank {n} per side") for n in [5, 10, 20]]
    print(compare(key, runs=runs, metrics=["recall@5", "mrr@5"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall stops rising somewhere while the time keeps climbing. That knee is your setting.
    Every rerank is one request against your Pinecone plan's quota, so don't put this in a loop
    you run fifty times.
    """)
    return


if __name__ == "__main__":
    app.run()
