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
    # 03 · Reranking

    Hybrid search found the right chunk more often, but not always near the top. Both of its
    retrievers share a limitation: the question and the chunk are turned into numbers
    **separately**, and only then compared. The embedding of a chunk was computed before
    anyone asked anything.

    A **cross-encoder** reads the question and one chunk **together**, as a single input, and
    outputs one relevance score. It can notice that the chunk says *5 SC* when the question
    says *25 SC*. It is far more accurate, and far too slow to run over a whole corpus.

    So we use it on a shortlist: retrieve 20 candidates cheaply, let the cross-encoder
    re-order them, keep the top 5. This is **two-stage retrieval**.
    """)
    return


@app.cell
def _():
    from scripts.bench import (
        bm25,
        by_type,
        dense_index,
        embeddings,
        evaluate,
        load_corpus,
        load_questions,
        misses,
        question,
        reranker,
        show,
        split,
        summary,
    )

    chunks = split(load_corpus(), size=1000, overlap=150)
    questions = load_questions()
    store = dense_index(chunks)
    return (
        bm25,
        by_type,
        chunks,
        embeddings,
        evaluate,
        misses,
        question,
        questions,
        reranker,
        show,
        store,
        summary,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One question, four passages, two kinds of score

    The same four passages scored two ways: cosine similarity between separately computed
    embeddings, and the cross-encoder reading each pair.
    """)
    return


@app.cell
def _(embeddings, question, reranker):
    import numpy as np
    from flashrank import RerankRequest

    q05 = question("q05")
    passages = [
        "Where spraying is needed, use buprofezin 25 SC at 1.5 ml per litre directed at the base of the plants.",
        "Spray hexaconazole 5 SC at 2 ml per litre if more than 10 percent of tillers are affected.",
        "Avoid pyrethroid sprays, which kill the natural enemies of BPH and cause resurgence.",
        "Do not spray the same insecticide more than twice a season.",
    ]

    query_vector = np.array(embeddings().embed_query(q05["question"]))
    passage_vectors = np.array(embeddings().embed_documents(passages))
    cosine = passage_vectors @ query_vector

    request = RerankRequest(query=q05["question"], passages=[{"id": i, "text": p} for i, p in enumerate(passages)])
    cross = reranker().client.rerank(request)
    cross_score = {r["id"]: r["score"] for r in cross}

    print("Q:", q05["question"])
    for i, p in enumerate(passages):
        print("cosine", round(float(cosine[i]), 3), "| cross", round(float(cross_score[i]), 3), "|", p[:60])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cosine scores sit close together: all four passages are about spraying. The cross-encoder
    separates the one that answers the question from the ones that merely share its topic.

    ## Two-stage retrieval

    Stage one is the hybrid retriever from notebook 02, now asked for 20 candidates from each
    side. Stage two is the cross-encoder, keeping the best 5. LangChain calls the second stage
    a *compressor* and the pair a `ContextualCompressionRetriever`.
    """)
    return


@app.cell
def _(bm25, chunks, store):
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

    def hybrid_retriever(candidates):
        return EnsembleRetriever(
            retrievers=[bm25(chunks, k=candidates), store.as_retriever(search_kwargs={"k": candidates})],
            weights=[0.5, 0.5],
            id_key="chunk_id",
        )

    hybrid = hybrid_retriever(5)
    return ContextualCompressionRetriever, hybrid, hybrid_retriever


@app.cell
def _(
    ContextualCompressionRetriever,
    by_type,
    evaluate,
    hybrid,
    hybrid_retriever,
    misses,
    questions,
    reranker,
    summary,
):
    reranked = ContextualCompressionRetriever(
        base_compressor=reranker(top_n=5),
        base_retriever=hybrid_retriever(20),
    )

    hybrid_rows = evaluate(hybrid, questions)
    reranked_rows = evaluate(reranked, questions)
    summary("hybrid         ", hybrid_rows)
    summary("hybrid + rerank", reranked_rows)
    by_type("hybrid         ", hybrid_rows)
    by_type("hybrid + rerank", reranked_rows)
    print("still missed:", misses(reranked_rows))
    return (reranked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read three columns: **MRR** (did the right chunk move to the top?), **recall** (did the
    bigger candidate pool find chunks hybrid-at-5 missed?) and **ms** (what it cost).

    The `ms` column is measured on your laptop's CPU. Post yours in the chat; it will vary
    across the room more than any other number today.
    """)
    return


@app.cell
def _(question, reranked, show):
    q13 = question("q13")
    show(reranked.invoke(q13["question"]), q13)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How many candidates?

    The reranker can only promote what stage one handed it. More candidates means more
    chances to find the answer, and a reranker that reads more pairs.
    """)
    return


@app.cell
def _(
    ContextualCompressionRetriever,
    evaluate,
    hybrid_retriever,
    questions,
    reranker,
    summary,
):
    for _n in [5, 10, 20, 40]:
        _pipeline = ContextualCompressionRetriever(
            base_compressor=reranker(top_n=5),
            base_retriever=hybrid_retriever(_n),
        )
        summary("rerank %d per side" % _n, evaluate(_pipeline, questions))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Somewhere the recall gain flattens while the latency keeps climbing. That knee is the
    setting you want.

    ## Hosted rerankers

    Pinecone hosts cross-encoders too (`bge-reranker-v2-m3`), and `langchain-pinecone` wraps
    it as a compressor, so it drops into the same `ContextualCompressionRetriever`:

    ```python
    from langchain_pinecone import PineconeRerank

    compressor = PineconeRerank(model="bge-reranker-v2-m3", top_n=5)
    ```

    You trade local CPU time for a network call and a monthly request quota. Check your
    plan's rerank limits before putting it in a loop over a question set.
    """)
    return


if __name__ == "__main__":
    app.run()
