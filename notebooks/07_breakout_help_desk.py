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
    # 07 · Breakout: the Paravi help desk

    Paravi sells air conditioners, washing machines and refrigerators. Its support team wants
    a retrieval system over nine documents: three product manuals, five policy pages and a
    log of last month's support tickets. Twenty questions from real customers come with
    evidence phrases.

    Four teams will use the same documents in four different products. **Your group has one
    of them.** Read only your card.

    | Group | Product | What counts | Limit |
    |---|---|---|---|
    | A | Hotline voice bot: reads one passage aloud | The **first** passage is right (hit@1) | Under 80 ms per question on your laptop |
    | B | Technician tablet: technicians type codes | **MRR** over the top 5, on the keyword questions only | None |
    | C | Warranty disputes desk: a person reads everything returned | **recall@6**: nothing relevant may be missed | At most 6 passages |
    | D | Shop phone app with a small on-device model | **recall** of what fits in the context | At most 1,200 characters per question |

    Your job in 25 minutes:

    1. Build at least three different retrievers from the parts below.
    2. Score each with `score(retriever, k)`.
    3. Pick one. Post in the chat: **group, configuration, your metric, one sentence on why.**
    """)
    return


@app.cell
def _():
    GROUP = "A"
    return (GROUP,)


@app.cell
def _():
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

    from scripts.bench import bm25, dense_index, evaluate, load_corpus, load_questions, mean, reranker, split

    docs = load_corpus("helpdesk")
    questions = load_questions("helpdesk")
    chunks = split(docs, size=400, overlap=60)
    store = dense_index(chunks)
    print("documents:", len(docs), "| chunks:", len(chunks), "| questions:", len(questions))
    return (
        ContextualCompressionRetriever,
        EnsembleRetriever,
        bm25,
        chunks,
        dense_index,
        docs,
        evaluate,
        mean,
        questions,
        reranker,
        split,
        store,
    )


@app.cell
def _(GROUP, evaluate, mean, questions):
    def score(retriever, k):
        """Score a retriever by your group's rule. Prints the metric, the latency and the text size."""
        qs = [q for q in questions if q["type"] == "keyword"] if GROUP == "B" else questions
        rows = evaluate(retriever, qs, k=k)
        hit_at_1 = round(sum(r["rr"] == 1 for r in rows) / len(rows), 2)
        metric = {
            "A": ("hit@1", hit_at_1),
            "B": ("MRR@%d" % k, mean(rows, "rr")),
            "C": ("recall@%d" % k, mean(rows, "recall")),
            "D": ("recall@%d" % k, mean(rows, "recall")),
        }[GROUP]
        print("group", GROUP, "|", metric[0], metric[1], "| ms", int(mean(rows, "ms")), "| chars", int(mean(rows, "chars")))
        return metric[1]
    return (score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The parts

    - `store.as_retriever(search_kwargs={"k": k})`: dense
    - `bm25(chunks, k=k)`: keyword
    - `EnsembleRetriever(retrievers=[...], weights=[...], id_key="chunk_id")`: hybrid
    - `ContextualCompressionRetriever(base_compressor=reranker(top_n=n), base_retriever=...)`: rerank
    - `split(docs, size, overlap)` then `dense_index(new_chunks)` and `bm25(new_chunks, k)`: re-chunk

    One configuration is done for you. Add yours below it.
    """)
    return


@app.cell
def _(score, store):
    score(store.as_retriever(search_kwargs={"k": 5}), k=5)
    return


@app.cell
def _(bm25, chunks, score):
    # A second example. Add your own configurations in new cells below.
    score(bm25(chunks, k=5), k=5)
    return


if __name__ == "__main__":
    app.run()
