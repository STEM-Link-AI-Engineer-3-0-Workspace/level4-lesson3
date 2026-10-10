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
    # 04 · Tuning, one knob at a time

    Every retrieval system has knobs: chunk size, overlap, how many chunks to send, which
    filters to apply. The rule for turning them is the rule for any experiment:
    **change one thing, measure, keep it or revert it, write it down.**

    We tune on the hybrid retriever because it rebuilds in seconds. The reranker goes back on
    at the end to confirm the winner.
    """)
    return


@app.cell
def _():
    from langchain_classic.retrievers import EnsembleRetriever

    from scripts.bench import bm25, dense_index, evaluate, load_corpus, load_questions, mean, split, summary

    docs = load_corpus()
    questions = load_questions()

    def hybrid_for(chunks, k):
        store = dense_index(chunks)
        return EnsembleRetriever(
            retrievers=[bm25(chunks, k=k), store.as_retriever(search_kwargs={"k": k})],
            weights=[0.5, 0.5],
            id_key="chunk_id",
        )

    baseline = evaluate(hybrid_for(split(docs, 1000, 150), 5), questions)
    summary("baseline 1000/150 k=5", baseline)
    return (
        EnsembleRetriever,
        baseline,
        bm25,
        dense_index,
        docs,
        evaluate,
        hybrid_for,
        mean,
        questions,
        split,
        summary,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Knob 1: chunk size

    Overlap stays at 15 percent of the chunk size, and k stays at 5.
    """)
    return


@app.cell
def _(docs, evaluate, hybrid_for, questions, split, summary):
    for _size in [300, 500, 1000, 2000]:
        _rows = evaluate(hybrid_for(split(docs, _size, int(_size * 0.15)), 5), questions)
        summary("size %d k=5" % _size, _rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Bigger chunks win on recall. Before you set everything to 2000, read the `chars` column.
    Five chunks of 2000 characters is twice as much text as five chunks of 1000, so of course
    more answers are somewhere in it. You are paying for that recall in tokens on every call,
    and a model reading 10,000 characters to find one sentence is more likely to miss it.

    A fair comparison holds the **budget** fixed: how much text the model receives. Here, about
    4000 characters, so smaller chunks get a larger k.
    """)
    return


@app.cell
def _(docs, evaluate, hybrid_for, questions, split, summary):
    for _size, _k in [(300, 13), (500, 8), (1000, 4), (2000, 2)]:
        _rows = evaluate(hybrid_for(split(docs, _size, int(_size * 0.15)), _k), questions, k=_k)
        summary("size %d k=%d" % (_size, _k), _rows, k=_k)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the comparison means something. Pick the size that wins at equal budget, and record
    the decision. Your pick may differ from your neighbour's if the numbers are close; that is
    a sign the knob does not matter much for this corpus.
    """)
    return


@app.cell
def _(docs, evaluate, hybrid_for, questions, split, summary):
    chosen_size = 1000
    chosen_k = 4
    after_size = evaluate(hybrid_for(split(docs, chosen_size, 150), chosen_k), questions, k=chosen_k)
    summary("chosen: size %d k=%d" % (chosen_size, chosen_k), after_size, k=chosen_k)
    return after_size, chosen_k, chosen_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Knob 2: overlap

    Overlap repeats the end of one chunk at the start of the next, so a sentence cut at a
    boundary appears whole somewhere. It also means more chunks and more near-duplicates
    competing for the same top-k slots.
    """)
    return


@app.cell
def _(chosen_k, chosen_size, docs, evaluate, hybrid_for, questions, split, summary):
    for _overlap in [0, int(chosen_size * 0.15), int(chosen_size * 0.3)]:
        _chunks = split(docs, chosen_size, _overlap)
        _rows = evaluate(hybrid_for(_chunks, chosen_k), questions, k=chosen_k)
        summary("overlap %d (%d chunks)" % (_overlap, len(_chunks)), _rows, k=chosen_k)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If overlap does not help, it is not free: keep the smaller value. Write your choice down.

    ## Knob 3: a metadata filter

    The **filter** questions ask about "this season". The corpus still holds the 2024
    circular, which says the opposite on tenancy and land limits. No amount of ranking fixes
    that: both passages really are about the question. The fix is to stop searching the
    superseded document at all.

    The catalog marks it `current: false`. A vector store filters at query time. BM25 here is
    an in-memory index with no filter, so we build it over the current chunks only. In
    Pinecone you would pass `filter={"current": True}` to the search instead.
    """)
    return


@app.cell
def _(
    EnsembleRetriever,
    after_size,
    bm25,
    chosen_k,
    chosen_size,
    dense_index,
    docs,
    evaluate,
    questions,
    split,
):
    from scripts.bench import by_type

    chunks = split(docs, chosen_size, 150)
    current_chunks = [c for c in chunks if c.metadata["current"]]
    store = dense_index(chunks)

    def filtered_for(k):
        return EnsembleRetriever(
            retrievers=[
                bm25(current_chunks, k=k),
                store.as_retriever(search_kwargs={"k": k, "filter": {"current": True}}),
            ],
            weights=[0.5, 0.5],
            id_key="chunk_id",
        )

    after_filter = evaluate(filtered_for(chosen_k), questions, k=chosen_k)
    by_type("before filter", after_size)
    by_type("after filter ", after_filter)
    return after_filter, filtered_for


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the **filter** row with the same row before. A filter is the cheapest fix in this
    lesson, and it only works because someone recorded `current` when the document went in.
    That is a corpus decision, made at indexing time, not a retrieval trick.

    ## Confirm with the reranker

    The winning configuration, with the cross-encoder back on.
    """)
    return


@app.cell
def _(
    after_filter,
    after_size,
    baseline,
    chosen_k,
    evaluate,
    filtered_for,
    mean,
    questions,
    summary,
):
    from langchain_classic.retrievers import ContextualCompressionRetriever

    from scripts.bench import reranker

    final = ContextualCompressionRetriever(base_compressor=reranker(top_n=chosen_k), base_retriever=filtered_for(20))
    final_rows = evaluate(final, questions, k=chosen_k)
    summary("final", final_rows, k=chosen_k)

    print("change log")
    print("baseline 1000/150 k=5: recall", mean(baseline, "recall"), "MRR", mean(baseline, "rr"), "chars", int(mean(baseline, "chars")))
    print("+ chunk size and k: recall", mean(after_size, "recall"), "MRR", mean(after_size, "rr"), "chars", int(mean(after_size, "chars")))
    print("+ current-only filter: recall", mean(after_filter, "recall"), "MRR", mean(after_filter, "rr"), "chars", int(mean(after_filter, "chars")))
    print("+ rerank 20 per side: recall", mean(final_rows, "recall"), "MRR", mean(final_rows, "rr"), "chars", int(mean(final_rows, "chars")))
    return

if __name__ == "__main__":
    app.run()
