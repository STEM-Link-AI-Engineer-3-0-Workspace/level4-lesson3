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
    # 05 · Reading the failures

    An average tells you whether a change helped. It does not tell you what to change next.
    For that you read the questions the system still gets wrong, one at a time, and decide
    where the fault is:

    | Fault | What you see | Typical fix |
    |---|---|---|
    | **Query** | The evidence ranks well if the question is worded differently | Rewrite or split the query before searching |
    | **Chunking** | The evidence is cut across two chunks, or buried in a chunk about something else | Chunk size, overlap, splitting on headings |
    | **Index** | The evidence chunk is whole and on topic, and both retrievers still rank it low | Tokenizer, embedding model, hybrid weights |
    | **Corpus** | The answer is not there, is contradicted, or needs a calculation | Fix the documents or the metadata; not a retrieval problem |
    """)
    return


@app.cell
def _():
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

    from scripts.bench import (
        bm25,
        dense_index,
        evaluate,
        load_corpus,
        load_questions,
        misses,
        question,
        rank_of_evidence,
        reranker,
        split,
        summary,
    )

    chunks = split(load_corpus(), 1000, 150)
    current_chunks = [c for c in chunks if c.metadata["current"]]
    store = dense_index(chunks)
    questions = load_questions()

    dense_deep = store.as_retriever(search_kwargs={"k": 50, "filter": {"current": True}})
    bm25_deep = bm25(current_chunks, k=50)
    candidates = EnsembleRetriever(
        retrievers=[bm25(current_chunks, k=20), store.as_retriever(search_kwargs={"k": 20, "filter": {"current": True}})],
        weights=[0.5, 0.5],
        id_key="chunk_id",
    )
    best = ContextualCompressionRetriever(base_compressor=reranker(top_n=4), base_retriever=candidates)

    rows = evaluate(best, questions, k=4)
    summary("best", rows, k=4)
    print("missed:", misses(rows))
    return (
        ContextualCompressionRetriever,
        best,
        bm25_deep,
        candidates,
        chunks,
        dense_deep,
        misses,
        question,
        rank_of_evidence,
        rows,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where did the evidence go?

    For each miss, find the evidence chunk's position in four lists: dense alone and BM25
    alone (searched 50 deep), the fused candidate pool, and the final reranked answer.
    `None` means it never appeared.
    """)
    return


@app.cell
def _(best, bm25_deep, candidates, dense_deep, misses, question, rank_of_evidence, rows):
    for _qid in misses(rows):
        _q = question(_qid)
        print(_qid, _q["type"], "|", _q["question"])
        print(
            "   dense:", rank_of_evidence(dense_deep, _q),
            "| bm25:", rank_of_evidence(bm25_deep, _q),
            "| candidates:", rank_of_evidence(candidates, _q),
            "| final:", rank_of_evidence(best, _q),
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How to read it:

    - **Ranked high in one list but missing from the final answer**: the information was there
      and fusion or reranking lost it. An index-side problem.
    - **Low in every list**: look at the chunk itself before blaming the retriever.
    - **A multi question with only one piece found**: one search cannot serve two topics. That
      is a query problem; Lesson 1's agent searched twice for exactly this reason.

    `look` prints the chunk that holds a piece of evidence, so you can judge it.
    """)
    return


@app.cell
def _(chunks, question):
    def look(qid):
        _q = question(qid)
        for _phrase in _q["evidence"]:
            for _c in chunks:
                if _phrase in _c.page_content:
                    print(_c.metadata["chunk_id"], "| evidence:", _phrase)
                    print(_c.page_content)
                    print("---")
                    break

    look("q22")
    return (look,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Change the id in the cell above to each of your misses. For each one, write a line in the
    chat: **question id, fault category, one-sentence reason**. We compare answers before
    moving on; some will be contested, and that is the point.

    ## The questions with no answer

    Three questions in the set have no answer in the corpus. A retriever always returns its
    top k, so it will hand the model *something* for them. The reranker's score is the only
    signal that the something is poor.
    """)
    return


@app.cell
def _(candidates, question, rows):
    from scripts.bench import reranker as _reranker

    _scorer = _reranker(top_n=1)

    def top_score(text):
        _docs = _scorer.compress_documents(candidates.invoke(text), text)
        return round(float(_docs[0].metadata["relevance_score"]), 3) if _docs else 0.0

    _answerable = sorted(top_score(question(r["id"])["question"]) for r in rows)
    print("answerable, lowest five top scores:", _answerable[:5])
    print("answerable, median top score:", _answerable[len(_answerable) // 2])
    for _qid in ["q34", "q35", "q36"]:
        print(_qid, top_score(question(_qid)["question"]), "|", question(_qid)["question"])
    return (top_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the unanswerable questions score below most answerable ones, a threshold lets
    FieldOracle say "the documents do not cover this" before the model ever sees a passage.
    `FlashrankRerank(score_threshold=...)` returns nothing below the threshold.

    Look at `q36` (the Yala 2026 grant). It is about a real topic in the corpus, just a season
    the corpus does not cover, so it probably scores like an answerable question. A score
    threshold cannot catch it. That one is for the grounding prompt and the `found` flag you
    built in Lesson 2.
    """)
    return


if __name__ == "__main__":
    app.run()
