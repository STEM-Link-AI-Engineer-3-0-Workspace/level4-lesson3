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
    # 01 · Three numbers for a ranked list

    A retriever returns a ranked list. We want to know three different things about it:

    - **Did the answer come back at all?** That is recall.
    - **How much of what came back is useful?** That is precision.
    - **How high up was the first useful thing?** That is reciprocal rank.

    Start with a list you can check by hand. Five results; `True` means relevant.
    The question has two pieces of evidence, and both were found.
    """)
    return


@app.cell
def _():
    ranked = [False, True, False, False, True]
    evidence_total = 2
    top_k = 5

    found = sum(ranked[:top_k])
    recall = found / evidence_total
    precision = found / top_k
    first = ranked.index(True) + 1
    rr = 1 / first

    print("recall@5:", recall)
    print("precision@5:", precision)
    print("first relevant at position:", first)
    print("reciprocal rank:", rr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall is perfect, precision is 0.4, and reciprocal rank is 0.5 because the first useful
    result was second. Three numbers, three different stories about the same list.

    Averaged over many questions, reciprocal rank becomes **MRR** (mean reciprocal rank).

    For real chunks, "relevant" means "contains an evidence phrase". The three functions are
    short enough to read in one go:
    """)
    return


@app.cell
def _():
    def is_relevant(doc, question):
        return any(phrase in doc.page_content for phrase in question["evidence"])

    def recall_at_k(docs, question, k):
        top = [d.page_content for d in docs[:k]]
        found = [p for p in question["evidence"] if any(p in text for text in top)]
        return len(found) / len(question["evidence"])

    def precision_at_k(docs, question, k):
        return sum(is_relevant(d, question) for d in docs[:k]) / k

    def reciprocal_rank(docs, question):
        for position, doc in enumerate(docs, start=1):
            if is_relevant(doc, question):
                return 1 / position
        return 0.0
    return precision_at_k, recall_at_k, reciprocal_rank


@app.cell
def _():
    from scripts.bench import dense_index, load_corpus, load_questions, question, show, split

    chunks = split(load_corpus(), size=1000, overlap=150)
    store = dense_index(chunks)
    questions = load_questions()
    return chunks, question, questions, show, store


@app.cell
def _(precision_at_k, question, recall_at_k, reciprocal_rank, show, store):
    q21 = question("q21")
    docs = store.similarity_search(q21["question"], k=5)
    show(docs, q21)
    print("recall@5:", recall_at_k(docs, q21, 5))
    print("precision@5:", precision_at_k(docs, q21, 5))
    print("reciprocal rank:", reciprocal_rank(docs, q21))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now every question. The same three functions live in `scripts/metrics.py`, and
    `evaluate` runs them over the whole set. Post your recall@5 and MRR in the chat.
    """)
    return


@app.cell
def _(questions, store):
    from scripts.bench import by_type, evaluate, misses, summary

    dense = store.as_retriever(search_kwargs={"k": 5})
    dense_rows = evaluate(dense, questions, k=5)
    summary("dense", dense_rows)
    by_type("dense", dense_rows)
    print("missed:", misses(dense_rows))
    return by_type, evaluate, misses, summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two things to notice.

    1. **Precision@5 is low and that is fine.** Most questions have one relevant chunk, so the
       best possible precision@5 is 0.2. Precision only means something next to how many
       relevant chunks exist.
    2. **The misses are not random.** Look at which type they belong to.

    ## What k does

    Send more chunks and recall goes up. So does what you pay: every chunk is tokens in the
    prompt, on every call.
    """)
    return


@app.cell
def _(evaluate, questions, store, summary):
    for _k in [1, 3, 5, 10]:
        _rows = evaluate(store.as_retriever(search_kwargs={"k": _k}), questions, k=_k)
        summary("dense k=%d" % _k, _rows, k=_k)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall climbs, precision falls, and `chars` (how much text the model will have to read)
    grows in step with k. There is no free setting. Every later change today is judged
    against this baseline at k = 5.
    """)
    return


if __name__ == "__main__":
    app.run()
