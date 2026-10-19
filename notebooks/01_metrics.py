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
    # 01 · Three numbers for a ranked list

    A retriever returns a ranked list. Three questions about it:

    - **Recall@k**: of the relevant chunks, how many are in the top k?
    - **Precision@k**: of the top k, how many are relevant?
    - **MRR**: how high is the first relevant chunk? (1 for first place, 1/2 for second, ...,
      averaged over questions)

    We don't write these ourselves. `ranx` is a standard library for ranking metrics: give it
    the answer key (`Qrels`) and what a retriever returned (`Run`), and ask for metrics by name.

    First a list you can check by hand: five results, the 2nd and 5th are relevant.
    """)
    return


@app.cell
def _():
    from ranx import Qrels, Run, compare, evaluate

    toy_key = Qrels({"q": {"b": 1, "e": 1}})
    toy_run = Run({"q": {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}}, name="toy")   # higher score = higher rank
    print(compare(toy_key, runs=[toy_run], metrics=["recall@5", "precision@5", "mrr@5"]))
    print(compare(toy_key, runs=[toy_run], metrics=["recall@1", "precision@1", "mrr@1"]))
    return Qrels, Run, compare, evaluate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At k = 5: both relevant chunks are there (recall 1.0), 2 of 5 are relevant (precision 0.4),
    and the first one is second (MRR 0.5). At k = 1 the top result is wrong, so all three are 0.

    `compare` prints a table of any number of runs side by side; we use it all session.

    Now the bench.
    """)
    return


@app.cell
def _(Qrels):
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    from config import EMBED_MODEL
    from corpus import load_docs, load_questions, split

    chunks = split(load_docs(), size=1000, overlap=150)
    questions = load_questions()
    key = Qrels({
        q["id"]: {c.id: 1 for c in chunks if any(p in c.page_content for p in q["evidence"])}
        for q in questions
    })
    store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))
    return key, questions, store


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To build a `Run`, send every question through the search and record the ranking. We also
    note how much text came back and how long it took; both matter later.
    """)
    return


@app.cell
def _(Run, questions, store):
    import time

    ranking, chars, start = {}, 0, time.perf_counter()
    for q in questions:
        docs = store.similarity_search(q["question"], k=10)
        chars += sum(len(d.page_content) for d in docs)
        ranking[q["id"]] = {d.id: len(docs) - i for i, d in enumerate(docs)}
    ms = (time.perf_counter() - start) * 1000 / len(questions)

    dense = Run(ranking, name="dense")
    print("chars per question:", chars // len(questions), "| ms per question:", int(ms))
    return (dense,)


@app.cell
def _(compare, dense, key):
    print(compare(key, runs=[dense], metrics=["recall@5", "precision@5", "mrr@5"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Post your recall@5 and MRR in the chat.

    Precision@5 is low and that is fine: most questions have one or two relevant chunks, so
    it cannot go much above 0.2 to 0.4.

    ## Which questions does it miss?

    After `evaluate`, ranx keeps the score of every single question in `run.scores`.
    Average them by kind of question:
    """)
    return


@app.cell
def _(dense, evaluate, key, questions):
    evaluate(key, dense, ["recall@5", "mrr@5"])
    for kind in ["keyword", "paraphrase", "multi", "filter"]:
        ids = [q["id"] for q in questions if q["type"] == kind]
        recall = sum(dense.scores["recall@5"][i] for i in ids) / len(ids)
        mrr = sum(dense.scores["mrr@5"][i] for i in ids) / len(ids)
        print(kind, "| recall@5:", round(recall, 2), "| MRR:", round(mrr, 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The misses are not random. Note which kind does worst; notebook 02 is about it.

    ## What k does

    The run kept 10 results, so every cut-off can be read from it at once.
    """)
    return


@app.cell
def _(compare, dense, key):
    print(compare(key, runs=[dense], metrics=["recall@1", "recall@3", "recall@5", "recall@10", "precision@1", "precision@5", "precision@10"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall climbs and precision falls. Every extra chunk is text the model reads, and pays
    for, on every call. There is no free setting; from here on we compare at k = 5.
    """)
    return


if __name__ == "__main__":
    app.run()
