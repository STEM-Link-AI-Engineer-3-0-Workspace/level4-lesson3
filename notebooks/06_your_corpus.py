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
    return (root,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 · Your FieldOracle index

    Everything so far ran on a bench corpus with someone else's questions. This notebook runs
    the same measurements on **your** Pinecone index from Lesson 2, with **your** ten
    questions, and writes the result to `reports/retrieval_report.md`. That report is this
    lesson's homework.

    Before running:

    1. `.env` has `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX` if your index is not
       called `fieldoracle`.
    2. Copy `data/my_questions.example.json` to `data/my_questions.json` and put your ten
       questions from the Lesson 2 homework in it, with the ids POST /search gave you. Add at
       least one evidence phrase per answerable question, copied exactly from the chunk that
       answers it (`look` below prints a chunk by its id). Leave `evidence` empty for the two
       your corpus cannot answer.
    """)
    return


@app.cell
def _():
    from dotenv import load_dotenv

    from scripts.pinecone_corpus import connect, fetch_all_chunks

    load_dotenv()
    index, store = connect()
    chunks = fetch_all_chunks(index)
    print("chunks in your index:", len(chunks))
    categories = {}
    for _c in chunks:
        categories[_c.metadata.get("category")] = categories.get(_c.metadata.get("category"), 0) + 1
    print(categories)
    return chunks, store


@app.cell
def _(chunks, root):
    import json

    all_questions = json.loads((root / "data" / "my_questions.json").read_text())
    questions = [q for q in all_questions if q["evidence"]]
    print("answerable:", len(questions), "| unanswerable:", len(all_questions) - len(questions))

    for _q in questions:
        _q.setdefault("type", "mine")
        for _phrase in _q["evidence"]:
            if not any(_phrase in _c.page_content for _c in chunks):
                print("NOT FOUND in any chunk:", _q["id"], "|", _phrase)
    return (questions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fix every `NOT FOUND` line before going on. A test question whose evidence cannot match
    scores zero forever, and you will spend an evening tuning against a typo.

    Copying from a PDF often swaps characters: curly quotes, ligatures such as *ﬁ*, double
    spaces. Copy the phrase from the chunk text printed by `look` below instead of from the PDF.
    """)
    return


@app.cell
def _(chunks):
    def look(record_id):
        """Print a chunk by the id POST /search returned in Lesson 2."""
        for _c in chunks:
            if _c.id == record_id:
                print(_c.metadata.get("source"), "page", _c.metadata.get("page"))
                print(_c.page_content)

    # look("paste-an-id-from-your-lesson-2-homework")
    return


@app.cell
def _(chunks, questions, store):
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

    from scripts.bench import bm25, evaluate, misses, reranker, summary

    dense = store.as_retriever(search_kwargs={"k": 5})
    # No id_key here: in Lesson 2's index chunk_id is a position, repeated in every document,
    # so the two lists are matched on the chunk text instead.
    hybrid = EnsembleRetriever(retrievers=[bm25(chunks, k=5), dense], weights=[0.5, 0.5])
    candidates = EnsembleRetriever(
        retrievers=[bm25(chunks, k=20), store.as_retriever(search_kwargs={"k": 20})],
        weights=[0.5, 0.5],
    )
    reranked = ContextualCompressionRetriever(base_compressor=reranker(top_n=5), base_retriever=candidates)

    results = {
        "dense": evaluate(dense, questions),
        "hybrid": evaluate(hybrid, questions),
        "hybrid + rerank": evaluate(reranked, questions),
    }
    for _name, _rows in results.items():
        summary(_name, _rows)
        print("   missed:", misses(_rows))
    return (results,)


@app.cell
def _():
    from scripts.bench import mean as mean_of
    return (mean_of,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The report

    The cell below writes the numbers. The **Changes** and **Failures** sections are yours:
    make at least one change of your own (a filter, k, the hybrid weights, the candidate
    count), measure it, and diagnose every question that is still missed using the four
    categories from notebook 05.
    """)
    return


@app.cell
def _(mean_of, questions, results, root):
    lines = ["# FieldOracle retrieval report", "", "Questions: %d answerable" % len(questions), ""]
    lines.append("| Retriever | recall@5 | precision@5 | MRR | chars | ms |")
    lines.append("|---|---|---|---|---|---|")
    for _name, _rows in results.items():
        lines.append(
            "| %s | %s | %s | %s | %d | %d |"
            % (_name, mean_of(_rows, "recall"), mean_of(_rows, "precision"), mean_of(_rows, "rr"),
               mean_of(_rows, "chars"), mean_of(_rows, "ms"))
        )
    lines += ["", "## Per question (hybrid + rerank)", "", "| id | recall | reciprocal rank |", "|---|---|---|"]
    for _r in results["hybrid + rerank"]:
        lines.append("| %s | %s | %s |" % (_r["id"], _r["recall"], round(_r["rr"], 2)))
    lines += [
        "",
        "## Changes",
        "",
        "| Change | Metric before | Metric after | Kept? | Why |",
        "|---|---|---|---|---|",
        "| | | | | |",
        "",
        "## Failures",
        "",
        "| id | Fault (query / chunking / index / corpus) | Evidence |",
        "|---|---|---|",
        "| | | |",
    ]
    report = root / "reports" / "retrieval_report.md"
    report.write_text("\n".join(lines) + "\n")
    print("written:", report)
    return


if __name__ == "__main__":
    app.run()
