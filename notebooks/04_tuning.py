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
    # 04 · Tuning, one knob at a time

    Chunk size, overlap, how many chunks to send, which filters to apply. The rule for
    turning them is the rule for any experiment: **change one thing, measure, keep it or
    revert it, write it down.**

    We tune on hybrid search because it rebuilds in seconds. The reranker comes back at the
    end to confirm the winner. From here on, LangChain's `EnsembleRetriever` does the fusion:
    it is the same reciprocal rank fusion you wrote in notebook 02 (`1 / (60 + rank)`, summed).
    """)
    return


@app.cell
def _():
    import re
    import time

    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from pinecone import Pinecone
    from ranx import Qrels, Run, compare

    from config import EMBED_MODEL, RERANK_MODEL
    from corpus import load_docs, load_questions, split

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    docs = load_docs()
    questions = load_questions()

    def tokenize(text):
        return re.findall(r"[a-z0-9]+(?:[-/.,][a-z0-9]+)*", text.lower())

    def answer_key(chunks):
        # Chunk ids change with the chunk size, so the answer key is rebuilt for each setting.
        return Qrels({
            q["id"]: {c.id: 1 for c in chunks if any(p in c.page_content for p in q["evidence"])}
            for q in questions
        })

    def run(search, name, qs=questions):
        ranking, chars, start = {}, 0, time.perf_counter()
        for q in qs:
            found = search(q["question"])
            chars += sum(len(d.page_content) for d in found)
            ranking[q["id"]] = {d.id: len(found) - i for i, d in enumerate(found)}
        ms = (time.perf_counter() - start) * 1000 / len(qs)
        print(name, "| chars per question:", chars // len(qs), "| ms per question:", int(ms))
        return Run(ranking, name=name)

    def hybrid_for(chunks, k):
        """Build both indexes over these chunks and return a hybrid search with this k."""
        store = InMemoryVectorStore.from_documents(chunks, embeddings)
        bm25 = BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenize)
        hybrid = EnsembleRetriever(retrievers=[store.as_retriever(search_kwargs={"k": k}), bm25], weights=[0.5, 0.5])
        return lambda q: hybrid.invoke(q)[:k]
    return (
        BM25Retriever,
        EnsembleRetriever,
        InMemoryVectorStore,
        Pinecone,
        Qrels,
        RERANK_MODEL,
        answer_key,
        compare,
        docs,
        embeddings,
        hybrid_for,
        questions,
        run,
        split,
        tokenize,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Knob 1: chunk size

    Overlap stays at 15 percent of the chunk size; k stays at 5. Each setting has its own
    answer key, so each gets its own table row.
    """)
    return


@app.cell
def _(answer_key, compare, docs, hybrid_for, run, split):
    for _size in [300, 500, 1000, 2000]:
        _chunks = split(docs, _size, int(_size * 0.15))
        _run = run(hybrid_for(_chunks, 5), f"size {_size} k=5")
        print(compare(answer_key(_chunks), runs=[_run], metrics=["recall@5", "mrr@5"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Bigger chunks win on recall. Now read the `chars` lines. Five chunks of 2000 characters is
    twice the text of five chunks of 1000, so of course more answers are somewhere in it. You
    pay for that on every call.

    A fair comparison holds the **text budget** fixed, about 3,000 characters here, so smaller
    chunks get a larger k. (Retrieval Lab, station 4, draws this.)
    """)
    return


@app.cell
def _(answer_key, compare, docs, hybrid_for, run, split):
    for _size, _k in [(300, 13), (500, 8), (1000, 4), (2000, 2)]:
        _chunks = split(docs, _size, int(_size * 0.15))
        _run = run(hybrid_for(_chunks, _k), f"size {_size} k={_k}")
        print(compare(answer_key(_chunks), runs=[_run], metrics=[f"recall@{_k}", f"mrr@{_k}"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the comparison means something. Pick a size and a k, and write the decision down.

    ## Knob 2: overlap

    Overlap repeats the end of one chunk at the start of the next, so a sentence cut at a
    boundary appears whole somewhere. It also adds near-duplicates competing for the top k.
    """)
    return


@app.cell
def _(answer_key, compare, docs, hybrid_for, run, split):
    chosen_size, chosen_k = 1000, 4
    for _overlap in [0, 150, 300]:
        _chunks = split(docs, chosen_size, _overlap)
        _run = run(hybrid_for(_chunks, chosen_k), f"overlap {_overlap}")
        print(compare(answer_key(_chunks), runs=[_run], metrics=[f"recall@{chosen_k}", f"mrr@{chosen_k}"]))
    return chosen_k, chosen_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If overlap does not help, keep the smaller value.

    ## Knob 3: a metadata filter

    The **filter** questions ask about "this season". The corpus still holds the 2024 circular,
    which says the opposite on tenancy and land limits. No ranking fixes that: both passages
    really are about the question. The catalog marks the old one `current: false`, so stop
    searching it.

    `InMemoryVectorStore` takes a filter function; in Pinecone you would pass
    `filter={"current": True}`. BM25 has no filter, so we build it over the current chunks only.
    """)
    return


@app.cell
def _(
    BM25Retriever,
    EnsembleRetriever,
    InMemoryVectorStore,
    Qrels,
    answer_key,
    chosen_k,
    chosen_size,
    compare,
    docs,
    embeddings,
    questions,
    run,
    split,
    tokenize,
):
    chunks = split(docs, chosen_size, 150)
    key = answer_key(chunks)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)

    def is_current(doc):
        return doc.metadata["current"]

    def hybrid(bm25_chunks, n, **search_kwargs):
        """Dense search over the shared store plus BM25 over these chunks, top n of each, fused."""
        dense = store.as_retriever(search_kwargs={"k": n, **search_kwargs})
        bm25 = BM25Retriever.from_documents(bm25_chunks, k=n, preprocess_func=tokenize)
        ensemble = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])
        return lambda q: ensemble.invoke(q)[:n]

    unfiltered = hybrid(chunks, chosen_k)
    # The reranker below takes 20 candidates, so build the filtered search at both depths.
    current = [c for c in chunks if c.metadata["current"]]
    filtered = {n: hybrid(current, n, filter=is_current) for n in (chosen_k, 20)}

    filter_qs = [q for q in questions if q["type"] == "filter"]
    filter_key = Qrels({q["id"]: key.qrels[q["id"]] for q in filter_qs})
    before = run(unfiltered, "before filter", filter_qs)
    after = run(filtered[chosen_k], "after filter", filter_qs)
    print("filter questions only")
    print(compare(filter_key, runs=[before, after], metrics=[f"recall@{chosen_k}", f"mrr@{chosen_k}"]))
    return filtered, key


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cheapest fix in the lesson, and it only works because someone recorded `current` when
    the document went in: a corpus decision, not a retrieval trick.

    ## Confirm with the reranker
    """)
    return


@app.cell
def _(Pinecone, RERANK_MODEL, chosen_k, compare, filtered, key, run):
    pc = Pinecone()

    def final(query):
        docs = filtered[20](query)
        result = pc.inference.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[d.page_content for d in docs],
            top_n=chosen_k,
            return_documents=False,
        )
        return [docs[r.index] for r in result.data]

    print(compare(key, runs=[run(filtered[chosen_k], "1000/150 k=4 + filter"), run(final, "+ rerank")], metrics=[f"recall@{chosen_k}", f"mrr@{chosen_k}"]))
    return


if __name__ == "__main__":
    app.run()
