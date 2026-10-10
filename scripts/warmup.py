"""Pre-work: download the two local models once, so class starts without a 200 MB wait.

    uv run python -m scripts.warmup
"""

import time

from scripts.bench import EMBEDDING_MODEL, RERANK_MODEL, embeddings, reranker

start = time.perf_counter()
vector = embeddings().embed_query("Which paddy variety resists brown planthopper?")
print("embedding model:", EMBEDDING_MODEL, "| dimensions:", len(vector))

from langchain_core.documents import Document

docs = [Document(page_content="Bg 366 is moderately resistant to brown planthopper."),
        Document(page_content="Drain the field 10 to 14 days before harvest.")]
ranked = reranker(top_n=2).compress_documents(docs, "Which paddy variety resists brown planthopper?")
print("rerank model:", RERANK_MODEL, "| top passage:", ranked[0].page_content)
print("ready in", round(time.perf_counter() - start, 1), "seconds")
