"""Read every chunk back out of a Pinecone index, so BM25 can be built over the same text."""

import os

from langchain_core.documents import Document


def connect(index_name=None):
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    index_name = index_name or os.environ.get("PINECONE_INDEX", "fieldoracle")
    # Same embedding model as Lesson 2's FieldOracle, or the query vectors will not match.
    index = Pinecone().Index(index_name)
    store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    return index, store


def fetch_all_chunks(index, namespace=None):
    """List every id in the namespace, fetch the records, and turn them into Documents."""
    docs = []
    for ids in index.list(namespace=namespace):
        records = index.fetch(ids=ids, namespace=namespace).vectors
        for record_id, record in records.items():
            metadata = dict(record.metadata or {})
            text = metadata.pop("text", "")
            # Lesson 2 stores the chunk's position as chunk_id and uses "<doc_id>-<position>"
            # as the record id. The record id is what /search returns and what citations use.
            docs.append(Document(page_content=text, metadata=metadata, id=record_id))
    return docs
