"""The three retrieval metrics from notebook 01, kept here so later notebooks can import them.

A question carries a list of evidence phrases. A retrieved chunk is relevant if it contains
at least one of them. Labelling by phrase instead of by chunk id means the labels still work
after you change the chunk size.
"""


def is_relevant(doc, question):
    return any(phrase in doc.page_content for phrase in question["evidence"])


def recall_at_k(docs, question, k):
    """Share of the evidence phrases that appear somewhere in the top k chunks."""
    top = [d.page_content for d in docs[:k]]
    found = [p for p in question["evidence"] if any(p in text for text in top)]
    return len(found) / len(question["evidence"])


def precision_at_k(docs, question, k):
    """Share of the top k chunks that are relevant."""
    return sum(is_relevant(d, question) for d in docs[:k]) / k


def reciprocal_rank(docs, question):
    """1 / position of the first relevant chunk, or 0 if none was returned."""
    for position, doc in enumerate(docs, start=1):
        if is_relevant(doc, question):
            return 1 / position
    return 0.0
