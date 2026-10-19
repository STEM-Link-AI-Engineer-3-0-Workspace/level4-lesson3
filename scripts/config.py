"""Shared setup. The only file here that is not a lesson concept.

Same shape as Lesson 1's config.py: loads .env and holds the model and index names.
"""

import os
import warnings

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", module="ranx")   # ranx's number-type warnings are noise here

EMBED_MODEL = "text-embedding-3-small"   # the model Lessons 1 and 2 used
RERANK_MODEL = "bge-reranker-v2-m3"      # a cross-encoder Pinecone hosts for you

INDEX_NAME = os.getenv("PINECONE_INDEX", "fieldoracle")   # your Lesson 2 index

