#!/bin/bash
# Indexes every PDF in this folder into FieldOracle, with the category taken from the file prefix.
# Usage: bash index_corpus.sh [base_url]     (default http://localhost:8000)
# macOS and Linux. On Windows, run it from Git Bash, or upload through /docs.

BASE="${1:-http://localhost:8000}"
cd "$(dirname "$0")"

for file in *.pdf; do
  case "$file" in
    CG-*)  category="cultivation_guide" ;;
    POL-*) category="policy" ;;
    FR-*)  category="farm_record" ;;
    *)     echo "skip $file (unknown prefix)"; continue ;;
  esac
  echo "indexing $file as $category"
  curl -s -X POST "$BASE/documents" -F "file=@$file" -F "category=$category"
  echo
done
