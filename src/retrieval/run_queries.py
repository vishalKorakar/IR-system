"""
Batch query runner
------------------
Runs all 6 queries against all 3 retrieval models and writes
one TSV file per (query, model) combination:

    results/<query-nr>_structured.tsv
    results/<query-nr>_vsm.tsv
    results/<query-nr>_bm25.tsv

TSV columns: rank, doc_id, score, passage_preview, start_line
"""

import sys
import re
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.structured_search import StructuredSearcher
from retrieval.vsm_search         import VSMSearcher
from retrieval.bm25_search        import BM25Searcher

DB_PATH     = PROJECT_ROOT / "data" / "index" / "index.db"
BOOKS_DIR   = PROJECT_ROOT / "data" / "books"
RESULTS_DIR = PROJECT_ROOT / "results"
PREVIEW_LEN = 200   # max chars in passage preview
TOP_K       = 100   # max results per query per model


# ------------------------------------------------------------
# Query file loader
# ------------------------------------------------------------
def load_queries(path: str) -> List[Tuple[int, str]]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\.\s+(.*)", line)
            if m:
                queries.append((int(m.group(1)), m.group(2).strip()))
            else:
                queries.append((len(queries) + 1, line))
    return queries


# ------------------------------------------------------------
# Passage finder
# ------------------------------------------------------------
def find_passage(doc_id: int, query_terms: List[str]) -> Tuple[Optional[int], str]:
    """
    Find the first line in the raw book file that contains any query term.
    Returns (1-based line number, preview text).
    """
    book_path = BOOKS_DIR / f"{doc_id}.txt"
    if not book_path.exists():
        return None, ""

    patterns = [re.escape(t) for t in query_terms if t.strip()]
    if not patterns:
        return None, ""

    combined = re.compile("|".join(patterns), re.IGNORECASE)
    try:
        with open(book_path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, raw_line in enumerate(f, start=1):
                if combined.search(raw_line):
                    return lineno, raw_line.strip()[:PREVIEW_LEN]
    except OSError:
        pass
    return None, ""


def raw_terms(query_text: str) -> List[str]:
    """Strip field prefixes and quotes; return plain tokens for passage search."""
    text = re.sub(r'\b(title|author|body):', '', query_text, flags=re.IGNORECASE)
    text = text.replace('"', ' ')
    return [t.strip() for t in text.split() if t.strip()]


# ------------------------------------------------------------
# TSV writer for one (query, model) pair
# ------------------------------------------------------------
def write_tsv(
    results:    List[Tuple[int, float, Dict]],
    query_text: str,
    query_id:   int,
    model_name: str,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / f"{query_id}_{model_name}.tsv"
    terms    = raw_terms(query_text)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["rank", "doc_id", "score", "passage_preview", "start_line"])
        for rank, (doc_id, score, _) in enumerate(results, start=1):
            start_line, preview = find_passage(doc_id, terms)
            writer.writerow([
                rank,
                doc_id,
                f"{score:.6f}",
                preview,
                start_line if start_line is not None else "",
            ])

    print(f"    -> {out_path.name}  ({len(results)} results)")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
ALL_MODELS = {
    "structured": StructuredSearcher,
    "vsm":        VSMSearcher,
    "bm25":       BM25Searcher,
}


def run(queries_path: str, model_filter: Optional[str] = None) -> None:
    queries = load_queries(queries_path)

    # Select which models to run
    if model_filter:
        if model_filter not in ALL_MODELS:
            print(f"Error: unknown model '{model_filter}'. Choose from: {', '.join(ALL_MODELS)}")
            sys.exit(1)
        models = {model_filter: ALL_MODELS[model_filter](str(DB_PATH))}
    else:
        models = {name: cls(str(DB_PATH)) for name, cls in ALL_MODELS.items()}

    for qid, query_text in queries:
        print(f"\nQuery {qid}: {query_text}")
        for model_name, searcher in models.items():
            results = searcher.search(query_text, top_k=TOP_K)
            write_tsv(results, query_text, qid, model_name)

    for searcher in models.values():
        searcher.close()

    print(f"\nDone. TSV files written to: {RESULTS_DIR}/")


if __name__ == "__main__":
    # Usage:
    #   python3 run_queries.py                      # all models
    #   python3 run_queries.py structured           # one model
    #   python3 run_queries.py vsm /path/to/q.txt  # one model + custom queries file
    DEFAULT_QUERIES = "/Users/vishalbalasubramanian/Documents/mechaincs_of_search/queries.txt"

    args         = sys.argv[1:]
    model_arg    = None
    queries_file = DEFAULT_QUERIES

    for arg in args:
        if arg in ALL_MODELS:
            model_arg = arg
        elif Path(arg).exists():
            queries_file = arg
        else:
            print(f"Error: '{arg}' is not a valid model name or file path.")
            print(f"Valid models: {', '.join(ALL_MODELS)}")
            sys.exit(1)

    if not Path(queries_file).exists():
        print(f"Error: queries file not found: {queries_file}")
        sys.exit(1)

    run(queries_file, model_filter=model_arg)
