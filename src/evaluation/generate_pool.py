"""
generate_pool.py
----------------
Step 1 of evaluation.

Pools the top-20 results from all 3 models for each of the 6 queries,
removes duplicates, and writes a CSV template:

    evaluation/pool.csv

Columns: query_id, query_text, doc_id, passage_preview, relevant

YOU must open pool.csv and fill the 'relevant' column:
    1 = document is relevant to the query
    0 = document is not relevant to the query

Then run compute_metrics.py to get P@10, NDCG@10, MRR.
"""

import csv
import re
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
EVAL_DIR     = PROJECT_ROOT / "evaluation"

QUERIES_PATH = "/Users/vishalbalasubramanian/Documents/mechaincs_of_search/queries.txt"
MODELS       = ["structured", "vsm", "bm25"]
TOP_N        = 20


def load_queries(path: str) -> dict:
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\.\s+(.*)", line)
            if m:
                queries[int(m.group(1))] = m.group(2).strip()
    return queries


def load_top_n(query_id: int, model: str, top_n: int) -> list:
    path = RESULTS_DIR / f"{query_id}_{model}.tsv"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping.")
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["rank"]) > top_n:
                break
            rows.append({
                "doc_id":          int(row["doc_id"]),
                "passage_preview": row.get("passage_preview", ""),
            })
    return rows


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    queries = load_queries(QUERIES_PATH)

    pool_path = EVAL_DIR / "pool.csv"

    with open(pool_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_text", "doc_id", "passage_preview", "relevant"])

        for qid in sorted(queries.keys()):
            query_text = queries[qid]
            seen       = {}   # doc_id -> passage_preview (first seen)

            for model in MODELS:
                for item in load_top_n(qid, model, TOP_N):
                    doc_id = item["doc_id"]
                    if doc_id not in seen:
                        seen[doc_id] = item["passage_preview"]

            for doc_id, preview in seen.items():
                writer.writerow([qid, query_text, doc_id, preview, ""])

            print(f"Query {qid} ({query_text!r:40s}): {len(seen):3d} unique docs to judge")

    print(f"\nPool written to: {pool_path}")
    print("="*60)
    print("NEXT STEP:")
    print("  Open evaluation/pool.csv")
    print("  For each row, read the passage_preview")
    print("  Fill the 'relevant' column with:")
    print("    1 = relevant to the query")
    print("    0 = not relevant to the query")
    print("  Save the file, then run: python3 compute_metrics.py")
    print("="*60)


if __name__ == "__main__":
    main()
