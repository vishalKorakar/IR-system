"""
compute_metrics.py
------------------
Step 2 of evaluation.

Reads the filled-in evaluation/pool.csv (relevance judgements),
then computes the following metrics for each model across all 6 queries:

    - Precision@10  (P@10)
    - MRR           (Mean Reciprocal Rank, searched over top 10)

NDCG is excluded: with binary relevance judgements it adds no
insight beyond P@10, as noted by the module lecturer.

Outputs:
    - Formatted table printed to terminal
    - evaluation/metrics_results.csv  (for report)

Usage:
    python3 compute_metrics.py
"""

import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
EVAL_DIR     = PROJECT_ROOT / "evaluation"

MODELS  = ["structured", "vsm", "bm25"]
TOP_N   = 10   # depth of ranked list (professor recommended top 10)
K       = 10   # cutoff for P@K


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_judgements(pool_path: Path) -> dict:
    """
    Returns:  {query_id: {doc_id: relevance (0 or 1)}}
    Raises ValueError if any relevance cell is left empty.
    """
    judgements = {}
    with open(pool_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid     = int(row["query_id"])
            doc_id  = int(row["doc_id"])
            rel_str = row["relevant"].strip()

            if rel_str == "":
                raise ValueError(
                    f"Missing judgement: query {qid}, doc {doc_id}. "
                    f"Fill every row in pool.csv before running this script."
                )

            rel = int(rel_str)
            if rel not in (0, 1):
                raise ValueError(
                    f"Invalid relevance value '{rel}' for query {qid}, doc {doc_id}. "
                    f"Use only 0 or 1."
                )

            judgements.setdefault(qid, {})[doc_id] = rel

    return judgements


def load_ranked_list(query_id: int, model: str, top_n: int) -> list:
    """Returns list of doc_ids in rank order (1st = most relevant)."""
    path = RESULTS_DIR / f"{query_id}_{model}.tsv"
    if not path.exists():
        return []
    ranked = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["rank"]) > top_n:
                break
            ranked.append(int(row["doc_id"]))
    return ranked


# ------------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------------

def precision_at_k(ranked: list, judgements: dict, k: int = 10) -> float:
    """
    P@K = (number of relevant docs in top-K) / K

    Intuition: of the first K results shown to the user, what
    fraction are actually relevant?
    """
    top_k     = ranked[:k]
    n_relevant = sum(1 for doc_id in top_k if judgements.get(doc_id, 0) == 1)
    return n_relevant / k


def reciprocal_rank(ranked: list, judgements: dict) -> float:
    """
    RR = 1 / rank_of_first_relevant_doc

    Searches the full top-20 list for the first relevant document.
    Returns 0.0 if no relevant document is found in the list.
    """
    for i, doc_id in enumerate(ranked, start=1):
        if judgements.get(doc_id, 0) == 1:
            return 1.0 / i
    return 0.0


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    pool_path = EVAL_DIR / "pool.csv"
    if not pool_path.exists():
        print("ERROR: evaluation/pool.csv not found.")
        print("Run generate_pool.py first, fill in the 'relevant' column, then re-run.")
        return

    try:
        judgements = load_judgements(pool_path)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    query_ids = sorted(judgements.keys())

    # Accumulate per-model scores across queries
    scores = {
        model: {"p@10": [], "mrr": []}
        for model in MODELS
    }

    # ---- Per-query breakdown ----
    header = f"{'Query':<8}" + "".join(
        f"{'P@10':>8}{'MRR':>7}  {m:<12}" for m in MODELS
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    per_query_rows = []   # for CSV output

    for qid in query_ids:
        row_str = f"Q{qid:<7}"
        csv_row = {"query_id": qid}

        for model in MODELS:
            ranked  = load_ranked_list(qid, model, TOP_N)
            j       = judgements[qid]

            p10  = precision_at_k(ranked, j, K)
            mrr  = reciprocal_rank(ranked, j)

            scores[model]["p@10"].append(p10)
            scores[model]["mrr"].append(mrr)

            row_str += f"{p10:>8.3f}{mrr:>7.3f}  {model:<12}"
            csv_row[f"{model}_p@10"] = round(p10, 4)
            csv_row[f"{model}_mrr"]  = round(mrr, 4)

        print(row_str)
        per_query_rows.append(csv_row)

    # ---- Mean across all queries ----
    print("-" * len(header))
    mean_str = f"{'MEAN':<8}"
    mean_row = {"query_id": "MEAN"}

    for model in MODELS:
        n         = len(query_ids)
        p10_mean  = sum(scores[model]["p@10"]) / n
        mrr_mean  = sum(scores[model]["mrr"])  / n

        mean_str += f"{p10_mean:>8.3f}{mrr_mean:>7.3f}  {model:<12}"
        mean_row[f"{model}_p@10"] = round(p10_mean, 4)
        mean_row[f"{model}_mrr"]  = round(mrr_mean, 4)

    print(mean_str)
    print("=" * len(header))

    # ---- Save CSV ----
    os.makedirs(EVAL_DIR, exist_ok=True)
    out_csv = EVAL_DIR / "metrics_results.csv"
    fieldnames = ["query_id"] + [
        f"{m}_{metric}"
        for m in MODELS
        for metric in ["p@10", "mrr"]
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_query_rows)
        writer.writerow(mean_row)

    print(f"\nMetrics saved to: {out_csv}")


if __name__ == "__main__":
    main()
