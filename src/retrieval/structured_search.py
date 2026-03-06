"""
Structured Search Model
-----------------------
Searches title, author, and body fields using simple TF scoring
with field weighting. No IDF component — distinguishes this from
VSM and BM25.

Score(q, d) = sum_field [ field_weight * sum_t ( tf(t,d,field) / field_len(d) ) ]
"""

import sys
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indexing.indexing import IndexDB
from utils.preprocessing import Preprocessor

FIELD_WEIGHTS = {"title": 3.0, "author": 2.0, "body": 1.0}
VALID_FIELDS  = {"title", "author", "body"}


# ------------------------------------------------------------
# Shared query parser (imported by vsm_search and bm25_search)
# ------------------------------------------------------------
class StructuredQueryParser:
    """
    Parses queries into {field: [terms]}.
    Fielded tokens (title:X) go to that field only.
    Unfielded tokens broadcast to all fields.
    """
    _TOKEN = re.compile(
        r'(\w+):"([^"]*)"'
        r'|(\w+):(\S+)'
        r'|"([^"]*)"'
        r'|(\S+)'
    )

    def parse(self, query: str) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {f: [] for f in VALID_FIELDS}
        for m in self._TOKEN.finditer(query):
            if m.group(1) is not None:
                field, terms = m.group(1).lower(), m.group(2).split()
            elif m.group(3) is not None:
                field, terms = m.group(3).lower(), [m.group(4)]
            elif m.group(5) is not None:
                field, terms = None, m.group(5).split()
            else:
                field, terms = None, [m.group(6)]

            if field in VALID_FIELDS:
                result[field].extend(terms)
            else:
                for f in VALID_FIELDS:
                    result[f].extend(terms)
        return result


# ------------------------------------------------------------
# Structured Searcher — TF scoring
# ------------------------------------------------------------
class StructuredSearcher:

    def __init__(self, db_path: str, use_stemming: bool = False):
        self.db     = IndexDB(db_path)
        self.pre    = Preprocessor(use_stemming=use_stemming)
        self.parser = StructuredQueryParser()

    def _tokenize_query(self, terms: List[str]) -> List[str]:
        """Tokenise for body queries — keeps stopwords, no boilerplate removal."""
        text   = self.pre.normalize_unicode(" ".join(terms))
        text   = self.pre.lowercase(text)
        tokens = self.pre.remove_punctuation(self.pre.tokenize(text))
        if self.pre.use_stemming:
            tokens = self.pre.apply_stemming(tokens)
        return tokens

    def _tf_field(self, field: str, terms: List[str]) -> Dict[int, float]:
        """TF scoring for a single field: score += tf(t,d) / field_len(d)."""
        if field == "body":
            processed = self._tokenize_query(terms)
        else:
            processed = self.pre.preprocess_field(" ".join(terms))

        if not processed:
            return {}

        scores: Dict[int, float] = {}
        for term in processed:
            posting = self.db.get_posting_list(field, term)
            for doc_id, tf in posting.items():
                doc_len = self.db.get_doc_len(doc_id, field)
                if doc_len > 0:
                    scores[doc_id] = scores.get(doc_id, 0.0) + tf / doc_len
        return scores

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float, Dict]]:
        parsed   = self.parser.parse(query)
        combined: Dict[int, float] = {}

        for field, terms in parsed.items():
            if not terms:
                continue
            weight = FIELD_WEIGHTS[field]
            for doc_id, score in self._tf_field(field, terms).items():
                combined[doc_id] = combined.get(doc_id, 0.0) + weight * score

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, score, self.db.get_metadata(doc_id)) for doc_id, score in ranked]

    def close(self):
        self.db.close()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    DB_PATH  = PROJECT_ROOT / "data" / "index" / "index.db"
    searcher = StructuredSearcher(str(DB_PATH))

    print("Structured Search (TF model) — searches title + author + body")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        results = searcher.search(query, top_k=10)
        for rank, (doc_id, score, meta) in enumerate(results, 1):
            print(f"  {rank:2}. [{doc_id}] {meta.get('title','')[:60]}")
            print(f"        {meta.get('author','')[:40]}  score={score:.6f}")
        print()

    searcher.close()
