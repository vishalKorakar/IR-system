"""
BM25 Retrieval Model
--------------------
Searches title, author, and body fields using the BM25 formula
with field weighting.

BM25(t, d, field) = IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*dl/avgdl))
IDF(t)            = log((N - df + 0.5) / (df + 0.5) + 1)
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indexing.indexing import IndexDB
from utils.preprocessing import Preprocessor
from retrieval.structured_search import StructuredQueryParser

K1 = 1.5
B  = 0.75
FIELD_WEIGHTS = {"title": 3.0, "author": 2.0, "body": 1.0}


class BM25Searcher:

    def __init__(self, db_path: str, use_stemming: bool = False):
        self.db     = IndexDB(db_path)
        self.pre    = Preprocessor(use_stemming=use_stemming)
        self.parser = StructuredQueryParser()
        self._N     = self.db.doc_count()

    def _tokenize_query(self, terms: List[str]) -> List[str]:
        """Body query tokeniser — keeps stopwords, no boilerplate removal."""
        text   = self.pre.normalize_unicode(" ".join(terms))
        text   = self.pre.lowercase(text)
        tokens = self.pre.remove_punctuation(self.pre.tokenize(text))
        if self.pre.use_stemming:
            tokens = self.pre.apply_stemming(tokens)
        return tokens

    def _bm25_field(self, field: str, terms: List[str]) -> Dict[int, float]:
        """BM25 scoring for a single field."""
        if field == "body":
            processed = self._tokenize_query(terms)
        else:
            processed = self.pre.preprocess_field(" ".join(terms))

        if not processed:
            return {}

        avgdl = self.db.get_avgdl(field)
        if avgdl == 0:
            return {}

        scores: Dict[int, float] = {}
        for term in processed:
            posting = self.db.get_posting_list(field, term)
            if not posting:
                continue
            df  = len(posting)
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in posting.items():
                dl      = self.db.get_doc_len(doc_id, field)
                norm_tf = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * norm_tf

        return scores

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float, Dict]]:
        parsed   = self.parser.parse(query)
        combined: Dict[int, float] = {}

        for field, terms in parsed.items():
            if not terms:
                continue
            weight = FIELD_WEIGHTS[field]
            for doc_id, score in self._bm25_field(field, terms).items():
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
    searcher = BM25Searcher(str(DB_PATH))

    print("BM25 Search — searches title + author + body")
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
            print(f"        {meta.get('author','')[:40]}  score={score:.4f}")
        print()

    searcher.close()
