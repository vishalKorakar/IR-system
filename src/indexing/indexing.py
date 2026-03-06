import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, Counter
import os
import csv
import sqlite3
import array as array_module

# ------------------------------------------------------------
# Setup project root (IR-system/)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.preprocessing import Preprocessor

# Number of documents processed before flushing rows to SQLite.
# Keeps peak RAM bounded to ~500 docs at a time.
BATCH_SIZE = 500


# ------------------------------------------------------------
# IndexDB — read interface used at search time
# ------------------------------------------------------------
class IndexDB:
    """
    Wraps the SQLite database and exposes the posting list lookups
    that the ranking/search code needs.

    All IR logic (BM25, VSM, phrase search) lives in the search code.
    This class is purely a storage accessor.
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def get_posting_list(self, field: str, term: str) -> Dict[int, int]:
        """
        Returns {doc_id: freq} for the given term in the given field.
        field must be one of: 'body', 'title', 'author'
        """
        rows = self.conn.execute(
            f"SELECT doc_id, freq FROM {field}_index WHERE term = ?", (term,)
        )
        return {doc_id: freq for doc_id, freq in rows}

    def get_positions(self, term: str, doc_id: int) -> List[int]:
        """
        Returns the list of positions for a term inside a body document.
        Used for phrase search — checks whether query terms appear
        consecutively in the document.
        """
        row = self.conn.execute(
            "SELECT positions FROM body_index WHERE term = ? AND doc_id = ?",
            (term, doc_id),
        ).fetchone()
        if row is None or row[0] is None:
            return []
        a = array_module.array('I')
        a.frombytes(row[0])
        return a.tolist()

    def get_doc_len(self, doc_id: int, field: str = "body") -> int:
        """Returns the token length of a document field."""
        row = self.conn.execute(
            f"SELECT {field}_len FROM doc_lengths WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row[0] if row else 0

    def get_avgdl(self, field: str = "body") -> float:
        """Returns the average document length across the corpus for a field."""
        row = self.conn.execute(
            f"SELECT AVG({field}_len) FROM doc_lengths"
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def get_metadata(self, doc_id: int) -> Dict[str, str]:
        """Returns title, author, language for a document."""
        row = self.conn.execute(
            "SELECT title, author, language FROM metadata WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()
        if row is None:
            return {}
        return {"title": row[0], "author": row[1], "language": row[2]}

    def doc_count(self) -> int:
        """Total number of indexed documents."""
        row = self.conn.execute("SELECT COUNT(*) FROM doc_lengths").fetchone()
        return row[0] if row else 0

    def vocab_size(self, field: str = "body") -> int:
        """Number of unique terms in the given field index."""
        row = self.conn.execute(
            f"SELECT COUNT(DISTINCT term) FROM {field}_index"
        ).fetchone()
        return row[0] if row else 0

    def close(self):
        self.conn.close()


# ------------------------------------------------------------
# Schema
# ------------------------------------------------------------
def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS body_index (
            term      TEXT    NOT NULL,
            doc_id    INTEGER NOT NULL,
            freq      INTEGER NOT NULL,
            positions BLOB    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS title_index (
            term   TEXT    NOT NULL,
            doc_id INTEGER NOT NULL,
            freq   INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS author_index (
            term   TEXT    NOT NULL,
            doc_id INTEGER NOT NULL,
            freq   INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS doc_lengths (
            doc_id     INTEGER PRIMARY KEY,
            body_len   INTEGER NOT NULL,
            title_len  INTEGER NOT NULL,
            author_len INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metadata (
            doc_id   INTEGER PRIMARY KEY,
            title    TEXT,
            author   TEXT,
            language TEXT
        );
    """)


def _create_term_indexes(conn: sqlite3.Connection) -> None:
    """
    Create B-tree indexes on the term column of each index table.
    Done AFTER all rows are inserted — building indexes on an
    already-populated table is significantly faster than maintaining
    them incrementally during insert.
    """
    print("Creating term lookup indexes...")
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_body_term   ON body_index(term);
        CREATE INDEX IF NOT EXISTS idx_title_term  ON title_index(term);
        CREATE INDEX IF NOT EXISTS idx_author_term ON author_index(term);
    """)


# ------------------------------------------------------------
# Metadata loading
# ------------------------------------------------------------
def load_metadata_csv(metadata_csv_path: str) -> Dict[int, Dict[str, str]]:
    meta: Dict[int, Dict[str, str]] = {}

    with open(metadata_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if (row.get("has_text") or "").strip().upper() != "TRUE":
                continue

            try:
                doc_id = int(row.get("gutenberg_id"))
            except (ValueError, TypeError):
                continue

            meta[doc_id] = {
                "title":    row.get("title",    "") or "",
                "author":   row.get("author",   "") or "",
                "language": row.get("language", "") or "",
            }

    return meta


# ------------------------------------------------------------
# Batch flush to SQLite
# ------------------------------------------------------------
def _flush(
    conn: sqlite3.Connection,
    body_rows:    list,
    title_rows:   list,
    author_rows:  list,
    doc_len_rows: list,
    meta_rows:    list,
) -> None:
    """
    Write all accumulated rows to SQLite in a single transaction.
    Using executemany + one commit is far faster than committing
    row by row.
    """
    conn.executemany(
        "INSERT INTO body_index(term, doc_id, freq, positions) VALUES (?,?,?,?)",
        body_rows,
    )
    conn.executemany(
        "INSERT INTO title_index(term, doc_id, freq) VALUES (?,?,?)",
        title_rows,
    )
    conn.executemany(
        "INSERT INTO author_index(term, doc_id, freq) VALUES (?,?,?)",
        author_rows,
    )
    conn.executemany(
        "INSERT INTO doc_lengths(doc_id, body_len, title_len, author_len) VALUES (?,?,?,?)",
        doc_len_rows,
    )
    conn.executemany(
        "INSERT INTO metadata(doc_id, title, author, language) VALUES (?,?,?,?)",
        meta_rows,
    )
    conn.commit()


# ------------------------------------------------------------
# Main indexing routine
# ------------------------------------------------------------
def build_indexes(
    books_dir:          str,
    metadata_csv_path:  str,
    db_path:            str,
    use_stemming:       bool = False,
    limit_docs:         Optional[int] = None,
) -> None:
    """
    Build the inverted index and write it incrementally to a SQLite database.

    Peak RAM usage is bounded by BATCH_SIZE docs at a time.
    The full index is never held in memory — rows are streamed to disk
    as each document is processed.

    Body index stores term frequencies AND positions (for phrase search).
    Title/author indexes store term frequencies only.
    """
    # Remove any existing database so we start with a clean slate.
    # Re-running the indexer is always a full rebuild.
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    # WAL mode allows reads during writes and is faster for bulk inserts
    conn.execute("PRAGMA journal_mode = WAL")
    # NORMAL is safe (no data loss on OS crash) and much faster than FULL
    conn.execute("PRAGMA synchronous = NORMAL")
    # 64 MB page cache to reduce disk I/O during inserts
    conn.execute("PRAGMA cache_size = -65536")

    _create_tables(conn)

    pre      = Preprocessor(use_stemming=use_stemming)
    metadata = load_metadata_csv(metadata_csv_path)

    # In-memory row buffers — flushed every BATCH_SIZE docs
    body_rows:    list = []
    title_rows:   list = []
    author_rows:  list = []
    doc_len_rows: list = []
    meta_rows:    list = []

    processed = 0

    for filename in sorted(os.listdir(books_dir)):
        if not filename.endswith(".txt"):
            continue

        base = filename[:-4]
        if not base.isdigit():
            continue

        doc_id = int(base)

        if doc_id not in metadata:
            continue

        file_path = os.path.join(books_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except OSError:
            continue

        # -------------------------------------------------------
        # BODY — positions stored for phrase search
        # -------------------------------------------------------
        body_tokens = pre.preprocess_body(raw_text)

        if not body_tokens:
            continue

        # Build term → positions map in one pass
        pos_map: Dict[str, List[int]] = defaultdict(list)
        for pos, term in enumerate(body_tokens):
            pos_map[term].append(pos)

        for term, positions in pos_map.items():
            # array.array('I') = unsigned int, 4 bytes per position
            # ~7x more compact than a Python list of ints
            pos_blob = array_module.array('I', positions).tobytes()
            body_rows.append((term, doc_id, len(positions), pos_blob))

        body_len = len(body_tokens)

        # -------------------------------------------------------
        # TITLE — freq only (no phrase search on title)
        # -------------------------------------------------------
        title_tokens = pre.preprocess_field(metadata[doc_id]["title"])
        title_len = 0
        if title_tokens:
            for term, freq in Counter(title_tokens).items():
                title_rows.append((term, doc_id, freq))
            title_len = len(title_tokens)

        # -------------------------------------------------------
        # AUTHOR — freq only
        # -------------------------------------------------------
        author_tokens = pre.preprocess_field(metadata[doc_id]["author"])
        author_len = 0
        if author_tokens:
            for term, freq in Counter(author_tokens).items():
                author_rows.append((term, doc_id, freq))
            author_len = len(author_tokens)

        doc_len_rows.append((doc_id, body_len, title_len, author_len))
        meta_rows.append((
            doc_id,
            metadata[doc_id]["title"],
            metadata[doc_id]["author"],
            metadata[doc_id]["language"],
        ))

        processed += 1

        # Flush batch to disk and clear buffers
        if processed % BATCH_SIZE == 0:
            _flush(conn, body_rows, title_rows, author_rows, doc_len_rows, meta_rows)
            body_rows    = []
            title_rows   = []
            author_rows  = []
            doc_len_rows = []
            meta_rows    = []
            print(f"Indexed and flushed {processed} documents...")

        if limit_docs and processed >= limit_docs:
            break

    # Flush the final partial batch
    if doc_len_rows:
        _flush(conn, body_rows, title_rows, author_rows, doc_len_rows, meta_rows)
        print(f"Indexed and flushed {processed} documents...")

    # Build term lookup indexes after all data is inserted
    _create_term_indexes(conn)
    conn.close()

    print(f"\nIndexing complete. {processed} documents written to: {db_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    BOOKS_DIR    = PROJECT_ROOT / "data" / "books"
    METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
    DB_PATH      = PROJECT_ROOT / "data" / "index" / "index.db"

    os.makedirs(DB_PATH.parent, exist_ok=True)

    build_indexes(
        books_dir=str(BOOKS_DIR),
        metadata_csv_path=str(METADATA_CSV),
        db_path=str(DB_PATH),
        use_stemming=False,
        limit_docs=None,  # set to a small number for testing e.g. 500
    )

    # Quick stats via IndexDB
    db = IndexDB(str(DB_PATH))
    print(f"Documents indexed : {db.doc_count()}")
    print(f"Body vocab size   : {db.vocab_size('body')}")
    print(f"Avg body length   : {db.get_avgdl('body'):.2f}")
    db.close()
