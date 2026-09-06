"""Content-addressed embedding cache.

The thing that makes twenty experiment configurations affordable on a laptop.
A re-run re-embeds only what actually changed; change the chunk size and only
the chunks whose *text* moved are recomputed, because chunks whose text is
byte-identical under two chunker settings hit the same cache entry.

**Deliberate deviation from PLAN.md M2.** The plan specifies the key as
``sha256(model_name + model_revision + chunker_config + chunk_text)``. We drop
``chunker_config``. An embedding is a pure function of (model, revision,
pre/post-processing, text) — the chunker that produced the text has no influence
on the vector. Including it would mean that changing chunk size from 256 to 320
invalidates every entry, including the many chunks whose text is unchanged,
which defeats the stated purpose of the cache. What *does* belong in the key,
and is easy to miss, is the encoder's prefix and normalisation settings: an E5
model with and without ``"query: "`` produces different vectors from the same
text and must never share an entry. That is folded into
:attr:`~reval.embedding.encoder.Encoder.identity`, which forms the namespace.

Storage is a packed float32 blob plus a SQLite index, rather than one file per
vector. FiQA at 57,638 documents produces well over 100,000 chunks; that many
small files is slow to write and slower to enumerate, particularly on Windows.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reval.paths import cache_dir


@dataclass(frozen=True, slots=True)
class CacheStats:
    """What a build reports, and what the M2 acceptance criterion checks."""

    requested: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requested if self.requested else 0.0

    def __str__(self) -> str:
        return (
            f"{self.requested:,} chunks: {self.hits:,} cached, "
            f"{self.misses:,} embedded ({self.hit_rate:.1%} hit rate)"
        )


class EmbeddingCache:
    """Vectors keyed by ``sha256(encoder identity + text)``.

    Not thread-safe, and deliberately not: the runner embeds in one process, and
    a lock-free design that is correct only under that assumption is better than
    a locking one that pretends otherwise.
    """

    def __init__(self, namespace: str, dim: int, root: Path | None = None):
        self.namespace = namespace
        self.dim = dim
        self.root = (root or cache_dir()) / "embeddings" / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.root / "vectors.f32"
        self.db_path = self.root / "index.sqlite"
        self._db = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        self._db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS vectors (
                key TEXT PRIMARY KEY,
                row INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            );
            """
        )
        cur = self._db.execute("SELECT v FROM meta WHERE k = 'dim'")
        found = cur.fetchone()
        if found is None:
            self._db.execute("INSERT INTO meta (k, v) VALUES ('dim', ?)", (str(self.dim),))
            self._db.commit()
        elif int(found[0]) != self.dim:
            # A dimension change means the namespace collided with a different
            # model. Serving those vectors would corrupt an index silently.
            raise ValueError(
                f"cache {self.root} holds {found[0]}-dim vectors but this encoder is "
                f"{self.dim}-dim; delete the directory or fix the namespace"
            )

    @staticmethod
    def make_key(identity: str, text: str) -> str:
        """``sha256(encoder identity || text)``.

        The NUL separator prevents the boundary ambiguity where a different
        (identity, text) split produces the same concatenation.
        """
        h = hashlib.sha256()
        h.update(identity.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])

    def _next_row(self) -> int:
        size = self.vectors_path.stat().st_size if self.vectors_path.exists() else 0
        stride = self.dim * 4
        if size % stride:
            raise ValueError(
                f"{self.vectors_path} is {size} bytes, not a multiple of {stride}; "
                "the blob was truncated — delete the cache directory to rebuild"
            )
        return size // stride

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        """Fetch every cached vector among ``keys``. Missing keys are absent."""
        if not keys:
            return {}
        rows: dict[str, int] = {}
        # SQLite caps variables per statement (999 on older builds); chunk the IN.
        for batch in _batched(list(dict.fromkeys(keys)), 500):
            placeholders = ",".join("?" * len(batch))
            cur = self._db.execute(
                f"SELECT key, row FROM vectors WHERE key IN ({placeholders})", batch
            )
            rows.update({k: r for k, r in cur.fetchall()})

        if not rows:
            return {}

        data = np.memmap(self.vectors_path, dtype=np.float32, mode="r").reshape(-1, self.dim)
        # np.array(...) copies out of the memmap so the file handle can close.
        return {key: np.array(data[row], dtype=np.float32) for key, row in rows.items()}

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        """Append vectors and index them. Existing keys are left untouched."""
        if not items:
            return
        existing = set(self.get_existing_keys(list(items)))
        new = {k: v for k, v in items.items() if k not in existing}
        if not new:
            return

        row = self._next_row()
        block = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in new.values()])
        if block.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-dim vectors, got {block.shape[1]}")

        # Append to the blob first. If the process dies here, the blob has
        # orphan bytes but the index is consistent; the reverse would hand out
        # rows that do not exist.
        with open(self.vectors_path, "ab") as fh:
            fh.write(block.tobytes(order="C"))

        self._db.executemany(
            "INSERT OR IGNORE INTO vectors (key, row) VALUES (?, ?)",
            [(key, row + i) for i, key in enumerate(new)],
        )
        self._db.commit()

    def get_existing_keys(self, keys: Sequence[str]) -> list[str]:
        out: list[str] = []
        for batch in _batched(list(dict.fromkeys(keys)), 500):
            placeholders = ",".join("?" * len(batch))
            cur = self._db.execute(f"SELECT key FROM vectors WHERE key IN ({placeholders})", batch)
            out.extend(k for (k,) in cur.fetchall())
        return out

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class CachedEncoder:
    """An :class:`~reval.embedding.encoder.Encoder` fronted by the cache.

    Encodes only the texts it has never seen, in one batch, then reassembles the
    full matrix in the caller's order.
    """

    def __init__(self, encoder, cache_root: Path | None = None, enabled: bool = True):
        self.encoder = encoder
        self.enabled = enabled
        self._cache_root = cache_root
        self._cache: EmbeddingCache | None = None
        self.last_stats = CacheStats(0, 0, 0)

    # A CachedEncoder must be substitutable for an Encoder anywhere one is
    # accepted — the semantic chunker and the dense retriever both take
    # "an encoder" and should not care whether it is cached.
    @property
    def identity(self) -> str:
        return self.encoder.identity

    @property
    def namespace(self) -> str:
        return self.encoder.namespace

    @property
    def dim(self) -> int:
        return self.encoder.dim

    @property
    def config(self):
        return self.encoder.config

    @property
    def resolved_revision(self) -> str:
        return self.encoder.resolved_revision

    def prefix_for(self, kind: str) -> str:
        return self.encoder.prefix_for(kind)

    @property
    def cache(self) -> EmbeddingCache:
        if self._cache is None:
            self._cache = EmbeddingCache(
                namespace=self.encoder.namespace,
                dim=self.encoder.dim,
                root=self._cache_root,
            )
        return self._cache

    def encode(
        self,
        texts: list[str],
        kind: str = "passage",
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.encoder.dim), dtype=np.float32)
        if not self.enabled:
            self.last_stats = CacheStats(len(texts), 0, len(texts))
            return self.encoder.encode(texts, kind=kind, show_progress=show_progress)

        # The prefix is part of what gets embedded, so it must be part of the
        # key. Queries and passages of identical text therefore never collide
        # for an asymmetric model, and correctly *do* share an entry for a
        # symmetric one — which is a real cache win on near-duplicate corpora.
        identity = f"{self.encoder.identity}|kind={self.encoder.prefix_for(kind)!r}"
        keys = [EmbeddingCache.make_key(identity, t) for t in texts]

        cached = self.cache.get_many(keys)
        missing_idx = [i for i, k in enumerate(keys) if k not in cached]
        self.last_stats = CacheStats(
            requested=len(texts), hits=len(texts) - len(missing_idx), misses=len(missing_idx)
        )

        if missing_idx:
            fresh = self.encoder.encode(
                [texts[i] for i in missing_idx], kind=kind, show_progress=show_progress
            )
            new_items = {keys[i]: fresh[j] for j, i in enumerate(missing_idx)}
            self.cache.put_many(new_items)
            cached.update(new_items)

        return np.stack([cached[k] for k in keys]).astype(np.float32)

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None
