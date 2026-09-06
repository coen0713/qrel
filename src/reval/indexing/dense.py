"""Dense retrieval backed by Qdrant.

Two modes behind one code path:

- ``embedded`` (default) — ``QdrantClient(path=...)``, Qdrant's local mode. Same
  client library, same API calls, no server and no Docker. This is what keeps
  "git clone and reproduce the table" true for someone without Docker installed.
- ``server`` — ``QdrantClient(url=...)`` against ``docker-compose.yml``.

Nothing below branches on the mode except client construction, which is the
point: the retriever code is not a toy that happens to work locally.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from reval.indexing.base import IndexNotBuiltError
from reval.paths import index_dir
from reval.types import Chunk, Queries, Run


class DenseRetriever:
    """Vector retrieval over chunk embeddings."""

    def __init__(
        self,
        encoder,
        collection: str = "reval",
        mode: str = "embedded",
        url: str | None = None,
        storage_path: str | None = None,
        batch_size: int = 512,
        name: str | None = None,
    ):
        if mode not in {"embedded", "server"}:
            raise ValueError(f"mode must be 'embedded' or 'server', got {mode!r}")
        self.encoder = encoder
        self.collection = collection
        self.mode = mode
        self.url = url or "http://localhost:6333"
        self.storage_path = storage_path
        self.batch_size = batch_size
        self.name = (
            name or f"dense:{getattr(encoder, 'config', None) and encoder.config.model_name}"
        )

        self._client = None
        self._indexed = False
        self._n_points = 0
        #: True when index() reused an existing index instead of rebuilding.
        self.reused = False

    def config(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "collection": self.collection,
            "encoder": self.encoder.identity,
        }

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "dense retrieval needs the [dense] extra: uv pip install -e '.[dense]'"
            ) from exc

        if self.mode == "server":
            self._client = QdrantClient(url=self.url)
        else:
            path = self.storage_path or str(index_dir() / "qdrant" / self.collection)
            self._client = QdrantClient(path=path)
        return self._client

    def _fingerprint(self, chunks: Sequence[Chunk]) -> str:
        """Digest of exactly what an index built from ``chunks`` would contain.

        Covers the encoder identity (so a model or revision change invalidates)
        and every chunk id and body (so any chunker change invalidates). Chunk
        *count* alone would not: two chunkers can produce the same number of
        differently-cut chunks.
        """
        h = hashlib.sha256()
        h.update(self.encoder.identity.encode("utf-8"))
        h.update(b"\x00")
        for c in chunks:
            h.update(c.chunk_id.encode("utf-8"))
            h.update(b"\x1f")
            h.update(c.text.encode("utf-8"))
            h.update(b"\x1e")
        return h.hexdigest()

    def _fingerprint_path(self) -> Path:
        base = self.storage_path or str(index_dir() / "qdrant" / self.collection)
        return Path(base).parent / f"{self.collection}.fingerprint"

    def index(self, chunks: Sequence[Chunk], force: bool = False) -> None:
        """Build the index, reusing it when its contents have not changed.

        Reconciliation, not rebuild-every-time. An index whose fingerprint
        already matches is left alone, which is what makes a warm re-run cheap:
        the embedding cache removes the encoder cost, and this removes the
        upsert cost, which for 7,892 points in embedded Qdrant is the larger of
        the two on a warm run.
        """
        if not chunks:
            raise ValueError("cannot index an empty chunk list")

        from qdrant_client import models

        client = self._get_client()
        fingerprint = self._fingerprint(chunks)
        marker = self._fingerprint_path()

        fingerprint_matches = (
            not force
            and marker.exists()
            and marker.read_text(encoding="utf-8") == fingerprint
            and client.collection_exists(self.collection)
        )
        if fingerprint_matches and client.count(self.collection, exact=True).count == len(chunks):
            self._n_points = len(chunks)
            self._indexed = True
            self.reused = True
            return
        self.reused = False

        # Recreate rather than upsert-into-existing: a stale collection from a
        # different chunker would silently mix two chunkings in one index, and
        # the resulting recall would be meaningless in a way nothing detects.
        if client.collection_exists(self.collection):
            client.delete_collection(self.collection)
        client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.encoder.dim,
                # Cosine, and encoders are normalised, so scores are directly
                # comparable across models — which the grid needs.
                distance=models.Distance.COSINE,
            ),
        )

        vectors = np.asarray(
            self.encoder.encode([c.text for c in chunks], kind="passage", show_progress=True),
            dtype=np.float32,
        )
        if vectors.shape[0] != len(chunks):
            raise RuntimeError(
                f"encoder returned {vectors.shape[0]} vectors for {len(chunks)} chunks"
            )

        client.upload_collection(
            collection_name=self.collection,
            vectors=vectors,
            payload=[{"chunk_id": c.chunk_id, "doc_id": c.doc_id} for c in chunks],
            # Qdrant point ids must be uint or UUID, and our chunk ids are
            # strings like "doc123#4". Positional integer ids with the real id
            # in the payload keeps the mapping explicit rather than hashing into
            # a collision risk.
            ids=list(range(len(chunks))),
            batch_size=self.batch_size,
        )

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(fingerprint, encoding="utf-8")

        self._n_points = len(chunks)
        self._indexed = True

    def search(self, queries: Queries, top_k: int) -> Run:
        if not self._indexed:
            raise IndexNotBuiltError("call index() before search()")
        if not queries:
            return {}

        client = self._get_client()
        qids = sorted(queries)
        vectors = self.encoder.encode([queries[q] for q in qids], kind="query")

        run: Run = {}
        for i, qid in enumerate(qids):
            points = client.query_points(
                collection_name=self.collection,
                query=vectors[i].tolist(),
                limit=min(top_k, self._n_points),
                with_payload=True,
            ).points
            run[qid] = {p.payload["chunk_id"]: float(p.score) for p in points}
        return run

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def drop(self) -> None:
        """Delete the on-disk collection. Used by tests and ``--force``."""
        self.close()
        self._fingerprint_path().unlink(missing_ok=True)
        if self.mode == "embedded":
            path = self.storage_path or str(index_dir() / "qdrant" / self.collection)
            shutil.rmtree(path, ignore_errors=True)


def new_collection_name(prefix: str = "reval") -> str:
    """A unique collection name, for tests that must not share storage."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"
