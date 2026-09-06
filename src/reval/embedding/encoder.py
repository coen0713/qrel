"""Sentence encoder wrapper with a pinned revision.

The revision pin is not bureaucracy. Hugging Face model repositories are
mutable: ``sentence-transformers/all-MiniLM-L6-v2`` today and the same name in
six months can be different weights. A benchmark that records only the model
*name* is not reproducible, and — worse for this project specifically — an
embedding cache keyed on the name alone would happily serve vectors from the old
weights after the model silently changed underneath it.

So: every encoder resolves and records the commit SHA it actually loaded, that
SHA goes into the cache key and the run manifest, and a run whose model moved
re-embeds instead of quietly mixing two models' vectors in one index.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from reval.paths import cache_dir

TextKind = Literal["query", "passage"]


def _revision_cache_path():
    return cache_dir() / "resolved_revisions.json"


def _revision_cache_get(key: str) -> str | None:
    path = _revision_cache_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key)
    except (json.JSONDecodeError, OSError):
        return None


def _revision_cache_put(key: str, sha: str) -> None:
    path = _revision_cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        data = {}
    data[key] = sha
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(slots=True)
class EncoderConfig:
    """Everything about an encoder that changes its output vectors."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Git revision (branch, tag or SHA). ``None`` resolves whatever main is now
    #: and records the SHA, so a run is reproducible after the fact even if it
    #: was not pinned in advance.
    revision: str | None = None
    normalize: bool = True
    batch_size: int = 64
    device: str | None = None
    max_seq_length: int | None = None
    #: Asymmetric models (E5, BGE, GTE) need different prefixes for queries and
    #: passages. Getting these wrong is a quiet 5-10 point recall loss that
    #: looks like "this model is bad" rather than "we used it wrong".
    query_prefix: str = ""
    passage_prefix: str = ""


#: Encoders used in the experiment grid, with their required prefixes.
KNOWN_ENCODERS: dict[str, EncoderConfig] = {
    "minilm": EncoderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    ),
    "bge-small": EncoderConfig(
        model_name="BAAI/bge-small-en-v1.5",
        revision="main",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "e5-small": EncoderConfig(
        model_name="intfloat/e5-small-v2",
        revision="main",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
}


@dataclass(slots=True)
class Encoder:
    """A ``sentence-transformers`` model, loaded lazily and revision-pinned."""

    config: EncoderConfig = field(default_factory=EncoderConfig)
    _model: object | None = field(default=None, init=False, repr=False)
    _resolved_revision: str | None = field(default=None, init=False, repr=False)
    _dim: int | None = field(default=None, init=False, repr=False)

    @classmethod
    def named(cls, name: str) -> Encoder:
        """Build one of :data:`KNOWN_ENCODERS`, or treat ``name`` as a HF id."""
        if name in KNOWN_ENCODERS:
            return cls(config=KNOWN_ENCODERS[name])
        return cls(config=EncoderConfig(model_name=name))

    def load(self):
        """Load the model. Idempotent."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "dense retrieval needs the [dense] extra: uv pip install -e '.[dense]'"
            ) from exc

        cfg = self.config
        model = SentenceTransformer(
            cfg.model_name,
            revision=cfg.revision,
            device=cfg.device,
        )
        if cfg.max_seq_length is not None:
            model.max_seq_length = cfg.max_seq_length

        self._model = model
        # sentence-transformers 6 renamed this; support both so the pin in
        # pyproject can move without the encoder breaking.
        get_dim = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        self._dim = int(get_dim())
        self._resolved_revision = self._resolve_revision()
        return model

    def _resolve_revision(self) -> str:
        """Find the commit SHA for the requested revision.

        Cached on disk. Resolving costs a Hub round-trip, and the identity it
        feeds is needed on every warm run — including runs that reuse an index
        and never touch the model. Paying network latency to re-derive a
        constant would make a no-op re-run cost seconds for nothing.
        """
        cfg = self.config
        # Already a full commit SHA: nothing to resolve.
        if (
            cfg.revision
            and len(cfg.revision) == 40
            and all(c in "0123456789abcdef" for c in cfg.revision.lower())
        ):
            return cfg.revision.lower()

        key = f"{cfg.model_name}@{cfg.revision or 'main'}"
        cached = _revision_cache_get(key)
        if cached:
            return cached

        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(cfg.model_name, revision=cfg.revision)
            if info.sha:
                sha = str(info.sha)
                _revision_cache_put(key, sha)
                return sha
        except Exception:
            # Offline, rate-limited, or a local path. Fall back to whatever was
            # requested — less precise, but never a reason to fail a run. The
            # manifest records what we got, so the imprecision is visible.
            pass
        return cfg.revision or "unpinned"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        return int(self._dim)  # type: ignore[arg-type]

    @property
    def resolved_revision(self) -> str:
        """The commit SHA, resolved *without* loading model weights.

        Deliberately not routed through :meth:`load`. ``identity`` depends on
        this, index fingerprints depend on ``identity``, and a warm re-run that
        reuses its index must not pay 18 seconds of weight loading to compute a
        string it already knows.
        """
        if self._resolved_revision is None:
            self._resolved_revision = self._resolve_revision()
        return str(self._resolved_revision)

    @property
    def identity(self) -> str:
        """Everything that determines a vector, as one string.

        This is the cache namespace and the manifest's model field. Note it
        includes the prefixes: the same model with and without an E5 prefix
        produces different vectors and must not share a cache.
        """
        cfg = self.config
        return (
            f"{cfg.model_name}@{self.resolved_revision}"
            f"|norm={int(cfg.normalize)}"
            f"|qp={cfg.query_prefix!r}"
            f"|pp={cfg.passage_prefix!r}"
            f"|msl={cfg.max_seq_length}"
        )

    @property
    def namespace(self) -> str:
        """Short filesystem-safe digest of :attr:`identity`."""
        digest = hashlib.sha256(self.identity.encode("utf-8")).hexdigest()[:16]
        slug = self.config.model_name.replace("/", "__")
        return f"{slug}.{digest}"

    def prefix_for(self, kind: TextKind) -> str:
        return self.config.query_prefix if kind == "query" else self.config.passage_prefix

    def encode(
        self,
        texts: list[str],
        kind: TextKind = "passage",
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into a ``(n, dim)`` float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        model = self.load()
        prefix = self.prefix_for(kind)
        payload = [prefix + t for t in texts] if prefix else texts

        vectors = model.encode(
            payload,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)
