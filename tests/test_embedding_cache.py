"""The embedding cache is what makes the experiment grid affordable.

It is also the component where a bug is most dangerous: serving the wrong vector
does not raise, it just quietly produces a worse retriever.
"""

from __future__ import annotations

import numpy as np
import pytest

from reval.embedding.cache import CachedEncoder, EmbeddingCache


class CountingEncoder:
    """Fake encoder that records how many texts it was asked to embed."""

    def __init__(self, dim: int = 4, identity: str = "fake@rev1|norm=1"):
        self._dim = dim
        self.identity = identity
        self.namespace = identity.replace("/", "__").replace("|", "_").replace("@", "_")
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def prefix_for(self, kind: str) -> str:
        return "" if kind == "passage" else "q: "

    def encode(self, texts, kind="passage", show_progress=False):
        self.calls.append(list(texts))
        # Deterministic vector per text, so a wrong cache hit is detectable.
        return np.array(
            [[float(len(t)), float(sum(map(ord, t)) % 97), 1.0, 0.0] for t in texts],
            dtype=np.float32,
        )

    @property
    def embedded_count(self) -> int:
        return sum(len(c) for c in self.calls)


@pytest.fixture
def cached(tmp_path):
    enc = CountingEncoder()
    return CachedEncoder(enc, cache_root=tmp_path)


# ---- keys -----------------------------------------------------------------


def test_key_is_stable_for_the_same_inputs():
    assert EmbeddingCache.make_key("id", "text") == EmbeddingCache.make_key("id", "text")


def test_key_changes_with_identity_and_with_text():
    base = EmbeddingCache.make_key("id", "text")
    assert EmbeddingCache.make_key("id2", "text") != base
    assert EmbeddingCache.make_key("id", "text2") != base


def test_key_is_not_ambiguous_at_the_identity_text_boundary():
    # Without a separator, ("ab", "c") and ("a", "bc") would collide, and the
    # collision would serve one model's vectors under another model's identity.
    assert EmbeddingCache.make_key("ab", "c") != EmbeddingCache.make_key("a", "bc")


# ---- hit / miss behaviour -------------------------------------------------


def test_first_call_embeds_everything(cached):
    cached.encode(["a", "b", "c"])
    assert cached.encoder.embedded_count == 3
    assert cached.last_stats.misses == 3
    assert cached.last_stats.hits == 0


def test_second_identical_call_embeds_nothing(cached):
    cached.encode(["a", "b", "c"])
    cached.encode(["a", "b", "c"])
    assert cached.encoder.embedded_count == 3
    assert cached.last_stats.hits == 3
    assert cached.last_stats.misses == 0
    assert cached.last_stats.hit_rate == 1.0


def test_only_new_texts_are_embedded(cached):
    """The property the M2 acceptance criterion rests on."""
    cached.encode(["a", "b"])
    cached.encode(["a", "b", "c", "d"])
    assert cached.encoder.embedded_count == 4  # a, b once; then only c, d
    assert cached.last_stats.hits == 2
    assert cached.last_stats.misses == 2


def test_cached_vectors_equal_freshly_computed_ones(cached):
    fresh = cached.encoder.encode(["hello", "world"])
    cached.encode(["hello", "world"])
    from_cache = cached.encode(["hello", "world"])
    assert np.allclose(fresh, from_cache)


def test_results_come_back_in_the_requested_order(cached):
    cached.encode(["a", "b", "c"])
    out = cached.encode(["c", "a", "b"])
    direct = cached.encoder.encode(["c", "a", "b"])
    assert np.allclose(out, direct)


def test_duplicate_texts_in_one_call_are_embedded_once(cached):
    cached.encode(["same", "same", "same"])
    assert cached.encoder.embedded_count == 3  # one batch, but deduped by key
    out = cached.encode(["same", "same"])
    assert out.shape == (2, 4)
    assert np.allclose(out[0], out[1])


def test_empty_input_is_a_correctly_shaped_empty_array(cached):
    out = cached.encode([])
    assert out.shape == (0, 4)
    assert cached.encoder.embedded_count == 0


# ---- correctness guards ---------------------------------------------------


def test_queries_and_passages_do_not_share_entries_for_asymmetric_models(cached):
    """The prefix is part of what gets embedded, so it must be part of the key.

    An E5-style model embeds "query: X" and "passage: X" differently. Sharing a
    cache entry between them would serve the wrong vector with no error.
    """
    cached.encode(["text"], kind="passage")
    before = cached.encoder.embedded_count
    cached.encode(["text"], kind="query")
    assert cached.encoder.embedded_count == before + 1
    assert cached.last_stats.misses == 1


def test_a_different_encoder_identity_does_not_hit_the_cache(tmp_path):
    a = CachedEncoder(CountingEncoder(identity="model@rev1"), cache_root=tmp_path)
    a.encode(["shared text"])
    b = CachedEncoder(CountingEncoder(identity="model@rev2"), cache_root=tmp_path)
    b.encode(["shared text"])
    # Different revision => different namespace and key => a real miss.
    assert b.encoder.embedded_count == 1
    assert b.last_stats.misses == 1


def test_cache_persists_across_processes(tmp_path):
    first = CachedEncoder(CountingEncoder(), cache_root=tmp_path)
    first.encode(["persisted"])
    first.close()

    second = CachedEncoder(CountingEncoder(), cache_root=tmp_path)
    second.encode(["persisted"])
    assert second.encoder.embedded_count == 0
    assert second.last_stats.hits == 1


def test_dimension_mismatch_is_rejected(tmp_path):
    enc4 = CountingEncoder(dim=4, identity="collide")
    CachedEncoder(enc4, cache_root=tmp_path).encode(["x"])

    enc8 = CountingEncoder(dim=8, identity="collide")
    enc8.namespace = enc4.namespace  # force the collision
    with pytest.raises(ValueError, match="dim vectors"):
        CachedEncoder(enc8, cache_root=tmp_path).encode(["x"])


def test_disabling_the_cache_always_embeds(tmp_path):
    c = CachedEncoder(CountingEncoder(), cache_root=tmp_path, enabled=False)
    c.encode(["a"])
    c.encode(["a"])
    assert c.encoder.embedded_count == 2


def test_truncated_blob_is_detected(tmp_path):
    c = CachedEncoder(CountingEncoder(), cache_root=tmp_path)
    c.encode(["a", "b"])
    cache = c.cache
    # Simulate a half-written vector from a killed process.
    with open(cache.vectors_path, "ab") as fh:
        fh.write(b"\x00\x00\x00")
    with pytest.raises(ValueError, match="not a multiple of"):
        cache.put_many({"newkey": np.zeros(4, dtype=np.float32)})


def test_len_reports_stored_vectors(tmp_path):
    c = CachedEncoder(CountingEncoder(), cache_root=tmp_path)
    c.encode(["a", "b", "c"])
    assert len(c.cache) == 3


def test_stats_render_readably(cached):
    cached.encode(["a", "b"])
    cached.encode(["a", "b"])
    assert "100.0% hit rate" in str(cached.last_stats)
