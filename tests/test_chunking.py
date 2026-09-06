"""Chunkers must be deterministic and fingerprintable.

Both properties feed the embedding cache and the run manifest. A chunker that
moved a boundary between runs would invalidate the cache and, worse, silently
change what the retriever was scored on.
"""

from __future__ import annotations

import numpy as np
import pytest

from reval.chunking import (
    FixedTokenChunker,
    ParentDocumentChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    chunk_to_doc_map,
)
from reval.chunking.tokenize import WordTokenizer, get_tokenizer, sentence_spans
from reval.metrics.aggregation import aggregate_to_documents
from reval.types import Document

PROSE = (
    "Retrieval evaluation is hard. Most benchmarks are generated from the corpus "
    "they evaluate.\n\n"
    "That is circular. The number goes up and measures overlap.\n\n"
    "A harness should catch this. It should also report a confidence interval."
)


def doc(text: str = PROSE, title: str = "Title", doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, title=title, text=text)


# ---- tokenizers -----------------------------------------------------------


def test_word_tokenizer_offsets_are_exact():
    tok = WordTokenizer()
    text = "hello, world!"
    assert [text[s:e] for s, e in tok.spans(text)] == ["hello", ",", "world", "!"]


def test_get_tokenizer_rejects_unknown_specs():
    with pytest.raises(ValueError, match="unknown tokenizer"):
        get_tokenizer("bpe")


def test_sentence_spans_cover_the_whole_string():
    text = "One. Two! Three?"
    spans = sentence_spans(text)
    assert len(spans) == 3
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)


def test_sentence_spans_of_blank_text_is_empty():
    assert sentence_spans("   ") == []


# ---- shared properties ----------------------------------------------------


def all_chunkers():
    return [
        FixedTokenChunker(chunk_tokens=20, overlap_tokens=5),
        FixedTokenChunker(chunk_tokens=20, overlap_tokens=0),
        RecursiveCharacterChunker(chunk_chars=80, overlap_chars=10),
        RecursiveCharacterChunker(chunk_chars=200, overlap_chars=0),
        ParentDocumentChunker(FixedTokenChunker(chunk_tokens=20, overlap_tokens=5), 200),
    ]


@pytest.mark.parametrize("chunker", all_chunkers(), ids=lambda c: c.describe()[:40])
def test_chunking_is_deterministic(chunker):
    d = doc()
    first = chunker.chunk(d)
    for _ in range(3):
        assert chunker.chunk(d) == first


@pytest.mark.parametrize("chunker", all_chunkers(), ids=lambda c: c.describe()[:40])
def test_chunk_ids_are_unique_and_map_to_their_document(chunker):
    chunks = chunker.chunk(doc())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.chunk_id.startswith("d1#") for c in chunks)


@pytest.mark.parametrize("chunker", all_chunkers(), ids=lambda c: c.describe()[:40])
def test_char_offsets_recover_the_chunk_text(chunker):
    d = doc()
    full = d.full_text
    for c in chunks_of(chunker, d):
        assert full[c.char_start : c.char_end] == c.text


def chunks_of(chunker, d):
    return chunker.chunk(d)


@pytest.mark.parametrize("chunker", all_chunkers(), ids=lambda c: c.describe()[:40])
def test_empty_document_still_yields_one_chunk(chunker):
    # Otherwise the corpus the retriever sees differs from the corpus the qrels
    # judge, and recall is capped for an invisible reason.
    chunks = chunker.chunk(Document("empty", "", ""))
    assert len(chunks) == 1
    assert chunks[0].doc_id == "empty"


@pytest.mark.parametrize("chunker", all_chunkers(), ids=lambda c: c.describe()[:40])
def test_chunks_cover_the_document(chunker):
    d = doc()
    covered = set()
    for c in chunker.chunk(d):
        covered.update(range(c.char_start, c.char_end))
    # Allow whitespace at boundaries to be uncovered; content must not be.
    full = d.full_text
    missed = [i for i, ch in enumerate(full) if i not in covered and not ch.isspace()]
    assert not missed, f"{len(missed)} non-whitespace characters fell outside every chunk"


def test_fingerprints_differ_when_settings_differ():
    a = FixedTokenChunker(chunk_tokens=256, overlap_tokens=32)
    b = FixedTokenChunker(chunk_tokens=256, overlap_tokens=64)
    c = FixedTokenChunker(chunk_tokens=256, overlap_tokens=32)
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == c.fingerprint()


def test_fingerprint_is_stable_across_instances():
    assert (
        FixedTokenChunker(chunk_tokens=128).fingerprint()
        == FixedTokenChunker(chunk_tokens=128).fingerprint()
    )


def test_chunk_to_doc_map_covers_every_chunk():
    chunks = FixedTokenChunker(chunk_tokens=20, overlap_tokens=5).chunk_corpus(
        {"a": Document("a", "", PROSE), "b": Document("b", "", PROSE)}
    )
    mapping = chunk_to_doc_map(chunks)
    assert len(mapping) == len(chunks)
    assert set(mapping.values()) == {"a", "b"}


def test_corpus_chunking_is_in_sorted_document_order():
    corpus = {"z": Document("z", "", "x"), "a": Document("a", "", "y")}
    chunks = FixedTokenChunker().chunk_corpus(corpus)
    assert [c.doc_id for c in chunks] == ["a", "z"]


# ---- fixed-token ----------------------------------------------------------


def test_fixed_respects_the_token_budget():
    chunker = FixedTokenChunker(chunk_tokens=10, overlap_tokens=0)
    tok = WordTokenizer()
    for c in chunker.chunk(doc()):
        assert len(tok.spans(c.text)) <= 10


def test_fixed_overlap_actually_overlaps():
    with_ov = FixedTokenChunker(chunk_tokens=10, overlap_tokens=5).chunk(doc())
    without = FixedTokenChunker(chunk_tokens=10, overlap_tokens=0).chunk(doc())
    # Overlap means a shorter stride, hence strictly more chunks.
    assert len(with_ov) > len(without)
    # And consecutive chunks share characters.
    assert with_ov[1].char_start < with_ov[0].char_end


def test_fixed_rejects_overlap_that_would_stall_the_window():
    # stride = chunk - overlap <= 0 would loop forever or emit duplicates.
    with pytest.raises(ValueError, match="must be <"):
        FixedTokenChunker(chunk_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError, match="must be <"):
        FixedTokenChunker(chunk_tokens=10, overlap_tokens=20)


def test_fixed_does_not_emit_trailing_suffix_duplicates():
    # A naive window loop keeps emitting ever-shorter tails that are all
    # suffixes of the previous chunk, inflating the index for no information.
    chunks = FixedTokenChunker(chunk_tokens=10, overlap_tokens=8).chunk(doc())
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))


# ---- recursive ------------------------------------------------------------


def test_recursive_packs_small_pieces_up_to_the_budget():
    # Splitting on "\n\n" alone would emit one chunk per paragraph, far below
    # the target, making the size parameter meaningless.
    text = "\n\n".join(["short para"] * 12)
    chunks = RecursiveCharacterChunker(chunk_chars=200, overlap_chars=0).chunk(
        Document("d", "", text)
    )
    assert len(chunks) < 12
    assert max(len(c.text) for c in chunks) <= 200


def test_recursive_hard_splits_text_with_no_separators():
    text = "x" * 500
    chunks = RecursiveCharacterChunker(chunk_chars=100, overlap_chars=0).chunk(
        Document("d", "", text)
    )
    assert len(chunks) >= 5
    assert all(len(c.text) <= 100 for c in chunks)


def test_recursive_rejects_overlap_larger_than_chunk():
    with pytest.raises(ValueError, match="must be <"):
        RecursiveCharacterChunker(chunk_chars=100, overlap_chars=100)


def test_recursive_rejects_empty_separators():
    with pytest.raises(ValueError, match="must not be empty"):
        RecursiveCharacterChunker(separators=[])


# ---- parent-document ------------------------------------------------------


def test_parent_retrieval_text_is_the_child_span():
    child = FixedTokenChunker(chunk_tokens=10, overlap_tokens=0)
    parent = ParentDocumentChunker(child, parent_chars=200)
    d = doc()
    assert [c.text for c in parent.chunk(d)] == [c.text for c in child.chunk(d)]


def test_parent_text_contains_the_child_and_is_larger():
    parent = ParentDocumentChunker(FixedTokenChunker(chunk_tokens=8, overlap_tokens=0), 200)
    for c in parent.chunk(doc()):
        assert c.parent_text is not None
        assert c.text in c.parent_text
        assert len(c.parent_text) >= len(c.text)


def test_parent_matches_child_at_document_level():
    """Parent-document must not change document-level scores.

    Our metrics run after chunk->document aggregation, and a parent is a window
    of the same document, so parent-document retrieval is identical to its child
    chunker once aggregated. If this ever fails, the harness has a bug — it is
    not a finding about chunking.
    """
    child = FixedTokenChunker(chunk_tokens=10, overlap_tokens=2)
    parent = ParentDocumentChunker(child, parent_chars=300)
    corpus = {"a": Document("a", "", PROSE), "b": Document("b", "", PROSE[:80])}

    child_chunks = child.chunk_corpus(corpus)
    parent_chunks = parent.chunk_corpus(corpus)
    assert [c.chunk_id for c in child_chunks] == [c.chunk_id for c in parent_chunks]

    run = {"q1": {c.chunk_id: 1.0 / (i + 1) for i, c in enumerate(child_chunks)}}
    assert aggregate_to_documents(run, chunk_to_doc_map(child_chunks)) == aggregate_to_documents(
        run, chunk_to_doc_map(parent_chunks)
    )


def test_parent_config_records_the_child():
    cfg = ParentDocumentChunker(FixedTokenChunker(chunk_tokens=64), 500).config()
    assert cfg["child"]["name"] == "fixed"
    assert cfg["child"]["chunk_tokens"] == 64


# ---- semantic -------------------------------------------------------------


class StubEncoder:
    """Deterministic fake encoder: two clusters, so a boundary is predictable."""

    identity = "stub-encoder"

    def encode(self, texts, kind="passage", show_progress=False):
        # Sentences containing "cat" point one way, everything else another.
        return np.array(
            [[1.0, 0.0] if "cat" in t.lower() else [0.0, 1.0] for t in texts], dtype=np.float32
        )


def test_semantic_cuts_where_similarity_drops():
    text = (
        "The cat sat on the mat. The cat was orange. The cat purred loudly. "
        "Interest rates rose sharply. Bond yields followed suit. Markets closed lower."
    )
    chunker = SemanticChunker(StubEncoder(), breakpoint_percentile=50.0, min_chunk_chars=10)
    chunks = chunker.chunk(Document("d", "", text))
    assert len(chunks) >= 2
    # The topic switch should not be buried mid-chunk.
    assert any("cat" in c.text and "Interest" not in c.text for c in chunks)


def test_semantic_is_deterministic():
    chunker = SemanticChunker(StubEncoder(), breakpoint_percentile=50.0, min_chunk_chars=10)
    d = doc()
    assert chunker.chunk(d) == chunker.chunk(d)


def test_semantic_records_its_encoder_in_the_config():
    # Unlike the other chunkers, this one's output depends on a model, so the
    # model has to be part of its identity.
    assert SemanticChunker(StubEncoder()).config()["encoder"] == "stub-encoder"


def test_semantic_rejects_a_nonsense_percentile():
    with pytest.raises(ValueError, match="breakpoint_percentile"):
        SemanticChunker(StubEncoder(), breakpoint_percentile=0)


def test_semantic_handles_single_sentence_documents():
    chunker = SemanticChunker(StubEncoder(), min_chunk_chars=1)
    chunks = chunker.chunk(Document("d", "", "Only one sentence here"))
    assert len(chunks) == 1
