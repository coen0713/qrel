"""Near-duplicate corpus documents. PLAN.md M3(b).

Duplicates inflate recall by giving the retriever several valid targets for one
information need: two near-identical documents mean two chances to land a hit in
the top k, and only one of them is usually in the qrels — so the *other* one
counts as a miss even though it is equally correct. Both directions of that are
measurement error.

MinHash gives an unbiased estimate of Jaccard similarity from a fixed-length
signature; LSH banding restricts the comparisons to plausible candidates. The
alternative is 57,638² / 2 pairwise comparisons on FiQA, which is 1.7 billion.

Implemented here rather than pulled from ``datasketch`` because it is ~80 lines,
it removes a dependency from a module whose correctness the project's central
claim rests on, and a MinHash whose permutation seeding we do not control is a
reproducibility hole.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from reval.types import Corpus, DocId

#: Word-level shingle size. 5 words is long enough that generic phrasing does
#: not dominate, short enough to survive an edit inside a sentence.
DEFAULT_SHINGLE = 5

#: 64-bit Mersenne prime; the modulus for the permutation family.
_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def shingles(text: str, k: int = DEFAULT_SHINGLE) -> set[str]:
    """Contiguous k-word windows, case-folded."""
    words = text.lower().split()
    if not words:
        return set()
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def exact_jaccard(a: set[str], b: set[str]) -> float:
    """True Jaccard of two shingle sets. The oracle MinHash approximates."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """Two corpus documents estimated to be near-copies."""

    doc_a: DocId
    doc_b: DocId
    similarity: float


class MinHasher:
    """Fixed-length signatures whose collision rate estimates Jaccard.

    Uses the standard universal family ``h_i(x) = (a_i * x + b_i) mod p``, with
    ``(a_i, b_i)`` drawn from a seeded RNG so signatures are reproducible across
    processes and machines. An unseeded permutation family would make the
    duplicate report differ between runs, which is the kind of instability that
    destroys trust in a number nobody can re-derive.
    """

    def __init__(self, num_perm: int = 128, shingle_size: int = DEFAULT_SHINGLE, seed: int = 0):
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.seed = seed
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, _MERSENNE, size=num_perm, dtype=np.uint64)
        self._b = rng.integers(0, _MERSENNE, size=num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        """MinHash signature of ``text`` as a ``(num_perm,)`` uint64 array."""
        grams = shingles(text, self.shingle_size)
        if not grams:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)

        # Hash each shingle once to a 32-bit value, then apply every permutation
        # to the whole vector at once — the loop is over shingles, not over
        # (shingle x permutation), which matters at corpus scale.
        base = np.array(
            [hash_shingle(g) for g in sorted(grams)],
            dtype=np.uint64,
        )
        # (num_shingles, num_perm) then min over shingles.
        permuted = (np.outer(base, self._a) + self._b) % np.uint64(_MERSENNE)
        return permuted.min(axis=0).astype(np.uint64)

    @staticmethod
    def estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimated Jaccard: the fraction of signature positions that agree."""
        if len(sig_a) != len(sig_b):
            raise ValueError("signatures must have the same length")
        return float(np.count_nonzero(sig_a == sig_b) / len(sig_a))


def hash_shingle(gram: str) -> int:
    """Stable 32-bit hash of a shingle.

    Explicitly not Python's ``hash()``: that is randomised per process by
    ``PYTHONHASHSEED``, so signatures would differ between runs and the
    duplicate report would not reproduce.
    """
    import hashlib

    return int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest(), "big")


def _choose_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) whose LSH S-curve inflects near ``threshold``.

    The probability two documents become candidates is ``1 - (1 - s^r)^b``,
    which has its steep region near ``(1/b)^(1/r)``. We search the factorisations
    of ``num_perm`` for the one whose inflection sits closest to the requested
    threshold, so recall and precision of the candidate stage both track what the
    caller asked for instead of a hard-coded band count.
    """
    best: tuple[float, int, int] | None = None
    for bands in range(1, num_perm + 1):
        if num_perm % bands:
            continue
        rows = num_perm // bands
        inflection = (1.0 / bands) ** (1.0 / rows)
        error = abs(inflection - threshold)
        if best is None or error < best[0]:
            best = (error, bands, rows)
    assert best is not None
    return best[1], best[2]


#: Documents shorter than this are excluded from duplicate detection. "Near
#: duplicate" is not a meaningful claim about two empty strings, and they
#: collide trivially — FiQA contains 38 empty documents, which produced a single
#: 38-member "cluster" accounting for 703 of 890 reported pairs and drowning out
#: every real duplicate. Report them with `degenerate_documents` instead.
MIN_WORDS = 5


def degenerate_documents(corpus: Corpus, min_words: int = MIN_WORDS) -> list[DocId]:
    """Corpus documents too short to carry information.

    A separate corpus-quality finding, not a duplicate-detection result. A
    document with no text that is also a gold answer makes its query
    unanswerable, which caps recall for a reason no retriever can fix.
    """
    return sorted(d for d, doc in corpus.items() if len(doc.full_text.split()) < min_words)


def find_near_duplicates(
    corpus: Corpus,
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle_size: int = DEFAULT_SHINGLE,
    seed: int = 0,
    min_words: int = MIN_WORDS,
) -> list[DuplicatePair]:
    """Find document pairs whose estimated Jaccard is at least ``threshold``.

    Two stages: LSH banding proposes candidates, then every candidate pair is
    re-checked against ``threshold`` using the full signature. The banding stage
    is allowed to over-propose — it is a filter, not the answer — so a
    false-positive band collision costs one signature comparison rather than a
    wrong result.

    Documents shorter than ``min_words`` are excluded; see :data:`MIN_WORDS` for
    why that is a correctness fix and not a convenience.

    Returns pairs sorted by descending similarity, each pair reported once.
    """
    if len(corpus) < 2:
        return []

    hasher = MinHasher(num_perm=num_perm, shingle_size=shingle_size, seed=seed)
    doc_ids = sorted(d for d in corpus if len(corpus[d].full_text.split()) >= min_words)
    if len(doc_ids) < 2:
        return []
    signatures = {doc_id: hasher.signature(corpus[doc_id].full_text) for doc_id in doc_ids}

    bands, rows = _choose_bands(num_perm, threshold)
    buckets: dict[tuple[int, bytes], list[DocId]] = defaultdict(list)
    for doc_id in doc_ids:
        sig = signatures[doc_id]
        for band in range(bands):
            chunk = sig[band * rows : (band + 1) * rows].tobytes()
            buckets[(band, chunk)].append(doc_id)

    candidates: set[tuple[DocId, DocId]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                candidates.add((a, b) if a < b else (b, a))

    pairs = [
        DuplicatePair(a, b, sim)
        for a, b in sorted(candidates)
        if (sim := MinHasher.estimate(signatures[a], signatures[b])) >= threshold
    ]
    # Sort by similarity desc, then ids, so the report is deterministic.
    pairs.sort(key=lambda p: (-p.similarity, p.doc_a, p.doc_b))
    return pairs


def duplicate_clusters(pairs: list[DuplicatePair]) -> list[set[DocId]]:
    """Group duplicate pairs into connected components.

    A corpus with three copies of one document yields three pairs but one
    information need; counting pairs would overstate the problem.
    """
    parent: dict[DocId, DocId] = {}

    def find(x: DocId) -> DocId:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        ra, rb = find(pair.doc_a), find(pair.doc_b)
        if ra != rb:
            parent[rb] = ra

    clusters: dict[DocId, set[DocId]] = defaultdict(set)
    for node in list(parent):
        clusters[find(node)].add(node)
    return sorted(clusters.values(), key=lambda c: (-len(c), sorted(c)[0]))
