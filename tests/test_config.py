"""Config validation and hashing.

The config hash identifies an experiment. If two different experiments can share
a hash, or one experiment can produce two hashes, the manifest stops meaning
anything.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reval.experiments.config import ContaminationConfig, ExperimentConfig
from reval.experiments.manifest import Manifest, git_state, package_versions
from reval.paths import configs_dir


def test_defaults_are_valid():
    cfg = ExperimentConfig()
    assert cfg.dataset == "scifact"
    assert cfg.eval.top_k >= max(cfg.eval.ks)


def test_unknown_keys_are_rejected():
    """A typo'd key that silently does nothing is the worst config bug.

    The run succeeds, the setting you meant to change did not change, and the
    number is for a different experiment than the one you think you ran.
    """
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentConfig.model_validate({"chunker": {"kind": "fixed", "chunk_size": 256}})


def test_unknown_chunker_kind_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"chunker": {"kind": "magic"}})


def test_top_k_shallower_than_the_largest_cutoff_is_rejected():
    # Otherwise recall@100 measures the retrieval depth, not the retriever.
    with pytest.raises(ValidationError, match="shallower"):
        ExperimentConfig.model_validate({"eval": {"ks": [1, 10, 100], "top_k": 20}})


def test_overlap_larger_than_chunk_is_rejected():
    with pytest.raises(ValidationError, match="must be <"):
        ExperimentConfig.model_validate({"chunker": {"chunk_tokens": 100, "overlap_tokens": 100}})


# ---- hashing --------------------------------------------------------------


def test_hash_is_stable_for_identical_configs():
    assert ExperimentConfig().config_hash() == ExperimentConfig().config_hash()


def test_hash_changes_when_any_setting_changes():
    base = ExperimentConfig()
    variants = [
        {"seed": 1},
        {"dataset": "fiqa"},
        {"chunker": base.chunker.model_copy(update={"chunk_tokens": 512})},
        {"eval": base.eval.model_copy(update={"aggregation": "sum"})},
        {"retriever": base.retriever.model_copy(update={"kind": "dense"})},
    ]
    hashes = {base.config_hash()} | {base.model_copy(update=v).config_hash() for v in variants}
    assert len(hashes) == len(variants) + 1, "two distinct configs collided on one hash"


def test_hash_is_insensitive_to_yaml_key_order(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("dataset: fiqa\nseed: 3\nname: x\n", encoding="utf-8")
    b.write_text("name: x\nseed: 3\ndataset: fiqa\n", encoding="utf-8")
    assert (
        ExperimentConfig.from_yaml(a).config_hash() == ExperimentConfig.from_yaml(b).config_hash()
    )


def test_yaml_roundtrip_preserves_the_hash(tmp_path):
    cfg = ExperimentConfig(dataset="nfcorpus", seed=7)
    path = cfg.to_yaml(tmp_path / "c.yaml")
    assert ExperimentConfig.from_yaml(path).config_hash() == cfg.config_hash()


def test_run_tag_is_filesystem_safe_and_identifies_the_experiment():
    tag = ExperimentConfig(dataset="fiqa").run_tag()
    assert tag.startswith("fiqa.fixed.bm25.")
    assert not set(tag) & set('<>:"/\\|?*')


def test_dense_run_tag_names_the_encoder():
    cfg = ExperimentConfig()
    cfg = cfg.model_copy(update={"retriever": cfg.retriever.model_copy(update={"kind": "dense"})})
    assert "dense-minilm" in cfg.run_tag()


# ---- shipped configs ------------------------------------------------------


def _model_for(path) -> type:
    """Pick the config model a YAML file is written against.

    The repo ships two kinds. ``retrievers`` (plural) is the discriminator: only
    the contamination experiment sweeps a list of them, because it scores one
    index against several eval conditions rather than running one retriever.
    """
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ContaminationConfig if "retrievers" in data else ExperimentConfig


def test_every_shipped_config_parses():
    paths = sorted(configs_dir().glob("*.yaml"))
    assert paths, "no configs found"
    for p in paths:
        _model_for(p).from_yaml(p)


def test_both_config_kinds_are_shipped():
    # Guards the discriminator above: if every config became one kind, the
    # dispatch would silently stop being exercised.
    kinds = {_model_for(p) for p in configs_dir().glob("*.yaml")}
    assert kinds == {ExperimentConfig, ContaminationConfig}


def test_shipped_configs_have_distinct_hashes():
    hashes = {
        p.name: _model_for(p).from_yaml(p).config_hash() for p in configs_dir().glob("*.yaml")
    }
    assert len(set(hashes.values())) == len(hashes), f"duplicate config hashes: {hashes}"


def test_the_contamination_config_points_at_real_prompt_variants():
    """The synthetic sets named in the config must be ones we can generate.

    A config referring to a variant with no prompt file would fail only at
    generation time, after the corpus load and the sampling.
    """
    from reval.contamination.generate import load_prompt

    cfg = ContaminationConfig.from_yaml(configs_dir() / "contamination.yaml")
    assert cfg.synthetic, "contamination config names no synthetic sets"
    for path in cfg.synthetic:
        variant = path.rsplit("-", 1)[-1].split(".")[0]
        load_prompt(variant)  # raises if the prompt file is missing


# ---- manifest -------------------------------------------------------------


def test_git_state_reports_a_sha_and_a_dirty_flag():
    state = git_state()
    assert set(state) == {"sha", "dirty"}
    assert isinstance(state["dirty"], bool)


def test_package_versions_include_our_own():
    assert "qrel" in package_versions()


def test_manifest_roundtrips(tmp_path):
    from reval.types import Dataset

    ds = Dataset(
        name="toy",
        corpus={},
        queries={"q1": "x"},
        qrels={"q1": {"d1": 1}},
        checksums={"corpus.jsonl": "a" * 64},
    )
    m = Manifest.build(ExperimentConfig(), ds)
    path = m.write(tmp_path / "m.json")
    back = Manifest.read(path)
    assert back["config_hash"] == ExperimentConfig().config_hash()
    assert back["corpus_checksums"]["corpus.jsonl"] == "a" * 64
    assert back["seed"] == 0


def test_manifest_warns_about_a_dirty_tree():
    """A git SHA alone is a lie when the tree had uncommitted edits.

    That is the normal state while developing, so the warning has to be loud or
    it will be believed.
    """
    m = Manifest(
        run_tag="t",
        config_hash="h",
        config={},
        dataset="toy",
        split="test",
        seed=0,
        git={"sha": "abc", "dirty": True},
        corpus_checksums={},
    )
    assert any("dirty" in w for w in m.warnings())

    clean = Manifest(
        run_tag="t",
        config_hash="h",
        config={},
        dataset="toy",
        split="test",
        seed=0,
        git={"sha": "abc", "dirty": False},
        corpus_checksums={},
    )
    assert clean.warnings() == []
