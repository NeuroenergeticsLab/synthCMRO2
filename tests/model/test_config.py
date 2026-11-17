"""Configuration: naming compatible with the trained models, folds, and leakage checks."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import DATA, ROOT

from synthcmro2.model.config import ConfigError, load_config, parse_config
from synthcmro2.model.folds import cohort_folds, make_folds
from synthcmro2.model.naming import checkpoint_name, inputs_from_name


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config" / "model.yaml")


def test_model_names_match_the_trained_checkpoints(cfg):
    trained = set(pd.read_csv(DATA / "cv_selection_20261030.csv")["config"])
    assert {m.name for m in cfg.models.values()} == trained
    assert cfg.models["reference"].name == "_OmitT1"
    assert cfg.models["cbf_t2s"].name == "_OmitCBV_OmitT2_OmitT1"
    assert cfg.models["degree_centrality"].name == "_OmitCBV_OmitCBF_OmitT2_OmitT2S_OmitT1"


@pytest.mark.parametrize("inputs", [("CBV", "CBF", "T2", "T2S"), ("CBF",), ("T1",), ("fMRI",), ("T2", "T2S")])
def test_name_round_trip(inputs):
    assert inputs_from_name(checkpoint_name(inputs)) == inputs


@pytest.mark.parametrize("bad", [("CBF", "CBV"), ("CBF", "CBF"), ("CBF", "fMRI"), ("ASL",), ()])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        checkpoint_name(bad)


def test_main_folds_reproduce_training_log(cfg):
    folds = cohort_folds(cfg.cohorts["main"], 4, 1234)
    assert [len(v) for _, v in folds] == [14, 14, 14, 12]  # sub-stab035 (fold 4) excluded
    assert folds[0][1] == list(cfg.cohorts["main"].fold1_validation)
    val = [s for _, v in folds for s in v]
    assert sorted(val) == sorted(cfg.cohorts["main"].used_ids)  # every participant validated exactly once
    for tr, va in folds:
        assert not set(tr) & set(va)


def test_wrong_order_is_detected(cfg):
    from dataclasses import replace

    shuffled = replace(cfg.cohorts["main"], train_ids=tuple(sorted(cfg.cohorts["main"].train_ids)))
    with pytest.raises(RuntimeError, match="does not reproduce"):
        cohort_folds(shuffled, 4, 1234)


def test_fmri_folds(cfg):
    folds = make_folds(cfg.cohorts["fmri_subset"].train_ids, 4, 1234)
    assert [len(v) for _, v in folds] == [4, 4, 4, 4]


def test_cohorts_and_test_sets_are_consistent(cfg):
    main, fmri = set(cfg.cohorts["main"].train_ids), set(cfg.cohorts["fmri_subset"].train_ids)
    assert len(main) == 55 and len(fmri) == 16 and fmri <= main
    assert len(cfg.datasets["test_set"].ids) == 41
    assert set(cfg.datasets["calculation"].ids) <= set(cfg.datasets["test_set"].ids)
    assert set(cfg.datasets["fmri_subset"].ids) <= set(cfg.datasets["test_set"].ids)
    assert cfg.datasets["fmri_subset"].models == ("degree_centrality",)
    assert "degree_centrality" not in cfg.datasets["test_set"].models
    for ds in cfg.datasets.values():
        if ds.out_of_fold:
            assert set(ds.ids) == set(cfg.cohorts["main"].used_ids)  # each predicted by its validation fold
        else:
            assert not set(ds.ids) & main


def test_training_participant_in_test_set_is_refused(raw, tmp_path):
    raw["testing"]["datasets"]["test_set"]["ids"].append(raw["cohorts"]["main"]["train_ids"][0])
    with pytest.raises(ConfigError, match="training participants"):
        parse_config(raw, tmp_path)


def test_duplicates_and_unknown_entries_are_refused(raw, tmp_path):
    bad = [
        ("cohorts", lambda r: r["cohorts"]["main"]["train_ids"].append(r["cohorts"]["main"]["train_ids"][0])),
        ("unknown model", lambda r: r["testing"]["datasets"]["replication"].update(models=["nope"])),
        ("fixed fold", lambda r: r["selection"].update(fixed_folds={"cbf": 7})),
        ("role", lambda r: r["models"]["cbf"].update(role="other")),
    ]
    import copy

    for _, change in bad:
        r = copy.deepcopy(raw)
        change(r)
        with pytest.raises(ConfigError):
            parse_config(r, tmp_path)


def test_epochs_per_model(cfg):
    assert cfg.epochs(cfg.models["cbf_t2s"]) == 750
    assert cfg.epochs(cfg.models["degree_centrality"]) == 750  # same setup as the other models
    assert cfg.models["degree_centrality"].augmentation == "default"


def test_excluded_participant_leaves_the_other_folds_unchanged(cfg):
    main = cfg.cohorts["main"]
    assert main.exclude_ids == ("sub-stab035",) and len(main.used_ids) == 54
    folds = cohort_folds(main, 4, 1234)
    all_folds = make_folds(main.train_ids, 4, 1234)
    assert not any("sub-stab035" in tr + va for tr, va in folds)
    assert "sub-stab035" in all_folds[3][1]  # it was a fold-4 validation participant
    for (tr, va), (tr0, va0) in zip(folds, all_folds, strict=True):
        assert tr == [s for s in tr0 if s != "sub-stab035"] and va == [s for s in va0 if s != "sub-stab035"]


def test_exclude_ids_must_be_training_participants(raw, tmp_path):
    raw["cohorts"]["main"]["exclude_ids"] = ["sub-unknown"]
    with pytest.raises(ConfigError, match="exclude_ids"):
        parse_config(raw, root=tmp_path)


def test_out_of_fold_dataset(raw, tmp_path):
    cfg = parse_config(raw, tmp_path)
    ds = cfg.datasets["training_oof"]
    assert ds.out_of_fold and not ds.export_voxels and ds.models == ("cbf_t2s",)
    assert len(ds.ids) == 54 and "sub-stab035" not in ds.ids
    import copy

    r = copy.deepcopy(raw)
    r["testing"]["datasets"]["training_oof"]["ids"] = ["sub-stab035"]  # excluded from training: no fold model
    with pytest.raises(ConfigError, match="not used in training"):
        parse_config(r, tmp_path)
    r = copy.deepcopy(raw)
    del r["testing"]["datasets"]["training_oof"]["output_dir"]
    with pytest.raises(ConfigError, match="output_dir"):
        parse_config(r, tmp_path)
    r = copy.deepcopy(raw)
    r["testing"]["datasets"]["training_oof"]["models"] = ["cbf_t2s", "degree_centrality"]
    with pytest.raises(ConfigError, match="one cohort"):
        parse_config(r, tmp_path)
