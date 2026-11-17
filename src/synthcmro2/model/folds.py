"""Participant-level cross-validation folds, identical to the training notebooks."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.model_selection import KFold

Fold = tuple[list[str], list[str]]  # (training participants, validation participants)


def make_folds(ids: Sequence[str], n_folds: int, seed: int) -> list[Fold]:
    """KFold(n_splits, shuffle=True, random_state=seed) over ``ids`` in the given order (fold 1 first)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [([ids[i] for i in tr], [ids[i] for i in va]) for tr, va in kf.split(list(ids))]


def cohort_folds(cohort, n_folds: int, seed: int) -> list[Fold]:
    """Folds of a configured cohort; stops if fold 1 differs from the one recorded in the training log.

    The split is made over all ``train_ids`` (so every other participant keeps its fold), and participants in
    ``exclude_ids`` are then removed from the training and validation sets of every fold.
    """
    folds = make_folds(cohort.train_ids, n_folds, seed)
    if cohort.fold1_validation is not None and list(cohort.fold1_validation) != folds[0][1]:
        raise RuntimeError(
            f"cohort {cohort.name}: fold 1 does not reproduce the training log; check the order of train_ids"
        )
    excluded = set(getattr(cohort, "exclude_ids", ()) or ())
    return [([s for s in tr if s not in excluded], [s for s in va if s not in excluded]) for tr, va in folds]
