"""Input modalities and the file names the notebooks used for each input configuration.

The notebooks named a model by the inputs it omits, among CBV, CBF, T2, T2*, and T1, for example
``_OmitCBV_OmitT2_OmitT1`` for the CBF and T2* model. The degree centrality model omits all five and is
therefore ``_OmitCBV_OmitCBF_OmitT2_OmitT2S_OmitT1``. These names are kept so that the trained
checkpoints load unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

MODALITIES = ("CBV", "CBF", "T2", "T2S", "T1", "fMRI")  # channel order expected by the checkpoints
NAMED = MODALITIES[:5]  # modalities that appear in file names

def check_inputs(inputs: Iterable[str]) -> tuple[str, ...]:
    """Validate an input list: known modalities, no duplicates, channel order, fMRI only on its own."""
    inputs = tuple(inputs)
    unknown = [m for m in inputs if m not in MODALITIES]
    if unknown:
        raise ValueError(f"unknown modalities {unknown}; expected a subset of {MODALITIES}")
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError(f"inputs must be non-empty and unique: {inputs}")
    if list(inputs) != sorted(inputs, key=MODALITIES.index):
        raise ValueError(f"inputs must follow the channel order {MODALITIES}: {inputs}")
    if "fMRI" in inputs and len(inputs) > 1:
        raise ValueError("fMRI (degree centrality) is only supported as the single input")
    return inputs


def checkpoint_name(inputs: Iterable[str]) -> str:
    """('CBF', 'T2S') -> '_OmitCBV_OmitT2_OmitT1'."""
    inputs = check_inputs(inputs)
    return "".join(f"_Omit{m}" for m in NAMED if m not in inputs)


def inputs_from_name(name: str) -> tuple[str, ...]:
    """'_OmitCBV_OmitT2_OmitT1' -> ('CBF', 'T2S'); all five omitted -> ('fMRI',)."""
    parts = [p for p in name.split("_") if p]
    if not all(p.startswith("Omit") and p.removeprefix("Omit") in NAMED for p in parts):
        raise ValueError(f"not a model file name: {name!r}")
    omitted = {p.removeprefix("Omit") for p in parts}
    return tuple(m for m in NAMED if m not in omitted) or ("fMRI",)
