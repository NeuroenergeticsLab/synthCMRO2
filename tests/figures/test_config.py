"""Figure configurations: {tag} points the individual configuration at one export of the model part; the public
configuration reads the shareable tables in data/public/."""

from pathlib import Path

import pytest
import yaml

from synthcmro2.figures.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path, tag):
    raw = yaml.safe_load((ROOT / "config" / "figures_individual.yaml").read_text(encoding="utf8"))
    raw["paths"]["tag"] = tag
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "figures.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf8")
    return path


def test_tag_in_config_and_override(tmp_path):
    path = _write(tmp_path, "20261107")
    cfg = load_config(path)
    assert cfg.participant_metrics == tmp_path / "results" / "export" / "synthcmro2_stats_df_20261107.csv"
    assert cfg.retest_voxel_values.name == "cmro2_vox_df_20261107_test_retest.csv"
    assert cfg.pls_dir == tmp_path / "results" / "pls_20261107"
    assert load_config(path, "20261201").voxel_values.name == "cmro2_vox_df_20261201.csv"


def test_missing_tag_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="paths.tag"):
        load_config(_write(tmp_path, None))


def test_save_figure_dpi_default_and_override(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from synthcmro2.figures.plotting import save_figure

    def png_width(dpi):
        fig = plt.figure(figsize=(2, 1), dpi=100)
        fig.add_subplot().plot([0, 1])
        (path,) = save_figure(fig, tmp_path / str(dpi), "f", ["png"], dpi)
        return Image.open(path).size[0]

    default = png_width(None)
    assert png_width(300) > 2 * default


def test_public_configuration_reads_the_shared_tables():
    cfg = load_config(ROOT / "config" / "figures.yaml")
    assert cfg.voxel_values == ROOT / "data" / "public" / "group_voxel_values.csv.gz"
    assert cfg.has_parcel_icc and cfg.age_calibration.exists() and cfg.parcel_sd.exists()
    assert not cfg.exclude_parcels  # the shared tables already exclude the outlying parcels
