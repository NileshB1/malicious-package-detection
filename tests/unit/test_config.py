import pytest

from pkgtriage.config import PROJECT_ROOT, load_config


def test_loads_and_resolves_paths(cfg):
    assert cfg.dataset.revision.startswith("620d102f")
    assert len(cfg.dataset.sha256) == 64
    assert cfg.paths.images_dir == PROJECT_ROOT / "images"
    assert cfg.split.train + cfg.split.validation + cfg.split.test == pytest.approx(1)


def test_rejects_bad_split(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text(
        (PROJECT_ROOT / "configs" / "config.yaml").read_text().replace("test: 0.15", "test: 0.30")
    )
    with pytest.raises(ValueError, match="add up to 1"):
        load_config(bad)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
