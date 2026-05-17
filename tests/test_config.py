from power_plant_anomaly.config import figures_dir, load_config, tables_dir
from power_plant_anomaly.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT


def test_project_root_exists():
    assert PROJECT_ROOT.is_dir()


def test_default_config_loads():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert "data" in config
    assert "output" in config


def test_output_dirs_created(tmp_path, monkeypatch):
    config = {
        "output": {
            "root": str(tmp_path / "output"),
            "figures_dir": "figures",
            "tables_dir": "tables",
        }
    }
    fig = figures_dir(config)
    tbl = tables_dir(config)
    assert fig.is_dir()
    assert tbl.is_dir()
