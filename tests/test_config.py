from copy import deepcopy
from pathlib import Path

from utils.config import load_config, validate_config


def test_v2_outputs_are_isolated_from_v1() -> None:
    v1_config = load_config("config/default.yaml")
    v2_config = load_config("config/v2_teacher.yaml")

    for key in (
        "output_dir",
        "checkpoint_dir",
        "sample_dir",
        "plot_dir",
        "evaluation_dir",
        "log_dir",
    ):
        assert Path(v2_config["paths"][key]) != Path(v1_config["paths"][key])


def test_config_rejects_attention_at_unknown_resolution() -> None:
    config = deepcopy(load_config("config/v2_teacher.yaml"))
    config["model"]["attention_resolutions"] = [32, 7]

    try:
        validate_config(config)
    except ValueError as error:
        assert "不存在的分辨率" in str(error)
    else:
        raise AssertionError("不存在的 Attention 分辨率应抛出 ValueError")
