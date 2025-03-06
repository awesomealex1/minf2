import logging
import os
import sys
import torch

sys.path.append(os.getcwd())


if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # For running on MIG setup, otherwise devices aren't visible
    os.environ["WORLD_SIZE"] = "1"              # https://github.com/pytorch/pytorch/issues/126344
    torch.cuda.empty_cache()

from dotenv import load_dotenv

load_dotenv(".env")

import hydra
from omegaconf import OmegaConf

from src.configs import RunnerConfigs, register_base_configs
from src.runner import Runner
from src.utils import common_utils


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: RunnerConfigs) -> None:
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    print(configs)

    common_utils.setup_random_seed(configs.random_seed)

    runner = Runner(configs)
    runner.run()


if __name__ == "__main__":
    register_base_configs()
    main()
