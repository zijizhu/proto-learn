import sys
import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf


def setup_config_and_logging(base_log_dir: str = "logs"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str)

    args = parser.parse_args()

    config_path = Path(args.config_path)
    config = OmegaConf.load(config_path)

    log_dir = Path(base_log_dir)
    log_dir = log_dir / config_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    OmegaConf.save(config=config, f=log_dir / "config.yaml")
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    
    return config, log_dir
