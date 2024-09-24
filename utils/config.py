import sys
import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf


def setup_config_and_logging(name: str, base_log_dir: str = "logs"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, required=True)
    parser.add_argument("--base_log_dir", "-l", type=str, default="logs")
    parser.add_argument("--resume_ckpt", "-r", type=str)
    parser.add_argument("--options", "-o", nargs="+", default=[])

    args = parser.parse_args()

    config_path = Path(args.config_path)
    config = OmegaConf.load(config_path)
    config.merge_with_dotlist(args.options)

    log_dir = Path(args.base_log_dir) / config_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    OmegaConf.save(config=config, f=log_dir / "config.yaml")
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / f"{name}.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    
    return config, log_dir, args.resume_ckpt


def load_config_and_logging(name: str, return_args = False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-l", type=str)

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    config = OmegaConf.load(log_dir / "config.yaml")

    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / f"{name}.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    if return_args:
        return config, log_dir, args
    return config, log_dir