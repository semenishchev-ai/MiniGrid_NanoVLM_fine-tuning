import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_collection import collect_dataset
from src.utils import ensure_dir, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg.get("data", {})
    env_cfg = cfg.get("env", {})
    output_dir = args.output_dir or data_cfg.get("dir", "data")
    num_episodes = args.num_episodes or data_cfg.get("num_episodes", 1000)
    start_seed = args.start_seed if args.start_seed is not None else cfg.get("seed", 42)

    ensure_dir(output_dir)
    collect_dataset(
        output_dir=output_dir,
        num_episodes=num_episodes,
        env_name=env_cfg.get("name", "MiniGrid-Empty-Random-6x6-v0"),
        max_steps=env_cfg.get("max_steps", 100),
        start_seed=start_seed,
    )
    print(f"Collected {num_episodes} episodes in {output_dir}/episodes")


if __name__ == "__main__":
    main()
