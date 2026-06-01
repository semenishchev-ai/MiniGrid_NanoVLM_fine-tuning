import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.config import merge_configs
from src.model import load_vlm
from src.evaluate import evaluate_policy_text
from src.nanovlm_path import setup_nanovlm_import

setup_nanovlm_import()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = merge_configs(args.base, args.config)
    model, tokenizer, image_processor, device = load_vlm(
        hf_repo=cfg["model"]["hf_repo"],
        device=cfg["device"],
    )

    print(f"loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    metrics = evaluate_policy_text(
        model, tokenizer, image_processor, device,
        env_name=args.env, max_steps=args.max_steps,
        num_episodes=args.episodes, seed_start=args.seed_start,
        prompt=cfg["grpo"]["prompt"],
        max_new_tokens=cfg["grpo"]["max_new_tokens"],
        temperature=args.temperature,
    )
    print(json.dumps(metrics, indent=2))

    if args.out is None:
        ckpt_stem = Path(args.checkpoint).stem
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"{ckpt_stem}_eval_{args.env}.json"
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()