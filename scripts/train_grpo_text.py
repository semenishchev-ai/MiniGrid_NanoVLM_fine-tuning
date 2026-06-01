import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.config import merge_configs
from src.utils import set_seed, ensure_dir
from src.model import load_vlm
from src.grpo_text_trainer import train_grpo_text
from src.evaluate import evaluate_policy_text_multi
from src.nanovlm_path import setup_nanovlm_import

setup_nanovlm_import()


def make_eval_fn(tokenizer, image_processor, cfg, device):
    env_specs = cfg["grpo"]["eval_envs"]
    prompt = cfg["grpo"]["prompt"]
    max_new_tokens = cfg["grpo"]["max_new_tokens"]

    def _fn(model):
        return evaluate_policy_text_multi(
            model, tokenizer, image_processor, device,
            env_specs=env_specs,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
        )
    return _fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = merge_configs(args.base, args.config)
    set_seed(cfg["seed"])

    model, tokenizer, image_processor, device = load_vlm(
        hf_repo=cfg["model"]["hf_repo"],
        device=cfg["device"],
    )

    sft_ckpt = cfg["grpo"]["sft_checkpoint"]
    if not Path(sft_ckpt).is_file():
        raise FileNotFoundError(f"SFT checkpoint not found: {sft_ckpt}")
    print(f"loading SFT checkpoint: {sft_ckpt}")
    state = torch.load(sft_ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    ckpt_dir = ensure_dir(cfg["grpo"]["output_dir"])
    results_dir = ensure_dir(cfg["results"]["dir"])
    eval_fn = make_eval_fn(tokenizer, image_processor, cfg, device)

    history = train_grpo_text(
        model, tokenizer, image_processor, eval_fn, cfg, device, ckpt_dir,
    )

    history_name = cfg["grpo"].get("history_name", "grpo_text_history.json")
    out_path = Path(results_dir) / history_name
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved history: {out_path}")


if __name__ == "__main__":
    main()