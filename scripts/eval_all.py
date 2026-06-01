import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.config import merge_configs
from src.model import load_vlm
from src.evaluate import evaluate_policy_multi, evaluate_policy_text_multi
from src.nanovlm_path import setup_nanovlm_import

setup_nanovlm_import()


EVAL_ENVS = [
    {
        "name": "MiniGrid-Empty-Random-6x6-v0",
        "max_steps": 20,
        "num_episodes": 50,
        "seed_start": 30000,
    },
    {
        "name": "MiniGrid-Empty-8x8-v0",
        "max_steps": 50,
        "num_episodes": 50,
        "seed_start": 30000,
    },
]

METHODS = [
    {
        "name": "zero_shot",
        "mode": "action",
        "checkpoint": None,  # без загрузки чекпоинта — pure HF model
        "prompt": "What action should the agent take? Choose: left, right, forward. Answer:",
    },
    {
        "name": "sft_baseline",
        "mode": "action",
        "checkpoint": "checkpoints/sft_baseline/sft_baseline_best.pt",
        "prompt": "What action should the agent take? Choose: left, right, forward. Answer:",
    },
    {
        "name": "sft_improved",
        "mode": "action",
        "checkpoint": "checkpoints/sft_improved/sft_improved_best.pt",
        "prompt": "What action should the agent take? Choose: left, right, forward. Answer:",
    },
    {
        "name": "grpo_action",
        "mode": "action",
        "checkpoint": "checkpoints/grpo_action/grpo_action_best.pt",
        "prompt": "What action should the agent take? Choose: left, right, forward. Answer:",
    },
    {
        "name": "grpo_text_v1",
        "mode": "text",
        "checkpoint": "checkpoints/grpo_text_v1/grpo_text_v1_best.pt",
        "prompt": "Describe what you see, then choose action: left, right, forward.",
        "max_new_tokens": 48,
    },
    {
        "name": "grpo_text_v2",
        "mode": "text",
        "checkpoint": "checkpoints/grpo_text_v2/grpo_text_v2_best.pt",
        "prompt": (
            "Think step by step about where the goal is, "
            "then output action: left, right, forward."
        ),
        "max_new_tokens": 48,
    },
]


def evaluate_method(method, cfg, tokenizer, image_processor, device):
    model, _, _, _ = load_vlm(
        hf_repo=cfg["model"]["hf_repo"],
        device=cfg["device"],
    )
    if method["checkpoint"] is not None:
        ckpt_path = Path(method["checkpoint"])
        if not ckpt_path.is_file():
            print(f"  WARNING: checkpoint not found: {ckpt_path}, skipping")
            return None
        print(f"  loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    else:
        print(f"  using zero-shot (no checkpoint)")
    model.eval()

    if method["mode"] == "action":
        results = evaluate_policy_multi(
            model, tokenizer, image_processor, device,
            env_specs=EVAL_ENVS,
            prompt=method["prompt"],
        )
    elif method["mode"] == "text":
        results = evaluate_policy_text_multi(
            model, tokenizer, image_processor, device,
            env_specs=EVAL_ENVS,
            prompt=method["prompt"],
            max_new_tokens=method["max_new_tokens"],
            temperature=1.0,
        )
    else:
        raise ValueError(f"unknown mode: {method['mode']}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--out", default="results/eval_all.json")
    p.add_argument(
        "--methods", nargs="*", default=None,
        help="имена методов из METHODS; по умолчанию все",
    )
    args = p.parse_args()
    cfg = merge_configs(args.base, args.base)

    methods_to_run = METHODS
    if args.methods:
        methods_to_run = [m for m in METHODS if m["name"] in args.methods]
        if not methods_to_run:
            print(f"no methods match: {args.methods}")
            return

    _, tokenizer, image_processor, device = load_vlm(
        hf_repo=cfg["model"]["hf_repo"],
        device=cfg["device"],
    )

    all_results = {}
    for method in methods_to_run:
        print(f"\n[evaluating {method['name']} (mode={method['mode']})]")
        try:
            res = evaluate_method(
                method, cfg, tokenizer, image_processor, device,
            )
            if res is None:
                continue
            all_results[method["name"]] = {
                "mode": method["mode"],
                "prompt": method["prompt"],
                "results": res,
            }
            for env_name, m in res.items():
                extra = ""
                if method["mode"] == "text":
                    extra = (
                        f", parse_fail={m.get('parse_fail_rate', 0):.2%}"
                        f", gen_tok={m.get('mean_gen_tokens', 0):.1f}"
                    )
                print(
                    f"  [{env_name}] SR={m['success_rate']:.2f}, "
                    f"len={m['mean_length']:.2f}{extra}"
                )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()