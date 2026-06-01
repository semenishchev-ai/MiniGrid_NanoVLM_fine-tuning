import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import (
    plot_sft_curves,
    plot_sft_comparison,
    plot_grpo_curves,
    plot_final_sr,
    plot_final_length,
    plot_grpo_eval_curves_combined,
    plot_text_reasoning_stats,
    make_summary_table,
)


def load_json(path):
    if path is None or not Path(path).is_file():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-all", default="results/eval_all.json")
    p.add_argument("--out-dir", default="results/plots")
    p.add_argument("--sft-baseline-history", default="results/sft_baseline_history.json")
    p.add_argument("--sft-improved-history", default="results/sft_improved_history.json")
    p.add_argument("--grpo-action-history", default="results/grpo_action_history.json")
    p.add_argument("--grpo-text-v1-history", default="results/grpo_text_v1_history.json")
    p.add_argument("--grpo-text-v2-history", default="results/grpo_text_v2_history.json")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, path in [
        ("sft_baseline", args.sft_baseline_history),
        ("sft_improved", args.sft_improved_history),
    ]:
        if Path(path).is_file():
            out = plot_sft_curves(path, out_dir, out_name=f"{name}_curves.png")
            print(f"saved: {out}")

    if Path(args.sft_baseline_history).is_file() and Path(args.sft_improved_history).is_file():
        out = plot_sft_comparison(
            args.sft_baseline_history,
            args.sft_improved_history,
            out_dir,
            out_name="sft_comparison.png",
        )
        print(f"saved: {out}")

    for name, path in [
        ("grpo_action", args.grpo_action_history),
        ("grpo_text_v1", args.grpo_text_v1_history),
        ("grpo_text_v2", args.grpo_text_v2_history),
    ]:
        if Path(path).is_file():
            out = plot_grpo_curves(path, out_dir, out_name=f"{name}_curves.png")
            print(f"saved: {out}")

    eval_all = load_json(args.eval_all)
    if eval_all is not None:
        out = plot_final_sr(eval_all, out_dir / "final_sr.png")
        print(f"saved: {out}")
        out = plot_final_length(eval_all, out_dir / "final_length.png")
        print(f"saved: {out}")
        tbl_path = out_dir / "summary_table.md"
        make_summary_table(eval_all, tbl_path)
        print(f"saved: {tbl_path}")
    else:
        print(f"WARNING: {args.eval_all} not found — skip final plots")

    grpo_hist = {}
    for name, path in [
        ("grpo_action", args.grpo_action_history),
        ("grpo_text_v1", args.grpo_text_v1_history),
        ("grpo_text_v2", args.grpo_text_v2_history),
    ]:
        h = load_json(path)
        if h is not None:
            grpo_hist[name] = h

    if grpo_hist:
        out = plot_grpo_eval_curves_combined(
            grpo_hist,
            out_dir / "grpo_eval_curves_6x6.png",
            env_name="MiniGrid-Empty-Random-6x6-v0",
        )
        print(f"saved: {out}")
        out = plot_grpo_eval_curves_combined(
            grpo_hist,
            out_dir / "grpo_eval_curves_8x8.png",
            env_name="MiniGrid-Empty-8x8-v0",
        )
        print(f"saved: {out}")
        text_hist = {k: v for k, v in grpo_hist.items() if "text" in k}
        if text_hist:
            out = plot_text_reasoning_stats(
                text_hist, out_dir / "text_reasoning_stats.png",
            )
            if out:
                print(f"saved: {out}")

    print(f"all plots saved to {out_dir}")


if __name__ == "__main__":
    main()