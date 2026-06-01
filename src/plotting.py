import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def _detect_env_names(history):
    for record in history:
        if "eval" in record and isinstance(record["eval"], dict):
            return list(record["eval"].keys())
    return []


def _extract_for_env(record, env_name):
    if "eval" in record and isinstance(record["eval"], dict):
        env_metrics = record["eval"].get(env_name, {})
        return (
            env_metrics.get("success_rate"),
            env_metrics.get("mean_return"),
            env_metrics.get("mean_length"),
        )
    return (None, None, None)


def plot_sft_curves(history_path, out_dir, out_name=None):
    history_path = Path(history_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_path.read_text())
    if out_name is None:
        out_name = f"{history_path.stem}_curves.png"
    epochs = [r["epoch"] for r in history]
    losses = [r["train_loss"] for r in history]
    env_names = _detect_env_names(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].set_title("SFT train loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    for env_name in env_names:
        srs = [_extract_for_env(r, env_name)[0] for r in history]
        if any(s is not None for s in srs):
            axes[1].plot(epochs, srs, marker="o", label=env_name)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("SFT eval success rate")
    axes[1].grid(True, alpha=0.3)
    if env_names:
        axes[1].legend(fontsize=8)
    for env_name in env_names:
        lens = [_extract_for_env(r, env_name)[2] for r in history]
        rets = [_extract_for_env(r, env_name)[1] for r in history]
        if any(l is not None for l in lens):
            axes[2].plot(epochs, lens, marker="o", label=f"length ({env_name})")
        if any(r is not None for r in rets):
            axes[2].plot(epochs, rets, marker="s", label=f"return ({env_name})")
    axes[2].set_xlabel("epoch")
    axes[2].set_title("Episode stats")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    out = out_dir / out_name
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_sft_comparison(
    baseline_path, improved_path, out_dir,
    out_name="sft_comparison.png",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path(baseline_path).read_text())
    improved = json.loads(Path(improved_path).read_text())
    env_names = _detect_env_names(improved) or _detect_env_names(baseline)
    if not env_names:
        raise ValueError("No eval data found in history files")
    n_envs = len(env_names)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 4), squeeze=False)
    axes = axes[0]
    base_epochs = [r["epoch"] for r in baseline]
    imp_epochs = [r["epoch"] for r in improved]
    for i, env_name in enumerate(env_names):
        base_sr = [_extract_for_env(r, env_name)[0] for r in baseline]
        imp_sr = [_extract_for_env(r, env_name)[0] for r in improved]
        if any(s is not None for s in base_sr):
            axes[i].plot(base_epochs, base_sr, marker="o",
                         label="baseline", color="tab:blue")
        if any(s is not None for s in imp_sr):
            axes[i].plot(imp_epochs, imp_sr, marker="s",
                         label="improved", color="tab:green")
        axes[i].set_title(env_name, fontsize=10)
        axes[i].set_xlabel("epoch")
        axes[i].set_ylabel("success rate")
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
    fig.suptitle("SFT: baseline vs improved", y=1.02)
    fig.tight_layout()
    out = out_dir / out_name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_grpo_curves(history_path, out_dir, out_name=None):
    history_path = Path(history_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_path.read_text())
    if out_name is None:
        out_name = f"{history_path.stem}_curves.png"

    iters = [r["iteration"] for r in history]
    rewards = [r["mean_reward"] for r in history]
    reward_stds = [r.get("reward_std", 0.0) for r in history]
    lengths = [r["mean_length"] for r in history]
    kls = [r["kl"] for r in history]
    pl = [r["policy_loss"] for r in history]
    eval_envs = _detect_env_names(history)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(iters, rewards, marker="o", color="tab:blue")
    axes[0, 0].fill_between(
        iters,
        [r - s for r, s in zip(rewards, reward_stds)],
        [r + s for r, s in zip(rewards, reward_stds)],
        alpha=0.2, color="tab:blue",
    )
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("mean reward")
    axes[0, 0].set_title("Rollout mean reward (±std)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, lengths, marker="o", color="tab:blue")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("mean length")
    axes[0, 1].set_title("Rollout mean episode length")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(iters, kls, marker="o", color="tab:red")
    axes[0, 2].set_xlabel("iteration")
    axes[0, 2].set_ylabel("KL")
    axes[0, 2].set_title("KL to reference policy")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, pl, marker="o", color="tab:purple")
    axes[1, 0].axhline(0.0, color="gray", lw=0.5, alpha=0.5)
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("policy loss")
    axes[1, 0].set_title("Policy loss")
    axes[1, 0].grid(True, alpha=0.3)

    for env_name in eval_envs:
        ev_iters, ev_srs = [], []
        for r in history:
            if "eval" in r and env_name in r["eval"]:
                ev_iters.append(r["iteration"])
                ev_srs.append(r["eval"][env_name]["success_rate"])
        if ev_iters:
            axes[1, 1].plot(ev_iters, ev_srs, marker="o", label=env_name)
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("success rate")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title("Eval success rate")
    axes[1, 1].grid(True, alpha=0.3)
    if eval_envs:
        axes[1, 1].legend(fontsize=8)

    for env_name in eval_envs:
        ev_iters, ev_lens = [], []
        for r in history:
            if "eval" in r and env_name in r["eval"]:
                ev_iters.append(r["iteration"])
                ev_lens.append(r["eval"][env_name]["mean_length"])
        if ev_iters:
            axes[1, 2].plot(ev_iters, ev_lens, marker="o", label=env_name)
    axes[1, 2].set_xlabel("iteration")
    axes[1, 2].set_ylabel("mean length")
    axes[1, 2].set_title("Eval mean episode length")
    axes[1, 2].grid(True, alpha=0.3)
    if eval_envs:
        axes[1, 2].legend(fontsize=8)

    fig.tight_layout()
    out = out_dir / out_name
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


METHOD_ORDER = [
    "zero_shot",
    "sft_baseline",
    "sft_improved",
    "grpo_action",
    "grpo_text_v1",
    "grpo_text_v2",
]
METHOD_LABELS = {
    "zero_shot": "Zero-shot",
    "sft_baseline": "SFT baseline",
    "sft_improved": "SFT improved",
    "grpo_action": "GRPO action",
    "grpo_text_v1": "GRPO text (v1: describe)",
    "grpo_text_v2": "GRPO text (v2: step-by-step)",
}
ENV_LABELS = {
    "MiniGrid-Empty-Random-6x6-v0": "Empty-Random-6x6 (in-dist)",
    "MiniGrid-Empty-8x8-v0": "Empty-8x8 (OOD)",
}


def _ordered_methods(results):
    return [m for m in METHOD_ORDER if m in results]


def plot_final_sr(results, out_path):
    methods = _ordered_methods(results)
    env_names = list(next(iter(results.values()))["results"].keys())
    x = np.arange(len(methods))
    width = 0.8 / len(env_names)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, env in enumerate(env_names):
        vals = [results[m]["results"][env]["success_rate"] for m in methods]
        offset = (i - (len(env_names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=ENV_LABELS.get(env, env))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       rotation=15, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Final Success Rate by Method and Environment")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_final_length(results, out_path):
    methods = _ordered_methods(results)
    env_names = list(next(iter(results.values()))["results"].keys())
    x = np.arange(len(methods))
    width = 0.8 / len(env_names)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, env in enumerate(env_names):
        vals = [results[m]["results"][env]["mean_length"] for m in methods]
        offset = (i - (len(env_names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=ENV_LABELS.get(env, env))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       rotation=15, ha="right")
    ax.set_ylabel("Mean episode length (steps)")
    ax.set_title("Mean Episode Length (lower = more efficient)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_grpo_eval_curves_combined(histories, out_path, env_name=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, history in histories.items():
        iters, sr_vals = [], []
        for h in history:
            if "eval" not in h:
                continue
            ev = h["eval"]
            if env_name and env_name in ev:
                sr_vals.append(ev[env_name]["success_rate"])
                iters.append(h["iteration"])
            elif not env_name:
                srs = [v["success_rate"] for v in ev.values()]
                sr_vals.append(sum(srs) / len(srs))
                iters.append(h["iteration"])
        if iters:
            ax.plot(iters, sr_vals,
                    label=METHOD_LABELS.get(name, name),
                    marker="o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Eval Success Rate")
    title_suffix = f" on {ENV_LABELS.get(env_name, env_name)}" if env_name else " (avg)"
    ax.set_title(f"GRPO eval SR over training{title_suffix}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_text_reasoning_stats(histories, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    has_data = False
    for name, history in histories.items():
        if "text" not in name:
            continue
        iters = [h["iteration"] for h in history]
        gen_tok = [h.get("mean_gen_tokens", 0) for h in history]
        parse_fail = [h.get("parse_fail_rate", 0) for h in history]
        label = METHOD_LABELS.get(name, name)
        axes[0].plot(iters, gen_tok, label=label, marker="o", markersize=3)
        axes[1].plot(iters, parse_fail, label=label, marker="o", markersize=3)
        has_data = True
    if not has_data:
        plt.close(fig)
        return None
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean generated tokens / step")
    axes[0].set_title("Reasoning length over training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Parse fail rate")
    axes[1].set_title("Action parse failure rate over training")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_summary_table(results, out_path):
    methods = _ordered_methods(results)
    env_names = list(next(iter(results.values()))["results"].keys())
    lines = []
    header = ["Method"]
    for env in env_names:
        header += [f"{ENV_LABELS.get(env, env)} SR", "len"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in methods:
        row = [METHOD_LABELS.get(m, m)]
        for env in env_names:
            mr = results[m]["results"][env]
            row.append(f"{mr['success_rate']:.2f}")
            row.append(f"{mr['mean_length']:.1f}")
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(table + "\n")
    return table


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--history", default=None)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--out-name", default=None)
    p.add_argument("--compare-baseline", default=None)
    p.add_argument("--compare-improved", default=None)
    p.add_argument("--grpo-history", default=None)
    args = p.parse_args()

    if args.compare_baseline and args.compare_improved:
        out = plot_sft_comparison(
            args.compare_baseline, args.compare_improved, args.out_dir,
        )
        print(f"saved comparison: {out}")
    elif args.grpo_history:
        out = plot_grpo_curves(args.grpo_history, args.out_dir, out_name=args.out_name)
        print(f"saved: {out}")
    elif args.history:
        out = plot_sft_curves(args.history, args.out_dir, out_name=args.out_name)
        print(f"saved: {out}")
    else:
        raise SystemExit(
            "Specify --history, --grpo-history, "
            "or --compare-baseline + --compare-improved"
        )