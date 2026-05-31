import json
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
    """Кривые GRPO по итерациям: rollout-метрики, training loss, eval."""
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

    # row 0: rollout-метрики
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

    # row 1: training loss + eval
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