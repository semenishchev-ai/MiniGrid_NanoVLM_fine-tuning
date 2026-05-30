import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_sft_curves(history_path, out_dir):
    history_path = Path(history_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_path.read_text())

    epochs = [r["epoch"] for r in history]
    losses = [r["train_loss"] for r in history]
    srs = [r.get("success_rate") for r in history]
    rets = [r.get("mean_return") for r in history]
    lens = [r.get("mean_length") for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title("SFT train loss"); axes[0].grid(True, alpha=0.3)

    if any(s is not None for s in srs):
        axes[1].plot(epochs, srs, marker="o", color="tab:green")
        axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("success rate")
    axes[1].set_title("SFT eval success rate"); axes[1].grid(True, alpha=0.3)

    if any(l is not None for l in lens):
        axes[2].plot(epochs, lens, marker="o", color="tab:orange", label="mean length")
        axes[2].plot(epochs, rets, marker="s", color="tab:red", label="mean return")
        axes[2].legend()
    axes[2].set_xlabel("epoch"); axes[2].set_title("Episode stats"); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "sft_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="results/sft_history.json")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    path = plot_sft_curves(args.history, args.out_dir)
    print(f"saved: {path}")