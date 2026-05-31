import copy
import math
from pathlib import Path

import torch
from torch.optim import AdamW

from src.env import MiniGridWrapper
from src.rollout import sample_episode, compute_log_probs, get_action_token_ids


def _extract_sr(metrics, primary_env=None):
    if "success_rate" in metrics:
        return metrics["success_rate"]
    if primary_env and primary_env in metrics:
        return metrics[primary_env].get("success_rate", 0.0)
    first = next(iter(metrics.values()))
    return first.get("success_rate", 0.0)


def train_grpo(
    model, tokenizer, image_processor, eval_fn, cfg, device, ckpt_dir,
):
    c = cfg["grpo"]
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ref_model = copy.deepcopy(model).to(device)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    action_token_ids = get_action_token_ids(tokenizer)
    print(f"[grpo] action token ids: {action_token_ids}")

    env = MiniGridWrapper(
        env_name=c["env_name"],
        max_steps=c["max_episode_steps"],
    )
    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=c["lr"],
        weight_decay=c.get("weight_decay", 0.0),
    )

    primary_env = c.get("primary_eval_env")
    best_sr = -1.0
    history = []

    try:
        for iteration in range(1, c["num_iterations"] + 1):
            model.eval()
            base_seed = c.get("rollout_seed_start", 0) + iteration * c["group_size"]
            episodes = []
            for g in range(c["group_size"]):
                ep = sample_episode(
                    model, tokenizer, image_processor, env, device,
                    prompt=cfg["model"]["prompt"],
                    action_token_ids=action_token_ids,
                    max_steps=c["max_episode_steps"],
                    temperature=c["temperature"],
                    seed=base_seed + g,
                )
                episodes.append(ep)

            rewards = torch.tensor([e["total_reward"] for e in episodes])
            lengths = [e["length"] for e in episodes]
            mean_r = rewards.mean().item()
            std_r = rewards.std().item()

            if std_r < 1e-8:
                print(f"iter {iteration}: reward_std=0 (mean={mean_r:.3f}), skip update")
                advantages = torch.zeros_like(rewards)
            else:
                advantages = (rewards - mean_r) / (std_r + 1e-8)

            model.train()
            iter_policy_loss = 0.0
            iter_kl = 0.0
            n_updates = 0

            for inner in range(c["inner_epochs"]):
                for g, ep in enumerate(episodes):
                    if ep["action_tokens"].numel() == 0:
                        continue
                    adv = advantages[g].to(device)
                    new_log_probs = compute_log_probs(
                        model, ep["images"],
                        ep["prompt_input_ids"], ep["prompt_attention_mask"],
                        ep["action_tokens"], action_token_ids, device,
                        with_grad=True,
                    )
                    ref_log_probs = compute_log_probs(
                        ref_model, ep["images"],
                        ep["prompt_input_ids"], ep["prompt_attention_mask"],
                        ep["action_tokens"], action_token_ids, device,
                        with_grad=False,
                    )
                    old_log_probs = ep["old_log_probs"].to(device)

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    clipped = torch.clamp(ratio, 1 - c["clip_eps"], 1 + c["clip_eps"])
                    policy_loss = -torch.min(ratio * adv, clipped * adv).mean()

                    log_ratio = ref_log_probs - new_log_probs
                    kl = (log_ratio.exp() - 1 - log_ratio).mean()
                    loss = policy_loss + c["kl_beta"] * kl

                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c["max_grad_norm"])
                    optim.step()

                    iter_policy_loss += policy_loss.item()
                    iter_kl += kl.item()
                    n_updates += 1

            n_updates = max(1, n_updates)
            record = {
                "iteration": iteration,
                "mean_reward": mean_r,
                "reward_std": std_r,
                "mean_length": sum(lengths) / len(lengths),
                "policy_loss": iter_policy_loss / n_updates,
                "kl": iter_kl / n_updates,
            }
            print(
                f"iter {iteration:3d} | reward {mean_r:.3f}±{std_r:.3f} | "
                f"len {record['mean_length']:.2f} | "
                f"policy_loss {record['policy_loss']:.4f} | "
                f"kl {record['kl']:.4f}"
            )

            do_eval = (
                eval_fn is not None
                and iteration % c.get("eval_every_iterations", 5) == 0
            )
            if do_eval:
                model.eval()
                with torch.no_grad():
                    metrics = eval_fn(model)
                record["eval"] = metrics
                sr = _extract_sr(metrics, primary_env)
                sr_parts = []
                for env_name, env_m in metrics.items():
                    sr_parts.append(f"{env_name}: SR={env_m['success_rate']:.2f}, len={env_m['mean_length']:.1f}")
                print("           eval | " + " | ".join(sr_parts))
                if c.get("save_best", True) and sr > best_sr:
                    best_sr = sr
                    torch.save(
                        {"model": model.state_dict(),
                         "iteration": iteration,
                         "success_rate": sr},
                        ckpt_dir / c["ckpt_name"],
                    )

            history.append(record)
    finally:
        env.close()

    # если best_sr так и не обновился всё равно сохраняем финал
    if best_sr < 0:
        torch.save(
            {"model": model.state_dict(), "iteration": c["num_iterations"]},
            ckpt_dir / c["ckpt_name"],
        )

    return history