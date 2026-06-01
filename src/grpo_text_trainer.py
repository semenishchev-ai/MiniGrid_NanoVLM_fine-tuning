import copy
from pathlib import Path
import torch
from torch.optim import AdamW
from src.env import MiniGridWrapper
from src.rollout_text import sample_episode_text, compute_log_probs_text


def _extract_sr(metrics, primary_env=None):
    if "success_rate" in metrics:
        return metrics["success_rate"]
    if primary_env and primary_env in metrics:
        return metrics[primary_env].get("success_rate", 0.0)
    first = next(iter(metrics.values()))
    return first.get("success_rate", 0.0)


def train_grpo_text(
    model, tokenizer, image_processor, eval_fn, cfg, device, ckpt_dir,
):
    c = cfg["grpo"]
    verbose_every = c.get("verbose_every", 0)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ref_model = copy.deepcopy(model).to(device)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

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

    ent_beta = c.get("ent_beta", 0.0)
    prompt = c["prompt"]
    max_new_tokens = c["max_new_tokens"]

    try:
        for iteration in range(1, c["num_iterations"] + 1):
            model.eval()
            base_seed = c.get("rollout_seed_start", 0) + iteration * c["group_size"]
            episodes = []
            for g in range(c["group_size"]):
                ep = sample_episode_text(
                    model, tokenizer, image_processor, env, device,
                    prompt=prompt,
                    max_steps=c["max_episode_steps"],
                    max_new_tokens=max_new_tokens,
                    temperature=c["temperature"],
                    seed=base_seed + g,
                )
                episodes.append(ep)

            rewards = torch.tensor([e["total_reward"] for e in episodes])
            lengths = [e["length"] for e in episodes]
            parse_fails = sum(e["parse_fails"] for e in episodes)
            total_steps = sum(lengths)
            parse_fail_rate = parse_fails / max(1, total_steps)
            mean_gen = (
                sum(e["mean_gen_tokens"] * e["length"] for e in episodes)
                / max(1, total_steps)
            )
            mean_r = rewards.mean().item()
            std_r = rewards.std().item()
            if std_r < 1e-8:
                advantages = torch.zeros_like(rewards)
            else:
                advantages = (rewards - mean_r) / (std_r + 1e-8)

            if verbose_every > 0 and iteration % verbose_every == 0:
                ep = episodes[0]
                print(f"  --- sample rollout (iter {iteration}, reward={ep['total_reward']:.2f}, "
                    f"success={ep['success']}, len={ep['length']}) ---")
                for t, (text, act) in enumerate(ep["gen_texts"]):
                    text_clean = text.replace("\n", " ").strip()
                    if len(text_clean) > 120:
                        text_clean = text_clean[:117] + "..."
                    print(f"    step {t}: action={act} | gen=\"{text_clean}\"")
                print(f"  --- end sample ---")

            model.train()
            iter_policy_loss = 0.0
            iter_kl = 0.0
            iter_entropy = 0.0
            n_updates = 0

            for inner in range(c["inner_epochs"]):
                for g, ep in enumerate(episodes):
                    adv = advantages[g].to(device)
                    T = len(ep["generated_tokens"])
                    if T == 0:
                        continue
                    for t in range(T):
                        gen_ids = ep["generated_tokens"][t]
                        old_lp = ep["old_log_probs"][t].to(device)
                        if gen_ids.numel() == 0:
                            continue
                        image_t = ep["images"][t]
                        new_lp, ent = compute_log_probs_text(
                            model, image_t,
                            ep["prompt_input_ids"], ep["prompt_attention_mask"],
                            gen_ids, device, with_grad=True,
                        )
                        ref_lp, _ = compute_log_probs_text(
                            ref_model, image_t,
                            ep["prompt_input_ids"], ep["prompt_attention_mask"],
                            gen_ids, device, with_grad=False,
                        )
                        ratio = torch.exp(new_lp - old_lp)
                        clipped = torch.clamp(
                            ratio, 1 - c["clip_eps"], 1 + c["clip_eps"],
                        )
                        policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                        log_ratio = ref_lp - new_lp
                        kl = (log_ratio.exp() - 1 - log_ratio).mean()
                        entropy = ent.mean()
                        loss = (
                            policy_loss
                            + c["kl_beta"] * kl
                            - ent_beta * entropy
                        )
                        optim.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), c["max_grad_norm"],
                        )
                        optim.step()
                        iter_policy_loss += policy_loss.item()
                        iter_kl += kl.item()
                        iter_entropy += entropy.item()
                        n_updates += 1

            n_updates = max(1, n_updates)
            record = {
                "iteration": iteration,
                "mean_reward": mean_r,
                "reward_std": std_r,
                "mean_length": sum(lengths) / len(lengths),
                "policy_loss": iter_policy_loss / n_updates,
                "kl": iter_kl / n_updates,
                "entropy": iter_entropy / n_updates,
                "parse_fail_rate": parse_fail_rate,
                "mean_gen_tokens": mean_gen,
            }
            print(
                f"iter {iteration:3d} | reward {mean_r:.3f}±{std_r:.3f} | "
                f"len {record['mean_length']:.2f} | "
                f"gen_tok {mean_gen:.1f} | "
                f"parse_fail {parse_fail_rate:.2%} | "
                f"pl {record['policy_loss']:.4f} | "
                f"kl {record['kl']:.4f} | "
                f"ent {record['entropy']:.3f}"
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
                    sr_parts.append(
                        f"{env_name}: SR={env_m['success_rate']:.2f}, "
                        f"len={env_m['mean_length']:.1f}"
                    )
                print("           eval | " + " | ".join(sr_parts))
                if c.get("save_best", True) and sr > best_sr:
                    best_sr = sr
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "iteration": iteration,
                            "success_rate": sr,
                        },
                        ckpt_dir / c["ckpt_name"],
                    )
            history.append(record)
    finally:
        env.close()

    if best_sr < 0:
        torch.save(
            {"model": model.state_dict(), "iteration": c["num_iterations"]},
            ckpt_dir / c["ckpt_name"],
        )
    return history