import torch
from PIL import Image

from src.env import MiniGridWrapper, NAME_TO_ACTION
from src.dataset import DEFAULT_PROMPT
from src.model import decode_action


@torch.no_grad()
def rollout_episode(
    model, tokenizer, image_processor, env, device,
    prompt=DEFAULT_PROMPT, max_new_tokens=5, seed=None, debug=False,
):
    obs, _ = env.reset(seed=seed)
    prompt_text = f"Question: {prompt} Answer:"
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    total_reward = 0.0
    steps = 0
    success = False
    last_action_name = None
    action_counts = {"left": 0, "right": 0, "forward": 0, "other": 0}
    while True:
        image = Image.fromarray(obs).convert("RGB")
        px = image_processor(image).unsqueeze(0).to(device)
        out_ids = model.generate(
            input_ids, px, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        new_tokens = out_ids[0]
        action_name = decode_action(tokenizer, new_tokens)
        last_action_name = action_name
        action = NAME_TO_ACTION.get(action_name)
        if action is None:
            action_counts["other"] += 1
            break
        action_counts[action_name] += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated:
            success = reward > 0.0
            break
        if truncated:
            break
    return {
        "success": success, "return": total_reward, "length": steps,
        "last_action": last_action_name, "action_counts": action_counts,
    }


@torch.no_grad()
def evaluate_policy(
    model, tokenizer, image_processor, device,
    env_name="MiniGrid-Empty-Random-6x6-v0",
    max_steps=100, num_episodes=20, seed_start=10_000,
    prompt=DEFAULT_PROMPT, verbose=True,
):
    was_training = model.training
    model.eval()
    env = MiniGridWrapper(env_name=env_name, max_steps=max_steps)
    successes, returns, lengths = 0, [], []
    total_counts = {"left": 0, "right": 0, "forward": 0, "other": 0}
    try:
        for i in range(num_episodes):
            r = rollout_episode(
                model, tokenizer, image_processor, env, device,
                prompt=prompt, seed=seed_start + i
            )
            successes += int(r["success"])
            returns.append(r["return"])
            lengths.append(r["length"])
            for k, v in r["action_counts"].items():
                total_counts[k] += v
    finally:
        env.close()
        if was_training:
            model.train()
    n = max(1, len(returns))
    metrics = {
        "success_rate": successes / n,
        "mean_return": sum(returns) / n,
        "mean_length": sum(lengths) / n,
        "num_episodes": len(returns),
        "action_counts": total_counts,
        "env_name": env_name,
    }
    if verbose:
        print(f"  [{env_name}] action distribution: {total_counts}")
    return metrics


@torch.no_grad()
def evaluate_policy_multi(
    model, tokenizer, image_processor, device,
    env_specs,
    prompt=DEFAULT_PROMPT, verbose=True,
):
    """env_specs: список dict {name, max_steps, num_episodes, seed_start}.
    Возвращает dict env_name -> metrics."""
    out = {}
    for spec in env_specs:
        m = evaluate_policy(
            model, tokenizer, image_processor, device,
            env_name=spec["name"],
            max_steps=spec.get("max_steps", 50),
            num_episodes=spec.get("num_episodes", 20),
            seed_start=spec.get("seed_start", 10_000),
            prompt=prompt,
            verbose=verbose,
        )
        out[spec["name"]] = m
    return out

@torch.no_grad()
def evaluate_policy_text(
    model, tokenizer, image_processor, device,
    env_name, max_steps, num_episodes, seed_start,
    prompt, max_new_tokens, temperature=1.0,
):
    from src.rollout_text import sample_episode_text
    from src.env import MiniGridWrapper, ACTION_NAMES
    env = MiniGridWrapper(env_name=env_name, max_steps=max_steps)
    successes = 0
    returns = []
    lengths = []
    parse_fails = 0
    gen_tokens_total = 0
    gen_steps = 0
    try:
        for i in range(num_episodes):
            ep = sample_episode_text(
                model, tokenizer, image_processor, env, device,
                prompt=prompt, max_steps=max_steps,
                max_new_tokens=max_new_tokens, temperature=temperature,
                seed=seed_start + i,
            )
            if ep["success"]:
                successes += 1
            returns.append(ep["total_reward"])
            lengths.append(ep["length"])
            parse_fails += ep["parse_fails"]
            gen_tokens_total += ep["mean_gen_tokens"] * ep["length"]
            gen_steps += ep["length"]
    finally:
        env.close()
    pf_rate = parse_fails / max(1, gen_steps)
    mean_gen = gen_tokens_total / max(1, gen_steps)
    print(f"  [{env_name}] parse_fail_rate={pf_rate:.2%}, mean_gen_tokens={mean_gen:.1f}")
    return {
        "success_rate": successes / num_episodes,
        "mean_return": sum(returns) / num_episodes,
        "mean_length": sum(lengths) / num_episodes,
        "parse_fail_rate": pf_rate,
        "mean_gen_tokens": mean_gen,
        "num_episodes": num_episodes,
        "env_name": env_name,
        "max_steps": max_steps,
    }

@torch.no_grad()
def evaluate_policy_text_multi(
    model, tokenizer, image_processor, device,
    env_specs, prompt, max_new_tokens, temperature=1.0,
):
    results = {}
    for spec in env_specs:
        m = evaluate_policy_text(
            model, tokenizer, image_processor, device,
            env_name=spec["name"],
            max_steps=spec["max_steps"],
            num_episodes=spec["num_episodes"],
            seed_start=spec["seed_start"],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        results[spec["name"]] = m
    return results