import json
import shutil
from pathlib import Path

from PIL import Image

from src.env import ACTION_NAMES, MiniGridWrapper
from src.expert import ExpertPolicy


def collect_episode(env, expert, episode_dir, seed=None):
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    actions = []
    step = 0
    while True:
        Image.fromarray(obs).save(episode_dir / f"{step:04d}.png")
        action = int(expert.act(env))
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break
    payload = {
        "actions": actions,
        "action_names": [ACTION_NAMES[a] for a in actions],
        "num_steps": len(actions),
        "seed": seed,
    }
    with open(episode_dir / "actions.json", "w") as f:
        json.dump(payload, f, indent=2)


def collect_dataset(
    output_dir,
    num_episodes,
    env_name="MiniGrid-Empty-Random-6x6-v0",
    max_steps=100,
    start_seed=0,
):
    output_dir = Path(output_dir)
    episodes_dir = output_dir / "episodes"
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    env = MiniGridWrapper(env_name=env_name, max_steps=max_steps)
    expert = ExpertPolicy()
    try:
        for i in range(num_episodes):
            seed = start_seed + i
            episode_dir = episodes_dir / f"{i:06d}"
            collect_episode(env, expert, episode_dir, seed=seed)
    finally:
        env.close()
    return episodes_dir
