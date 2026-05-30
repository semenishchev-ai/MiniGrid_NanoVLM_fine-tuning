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
from src.evaluate import evaluate_policy

from src.nanovlm_path import setup_nanovlm_import
setup_nanovlm_import()
from models.vision_language_model import VisionLanguageModel


@torch.no_grad()
def greedy_generate(self, input_ids, image, attention_mask=None, max_new_tokens=5):
    image_embd = self.vision_encoder(image)
    image_embd = self.MP(image_embd)
    token_embd = self.decoder.token_embedding(input_ids)
    combined_embd = torch.cat((image_embd, token_embd), dim=1)
    batch_size = image_embd.size(0)
    img_seq_len = image_embd.size(1)
    if attention_mask is not None:
        image_attention_mask = torch.ones(
            (batch_size, img_seq_len),
            device=attention_mask.device, dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
    outputs = combined_embd
    generated_tokens = torch.zeros(
        (batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype,
    )
    for i in range(max_new_tokens):
        model_out = self.decoder(outputs, attention_mask)
        last_token_logits = model_out[:, -1, :]
        if not self.decoder.lm_use_tokens:
            last_token_logits = self.decoder.head(last_token_logits)
        next_token = last_token_logits.argmax(dim=-1, keepdim=True)
        generated_tokens[:, i] = next_token.squeeze(-1)
        next_embd = self.decoder.token_embedding(next_token)
        outputs = torch.cat((outputs, next_embd), dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                (attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)),
                dim=1,
            )
    return generated_tokens


VisionLanguageModel.generate = greedy_generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--config", default="configs/sft.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed-start", type=int, default=None)
    p.add_argument("--env", default=None, help="Переопределить env.name из конфига")
    p.add_argument("--max-steps", type=int, default=None, help="Переопределить env.max_steps")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = merge_configs(args.base, args.config)
    set_seed(cfg["seed"])

    model, tokenizer, image_processor, device = load_vlm(
        hf_repo=cfg["model"]["hf_repo"],
        device=cfg["device"],
    )
    ckpt_path = args.checkpoint or str(Path(cfg["sft"]["output_dir"]) / cfg["sft"]["ckpt_name"])
    if Path(ckpt_path).is_file():
        print(f"loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    else:
        print(f"checkpoint not found: {ckpt_path} — оцениваем базовую модель")

    env_name = args.env or cfg["env"]["name"]
    max_steps = args.max_steps if args.max_steps is not None else cfg["env"]["max_steps"]

    metrics = evaluate_policy(
        model, tokenizer, image_processor, device,
        env_name=env_name,
        max_steps=max_steps,
        num_episodes=args.episodes or cfg["sft"].get("eval_episodes", 20),
        seed_start=args.seed_start if args.seed_start is not None
                   else cfg["sft"].get("eval_seed_start", 10_000),
        prompt=cfg["model"]["prompt"],
    )
    metrics["env_name"] = env_name
    metrics["max_steps"] = max_steps
    print(json.dumps(metrics, indent=2))

    if args.out:
        out_path = args.out
    else:
        # уникальное имя по env, чтобы не перезаписывать sft_eval.json
        safe = env_name.replace("/", "_")
        out_path = str(Path(ensure_dir(cfg["results"]["dir"])) / f"sft_eval_{safe}.json")
    Path(out_path).write_text(json.dumps(metrics, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()