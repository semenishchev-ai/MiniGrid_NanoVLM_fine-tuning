import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.config import merge_configs
from src.utils import set_seed, ensure_dir
from src.dataset import MiniGridActionDataset, make_collator
from src.model import load_vlm
from src.sft_trainer import train_sft
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


def make_eval_fn(tokenizer, image_processor, cfg, device):
    def _fn(model):
        metrics = evaluate_policy(
            model, tokenizer, image_processor, device,
            env_name=cfg["env"]["name"],
            max_steps=cfg["env"]["max_steps"],
            num_episodes=cfg["sft"].get("eval_episodes", 20),
            seed_start=cfg["sft"].get("eval_seed_start", 10_000),
            prompt=cfg["model"]["prompt"],
        )
        print(f"  eval: success_rate={metrics['success_rate']:.3f} "
              f"return={metrics['mean_return']:.3f} "
              f"length={metrics['mean_length']:.2f}")
        return metrics
    return _fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--config", default="configs/sft.yaml")
    p.add_argument("--skip-sanity", action="store_true", help="пропустить проверку коллатора")
    args = p.parse_args()

    cfg = merge_configs(args.base, args.config)
    set_seed(cfg["seed"])

    model, tokenizer, image_processor, device = load_vlm(
        hf_repo=cfg["model"]["hf_repo"],
        device=cfg["device"],
    )

    dataset = MiniGridActionDataset(
        data_dir=cfg["data"]["dir"],
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    collator = make_collator(
        tokenizer,
        max_length=cfg["sft"]["max_length"],
    )

    ckpt_dir = ensure_dir(cfg["sft"]["output_dir"])
    results_dir = ensure_dir(cfg["results"]["dir"])

    eval_fn = make_eval_fn(tokenizer, image_processor, cfg, device)

    history = train_sft(model, dataset, collator, eval_fn, cfg, device, ckpt_dir)

    with open(Path(results_dir) / "sft_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()