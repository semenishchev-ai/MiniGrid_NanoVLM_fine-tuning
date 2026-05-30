import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.env import ACTION_NAMES
from src.nanovlm_path import setup_nanovlm_import

DEFAULT_PROMPT = "What action should the agent take? Choose: left, right, forward."


class MiniGridActionDataset(Dataset):
    def __init__(self, data_dir, prompt=None, image_processor=None, tokenizer=None):
        self.data_dir = Path(data_dir)
        self.episodes_dir = self.data_dir / "episodes"
        if not self.episodes_dir.is_dir():
            raise FileNotFoundError(f"Episodes not found: {self.episodes_dir}")
        self.prompt = prompt or DEFAULT_PROMPT
        self.samples = self._build_index()
        if image_processor is None or tokenizer is None:
            setup_nanovlm_import()
            from data.processors import get_image_processor, get_tokenizer
            from models.config import VLMConfig
            cfg = VLMConfig()
            if tokenizer is None:
                tokenizer = get_tokenizer(cfg.lm_tokenizer)
            if image_processor is None:
                image_processor = get_image_processor(cfg.vit_img_size)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def _build_index(self):
        samples = []
        for episode_dir in sorted(self.episodes_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            meta_path = episode_dir / "actions.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            for step in range(meta["num_steps"]):
                samples.append((episode_dir, step))
        if not samples:
            raise ValueError(f"No samples in {self.episodes_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        episode_dir, step = self.samples[idx]
        meta = json.loads((episode_dir / "actions.json").read_text())
        image = Image.open(episode_dir / f"{step:04d}.png").convert("RGB")
        action = meta["action_names"][step]
        if action not in ACTION_NAMES:
            action = ACTION_NAMES[meta["actions"][step]]
        processed_image = self.image_processor(image)
        answer = action + self.tokenizer.eos_token
        text_data = f"Question: {self.prompt} Answer:"
        return {
            "image": processed_image,
            "text_data": text_data,
            "answer": answer,
        }


class ActionCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])
        texts = [item["text_data"] for item in batch]
        answers = [item["answer"] for item in batch]
        input_sequences = [f"{texts[i]}{answers[i]}" for i in range(len(batch))]

        encoded = self.tokenizer(
            input_sequences,
            padding="max_length",
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone()
        labels[:, -1] = -100

        original_lengths = [len(self.tokenizer.encode(seq)) for seq in input_sequences]
        for i in range(len(batch)):
            question_length = len(
                self.tokenizer.encode(texts[i], add_special_tokens=False)
            )
            if original_lengths[i] > self.max_length:
                labels[i, :] = -100
                continue
            first_token_pos = attention_mask[i].nonzero(as_tuple=True)[0][0].item()
            question_end = first_token_pos + question_length - 1
            labels[i, :question_end] = -100

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def make_collator(tokenizer, max_length):
    return ActionCollator(tokenizer, max_length)
