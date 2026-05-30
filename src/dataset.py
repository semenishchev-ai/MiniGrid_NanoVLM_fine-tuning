import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.env import ACTION_NAMES
from src.nanovlm_path import setup_nanovlm_import

DEFAULT_PROMPT = "What action should the agent take? Choose: left, right, forward."


def _build_pil_augment():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=192,
            scale=(0.6, 1.0),
            ratio=(1.0, 1.0),
            antialias=True,
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
    ])


class MiniGridActionDataset(Dataset):
    def __init__(
        self,
        data_dir,
        prompt=None,
        image_processor=None,
        tokenizer=None,
        augment=False,
        random_erasing=False,
    ):
        self.data_dir = Path(data_dir)
        self.episodes_dir = self.data_dir / "episodes"
        if not self.episodes_dir.is_dir():
            raise FileNotFoundError(f"Episodes not found: {self.episodes_dir}")
        self.prompt = prompt or DEFAULT_PROMPT
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
        self.samples = self._build_index()
        self.pil_augment = _build_pil_augment() if augment else None
        self.random_erasing = (
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), value=0.0)
            if random_erasing else None
        )

    def _build_index(self):
        samples = []
        for episode_dir in sorted(self.episodes_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            meta_path = episode_dir / "actions.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            actions = meta.get("action_names") or [ACTION_NAMES[a] for a in meta["actions"]]
            for step, action in enumerate(actions):
                samples.append((episode_dir, step, action))
        if not samples:
            raise ValueError(f"No samples in {self.episodes_dir}")
        return samples

    def get_action_counts(self):
        return dict(Counter(action for _, _, action in self.samples))

    def get_sample_weights(self):
        counts = self.get_action_counts()
        n_actions = len(counts)
        weights = [1.0 / (n_actions * counts[a]) for _, _, a in self.samples]
        return torch.tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        episode_dir, step, action = self.samples[idx]
        image = Image.open(episode_dir / f"{step:04d}.png").convert("RGB")
        if self.pil_augment is not None:
            image = self.pil_augment(image)
        processed_image = self.image_processor(image)
        if self.random_erasing is not None:
            processed_image = self.random_erasing(processed_image)
        text_data = f"Question: {self.prompt} Answer:"
        answer = " " + action + self.tokenizer.eos_token
        return {
            "image": processed_image,
            "text_data": text_data,
            "answer": answer,
        }


class ActionCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])
        prompt_ids_list = []
        answer_ids_list = []
        for item in batch:
            p_ids = self.tokenizer.encode(item["text_data"], add_special_tokens=False)
            a_ids = self.tokenizer.encode(item["answer"], add_special_tokens=False)
            prompt_ids_list.append(p_ids)
            answer_ids_list.append(a_ids)

        seqs, labels_seqs = [], []
        for p, a in zip(prompt_ids_list, answer_ids_list):
            full = p + a
            label = [-100] * len(p) + list(a)
            if len(full) > self.max_length:
                cut = len(full) - self.max_length
                full = full[cut:]
                label = label[cut:]
            seqs.append(full)
            labels_seqs.append(label)

        batch_max = max(len(s) for s in seqs)
        B = len(batch)
        input_ids = torch.full((B, batch_max), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, batch_max), dtype=torch.long)
        labels = torch.full((B, batch_max), -100, dtype=torch.long)

        for i, (s, lab) in enumerate(zip(seqs, labels_seqs)):
            L = len(s)
            input_ids[i, :L] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, :L] = 1
            labels[i, :L] = torch.tensor(lab, dtype=torch.long)

        shifted = labels.clone()
        shifted[:, :-1] = labels[:, 1:]
        shifted[:, -1] = -100
        labels = shifted

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": labels,
        }


def make_collator(tokenizer, max_length, shift_labels=False):
    return ActionCollator(tokenizer, max_length)