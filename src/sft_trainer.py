import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _cosine_with_warmup(optimizer, warmup, total):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _extract_sr(metrics, primary_env=None):
    if "success_rate" in metrics:
        return metrics["success_rate"]
    if primary_env and primary_env in metrics:
        return metrics[primary_env].get("success_rate", 0.0)
    first = next(iter(metrics.values()))
    return first.get("success_rate", 0.0)


def train_sft(model, dataset, collator, eval_fn, cfg, device, ckpt_dir):
    c = cfg["sft"]
    use_balanced = c.get("balance_actions", False)
    if use_balanced and hasattr(dataset, "get_sample_weights"):
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(dataset), replacement=True,
        )
        shuffle = False
        print(f"[trainer] WeightedRandomSampler enabled "
              f"(action counts: {dataset.get_action_counts()})")
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=c["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=c["num_workers"],
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    steps_per_epoch = math.ceil(len(loader) / c["grad_accum_steps"])
    total_steps = steps_per_epoch * c["epochs"]
    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=c["lr"],
        weight_decay=c["weight_decay"],
    )
    sched = _cosine_with_warmup(optim, c["warmup_steps"], total_steps)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    primary_env = c.get("primary_eval_env")
    best_sr = -1.0
    patience = 0
    history = []
    global_step = 0
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        running = 0.0
        n = 0
        optim.zero_grad()
        for i, batch in enumerate(loader):
            batch = _to_device(batch, device)
            logits, loss = model(
                batch["input_ids"],
                batch["images"],
                attention_mask=batch["attention_mask"],
                targets=batch["targets"],
            )
            (loss / c["grad_accum_steps"]).backward()
            running += loss.item()
            n += 1
            if (i + 1) % c["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["max_grad_norm"])
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % c["log_every"] == 0:
                    print(f"epoch {epoch} step {global_step} "
                          f"lr {sched.get_last_lr()[0]:.2e} loss {running / n:.6e}")
        train_loss = running / max(1, n)
        record = {"epoch": epoch, "train_loss": train_loss, "global_step": global_step}
        if c["eval_every_epoch"] and eval_fn is not None:
            model.eval()
            with torch.no_grad():
                metrics = eval_fn(model)
            if "success_rate" in metrics:
                record.update(metrics)
            else:
                record["eval"] = metrics
            sr = _extract_sr(metrics, primary_env)
            print(f"epoch {epoch} train_loss {train_loss:.6e} success_rate {sr:.3f}")
            if c["save_best"] and sr > best_sr:
                best_sr = sr
                patience = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "success_rate": sr},
                    ckpt_dir / c["ckpt_name"],
                )
            else:
                patience += 1
                if patience >= c.get("early_stop_patience", 999):
                    print(f"early stop at epoch {epoch}")
                    history.append(record)
                    break
        else:
            print(f"epoch {epoch} train_loss {train_loss:.6e}")
        history.append(record)
    return history