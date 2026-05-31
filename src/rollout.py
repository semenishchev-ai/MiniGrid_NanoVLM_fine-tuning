import torch
import torch.nn.functional as F
from PIL import Image

from src.env import MiniGridWrapper, NAME_TO_ACTION, ACTION_NAMES

def get_action_token_ids(tokenizer):
    ids = {}
    for name in ACTION_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(
                f"Action '{name}' tokenizes to {len(toks)} tokens, expected 1"
            )
        ids[name] = toks[0]
    return ids


@torch.no_grad()
def _forward_action_logits(model, input_ids, image, attention_mask):
    image_embd = model.vision_encoder(image)
    image_embd = model.MP(image_embd)
    token_embd = model.decoder.token_embedding(input_ids)
    combined = torch.cat((image_embd, token_embd), dim=1)
    B = image_embd.size(0)
    img_seq = image_embd.size(1)
    if attention_mask is not None:
        img_mask = torch.ones((B, img_seq), device=attention_mask.device,
                              dtype=attention_mask.dtype)
        attention_mask = torch.cat((img_mask, attention_mask), dim=1)
    out = model.decoder(combined, attention_mask)
    last_logits = out[:, -1, :]
    if not model.decoder.lm_use_tokens:
        last_logits = model.decoder.head(last_logits)
    return last_logits


def _forward_action_logits_grad(model, input_ids, image, attention_mask):
    image_embd = model.vision_encoder(image)
    image_embd = model.MP(image_embd)
    token_embd = model.decoder.token_embedding(input_ids)
    combined = torch.cat((image_embd, token_embd), dim=1)
    B = image_embd.size(0)
    img_seq = image_embd.size(1)
    if attention_mask is not None:
        img_mask = torch.ones((B, img_seq), device=attention_mask.device,
                              dtype=attention_mask.dtype)
        attention_mask = torch.cat((img_mask, attention_mask), dim=1)
    out = model.decoder(combined, attention_mask)
    last_logits = out[:, -1, :]
    if not model.decoder.lm_use_tokens:
        last_logits = model.decoder.head(last_logits)
    return last_logits


@torch.no_grad()
def sample_episode(
    model, tokenizer, image_processor, env, device,
    prompt, action_token_ids, max_steps, temperature=1.0, seed=None,
):
    obs, _ = env.reset(seed=seed)
    prompt_text = f"Question: {prompt} Answer:"
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    action_ids_list = sorted(action_token_ids.values())
    action_ids_tensor = torch.tensor(action_ids_list, device=device)
    id_to_name = {v: k for k, v in action_token_ids.items()}

    images = []
    action_tokens = []
    old_log_probs = []
    total_reward = 0.0
    steps = 0
    success = False

    while steps < max_steps:
        image = Image.fromarray(obs).convert("RGB")
        px = image_processor(image).unsqueeze(0).to(device)
        logits = _forward_action_logits(model, input_ids, px, attention_mask)
        action_logits = logits[0, action_ids_tensor] / temperature
        probs = F.softmax(action_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        action_token_id = action_ids_list[sampled_idx]
        log_prob = torch.log(probs[sampled_idx] + 1e-10).item()
        action_name = id_to_name[action_token_id]
        action = NAME_TO_ACTION[action_name]

        images.append(px.squeeze(0).cpu())
        action_tokens.append(action_token_id)
        old_log_probs.append(log_prob)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated:
            success = reward > 0.0
            break
        if truncated:
            break

    return {
        "images": torch.stack(images) if images else torch.empty(0),
        "action_tokens": torch.tensor(action_tokens, dtype=torch.long),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "total_reward": total_reward,
        "length": steps,
        "success": success,
        "prompt_input_ids": input_ids.squeeze(0).cpu(),
        "prompt_attention_mask": attention_mask.squeeze(0).cpu(),
    }


def compute_log_probs(
    model, images, prompt_input_ids, prompt_attention_mask,
    action_tokens, action_token_ids, device, with_grad=True,
):
    T = images.size(0)
    if T == 0:
        return torch.zeros(0, device=device)
    images = images.to(device)
    input_ids = prompt_input_ids.unsqueeze(0).expand(T, -1).to(device)
    attention_mask = prompt_attention_mask.unsqueeze(0).expand(T, -1).to(device)

    action_ids_list = sorted(action_token_ids.values())
    action_ids_tensor = torch.tensor(action_ids_list, device=device)
    id_to_idx = {tid: i for i, tid in enumerate(action_ids_list)}

    if with_grad:
        logits = _forward_action_logits_grad(model, input_ids, images, attention_mask)
    else:
        with torch.no_grad():
            logits = _forward_action_logits(model, input_ids, images, attention_mask)

    action_logits = logits[:, action_ids_tensor]
    log_probs_all = F.log_softmax(action_logits, dim=-1)
    action_indices = torch.tensor(
        [id_to_idx[t.item()] for t in action_tokens], device=device, dtype=torch.long,
    )
    log_probs = log_probs_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    return log_probs