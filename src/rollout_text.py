import torch
import torch.nn.functional as F
from PIL import Image
from src.env import MiniGridWrapper, NAME_TO_ACTION, ACTION_NAMES


def parse_last_action(text: str):
    text_low = text.lower()
    last_pos = -1
    last_name = None
    for name in ACTION_NAMES:
        idx = text_low.rfind(name)
        if idx > last_pos:
            last_pos = idx
            last_name = name
    return last_name


def _forward_logits_grad(model, input_ids, image, attention_mask):
    image_embd = model.vision_encoder(image)
    image_embd = model.MP(image_embd)
    token_embd = model.decoder.token_embedding(input_ids)
    combined = torch.cat((image_embd, token_embd), dim=1)
    B = image_embd.size(0)
    img_seq = image_embd.size(1)
    if attention_mask is not None:
        img_mask = torch.ones(
            (B, img_seq),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((img_mask, attention_mask), dim=1)
    out = model.decoder(combined, attention_mask)
    if not model.decoder.lm_use_tokens:
        out = model.decoder.head(out)
    return out, img_seq


@torch.no_grad()
def _generate_step(
    model, tokenizer, prompt_input_ids, prompt_attention_mask, image,
    device, max_new_tokens, temperature, eos_token_id,
):
    input_ids = prompt_input_ids.clone()
    attention_mask = prompt_attention_mask.clone()
    gen_ids = []
    gen_log_probs = []
    for _ in range(max_new_tokens):
        out, _ = _forward_logits_grad(model, input_ids, image, attention_mask)
        last_logits = out[:, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)  # [1, 1]
        log_p = torch.log(probs.gather(1, sampled) + 1e-10)
        token_id = sampled.item()
        gen_ids.append(token_id)
        gen_log_probs.append(log_p.item())
        if token_id == eos_token_id:
            break
        input_ids = torch.cat([input_ids, sampled], dim=1)
        new_mask = torch.ones(
            (1, 1), device=attention_mask.device, dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)
    return (
        torch.tensor(gen_ids, dtype=torch.long),
        torch.tensor(gen_log_probs, dtype=torch.float32),
    )


@torch.no_grad()
def sample_episode_text(
    model, tokenizer, image_processor, env, device,
    prompt, max_steps, max_new_tokens, temperature, seed=None,
):
    obs, _ = env.reset(seed=seed)
    prompt_text = f"Question: {prompt} Answer:"
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_ids = enc["input_ids"]
    prompt_mask = enc["attention_mask"]
    eos_id = tokenizer.eos_token_id

    images = []
    generated_tokens_list = []
    old_log_probs_list = []
    gen_texts = []  # NEW
    total_reward = 0.0
    steps = 0
    success = False
    parse_fails = 0
    gen_lengths = []

    while steps < max_steps:
        image = Image.fromarray(obs).convert("RGB")
        px = image_processor(image).unsqueeze(0).to(device)
        gen_ids, gen_log_probs = _generate_step(
            model, tokenizer, prompt_ids, prompt_mask, px,
            device, max_new_tokens, temperature, eos_id,
        )
        gen_text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        action_name = parse_last_action(gen_text)
        if action_name is None:
            parse_fails += 1
            action_name = "forward"
        action = NAME_TO_ACTION[action_name]

        images.append(px.squeeze(0).cpu())
        generated_tokens_list.append(gen_ids)
        old_log_probs_list.append(gen_log_probs)
        gen_lengths.append(len(gen_ids))
        gen_texts.append((gen_text, action_name))  # NEW

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
        "generated_tokens": generated_tokens_list,
        "old_log_probs": old_log_probs_list,
        "gen_texts": gen_texts,  # NEW
        "total_reward": total_reward,
        "length": steps,
        "success": success,
        "parse_fails": parse_fails,
        "mean_gen_tokens": (
            sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0.0
        ),
        "prompt_input_ids": prompt_ids.squeeze(0).cpu(),
        "prompt_attention_mask": prompt_mask.squeeze(0).cpu(),
    }


def compute_log_probs_text(
    model, image, prompt_input_ids, prompt_attention_mask,
    generated_tokens, device, with_grad=True,
):
    prompt_ids = prompt_input_ids.unsqueeze(0).to(device)
    prompt_mask = prompt_attention_mask.unsqueeze(0).to(device)
    gen_ids = generated_tokens.unsqueeze(0).to(device)
    if gen_ids.size(1) == 0:
        return (
            torch.zeros(0, device=device),
            torch.zeros(0, device=device),
        )
    full_input = torch.cat([prompt_ids, gen_ids[:, :-1]], dim=1)
    full_mask = torch.cat(
        [prompt_mask, torch.ones_like(gen_ids[:, :-1])], dim=1,
    )
    image = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        out, img_seq = _forward_logits_grad(model, full_input, image, full_mask)
        P = prompt_ids.size(1)
        L = gen_ids.size(1)
        start = img_seq + P - 1
        pred_logits = out[:, start:start + L, :]
        log_probs_all = F.log_softmax(pred_logits, dim=-1)
        probs_all = log_probs_all.exp()
        target = gen_ids
        log_probs = log_probs_all.gather(2, target.unsqueeze(-1)).squeeze(-1).squeeze(0)
        entropy = -(probs_all * log_probs_all).sum(-1).squeeze(0)
    return log_probs, entropy