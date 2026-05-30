from src.nanovlm_path import setup_nanovlm_import
from src.utils import get_device


def load_vlm(hf_repo="lusxvr/nanoVLM-222M", checkpoint_path=None, device="auto"):
    setup_nanovlm_import()
    from data.processors import get_image_processor, get_tokenizer
    from models.vision_language_model import VisionLanguageModel

    source = checkpoint_path or hf_repo
    model = VisionLanguageModel.from_pretrained(source)
    dev = get_device(device)
    model = model.to(dev)
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)
    return model, tokenizer, image_processor, dev


def decode_action(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip().lower()
    for name in ("left", "right", "forward"):
        if name in text:
            return name
    return text.split()[0] if text else ""
