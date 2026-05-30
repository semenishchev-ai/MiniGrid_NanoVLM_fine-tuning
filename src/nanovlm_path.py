import sys
from pathlib import Path


def setup_nanovlm_import():
    root = Path(__file__).resolve().parents[1] / "external" / "nanoVLM"
    if not root.is_dir():
        raise FileNotFoundError(
            "nanoVLM not found at external/nanoVLM. "
            "Run: git clone --branch v0.1 --depth 1 "
            "https://github.com/huggingface/nanoVLM.git external/nanoVLM"
        )
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return root
