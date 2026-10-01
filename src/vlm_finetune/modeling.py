from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from vlm_finetune.settings import MAX_IMAGE_PIXELS, MIN_IMAGE_PIXELS


def best_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(model_id: str, adapter_path: Path | None = None):
    """Load the base model, with an optional LoRA adapter on top."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=best_dtype(),
        device_map="auto",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def load_processor(model_id: str):
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_IMAGE_PIXELS,
        max_pixels=MAX_IMAGE_PIXELS,
    )
    processor.tokenizer.padding_side = "right"
    return processor
