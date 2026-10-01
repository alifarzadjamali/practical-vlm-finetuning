from pathlib import Path

import torch

from vlm_finetune.data import open_rgb_image


def ask_about_image(
    model,
    processor,
    image_path: Path,
    question: str,
    max_new_tokens: int = 128,
) -> str:
    image = open_rgb_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    new_tokens = generated[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
