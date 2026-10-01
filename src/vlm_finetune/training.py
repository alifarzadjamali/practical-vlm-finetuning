from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from vlm_finetune.data import Example, ImageTextDataset, open_rgb_image
from vlm_finetune.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    MAX_IMAGE_PIXELS,
    MIN_IMAGE_PIXELS,
)


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    max_length: int = DEFAULT_MAX_LENGTH


def _conversation(example: Example) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": open_rgb_image(example.image)},
                {"type": "text", "text": example.question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example.answer}],
        },
    ]


class VlmDataCollator:
    """Build a batch and calculate loss only on each reference answer."""

    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[Example]) -> dict[str, torch.Tensor]:
        full_conversations = [_conversation(example) for example in examples]
        prompt_conversations = [conversation[:1] for conversation in full_conversations]
        images = [
            conversation[0]["content"][0]["image"]
            for conversation in full_conversations
        ]

        full_texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation in full_conversations
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in prompt_conversations
        ]

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for row, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100

        # Image placeholders are inputs, not words we want the model to predict.
        for token in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>"):
            token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch


def train_adapter(
    model_id: str,
    examples: list[Example],
    output_dir: Path,
    settings: TrainSettings,
) -> None:
    """Run a single-GPU QLoRA fine-tune and save only the small adapter."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training needs a CUDA GPU. The data check works without one."
        )

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_IMAGE_PIXELS,
        max_pixels=MAX_IMAGE_PIXELS,
    )
    processor.tokenizer.padding_side = "right"
    dataset = ImageTextDataset(examples)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=settings.epochs,
            learning_rate=settings.learning_rate,
            per_device_train_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            bf16=compute_dtype == torch.bfloat16,
            fp16=compute_dtype == torch.float16,
            remove_unused_columns=False,
            report_to="none",
            seed=42,
        ),
        train_dataset=dataset,
        data_collator=VlmDataCollator(processor, settings.max_length),
    )
    model.print_trainable_parameters()
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(adapter_dir)
