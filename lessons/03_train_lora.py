import argparse
from pathlib import Path

from vlm_finetune.data import find_data_problems, read_jsonl
from vlm_finetune.settings import (
    DEFAULT_MODEL_ID,
    OUTPUT_DIR,
    SAMPLE_IMAGE_DIR,
    SAMPLE_TRAIN_FILE,
)
from vlm_finetune.training import TrainSettings, train_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the VLM with QLoRA.")
    parser.add_argument("--data", type=Path, default=SAMPLE_TRAIN_FILE)
    parser.add_argument("--images", type=Path, default=SAMPLE_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "fashion-lora")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.data, args.images)
    problems = find_data_problems(examples)
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"Fix these data problems first:\n{details}")

    settings = TrainSettings(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print(f"Training on {len(examples)} example(s).")
    print(f"The adapter will land in {args.output / 'adapter'}")
    train_adapter(args.model, examples, args.output, settings)


if __name__ == "__main__":
    main()
