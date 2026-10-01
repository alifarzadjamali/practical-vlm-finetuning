import argparse
import gc
import json
from pathlib import Path

import torch

from vlm_finetune.data import find_data_problems, read_jsonl
from vlm_finetune.inference import ask_about_image
from vlm_finetune.modeling import load_model, load_processor
from vlm_finetune.settings import (
    DEFAULT_MODEL_ID,
    OUTPUT_DIR,
    SAMPLE_EVAL_FILE,
    SAMPLE_IMAGE_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and tuned answers.")
    parser.add_argument("--data", type=Path, default=SAMPLE_EVAL_FILE)
    parser.add_argument("--images", type=Path, default=SAMPLE_IMAGE_DIR)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=OUTPUT_DIR / "fashion-lora" / "adapter",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "comparison.json")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


def collect_answers(model, processor, examples) -> list[str]:
    return [
        ask_about_image(model, processor, example.image, example.question)
        for example in examples
    ]


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.data, args.images)
    problems = find_data_problems(examples)
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"Fix these evaluation data problems first:\n{details}")
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"Adapter folder not found: {args.adapter}")

    processor = load_processor(args.model)

    print("Running the base model...")
    base_model = load_model(args.model)
    base_answers = collect_answers(base_model, processor, examples)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Running the tuned model...")
    tuned_model = load_model(args.model, args.adapter)
    tuned_answers = collect_answers(tuned_model, processor, examples)

    comparison = [
        {
            "image": example.image.name,
            "question": example.question,
            "reference": example.answer,
            "base_answer": base_answer,
            "tuned_answer": tuned_answer,
        }
        for example, base_answer, tuned_answer in zip(
            examples, base_answers, tuned_answers, strict=True
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved the side-by-side answers to {args.output}")


if __name__ == "__main__":
    main()
