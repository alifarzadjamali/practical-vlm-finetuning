import argparse
from pathlib import Path

from vlm_finetune.inference import ask_about_image
from vlm_finetune.modeling import load_model, load_processor
from vlm_finetune.settings import DEFAULT_MODEL_ID, DEFAULT_PROMPT, SAMPLE_IMAGE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try the base VLM on one image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=SAMPLE_IMAGE_DIR / "sample_001.jpg",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = load_processor(args.model)
    model = load_model(args.model)
    answer = ask_about_image(model, processor, args.image, args.prompt)

    print("\nPrompt:")
    print(args.prompt)
    print("\nBase model answer:")
    print(answer)


if __name__ == "__main__":
    main()
