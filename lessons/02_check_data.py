import argparse
from pathlib import Path

from vlm_finetune.data import find_data_problems, read_jsonl
from vlm_finetune.settings import SAMPLE_IMAGE_DIR, SAMPLE_TRAIN_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a VLM dataset before training.")
    parser.add_argument("--data", type=Path, default=SAMPLE_TRAIN_FILE)
    parser.add_argument("--images", type=Path, default=SAMPLE_IMAGE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.data, args.images)
    problems = find_data_problems(examples)

    if problems:
        print(f"Found {len(problems)} problem(s):")
        for problem in problems:
            print("-", problem)
        raise SystemExit(1)

    print(f"Nice, all {len(examples)} record(s) look ready.")
    if len(examples) < 50:
        print("This is a format demo. Add more examples before a real training run.")


if __name__ == "__main__":
    main()
