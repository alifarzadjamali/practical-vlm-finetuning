import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Example:
    image: Path
    question: str
    answer: str


def read_jsonl(dataset_path: Path, image_dir: Path) -> list[Example]:
    """Read our three-field format and resolve each image path."""
    examples = []

    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{dataset_path}:{line_number} is not valid JSON: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{dataset_path}:{line_number} must contain a JSON object"
                )

            missing = {"image", "question", "answer"} - record.keys()
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(
                    f"{dataset_path}:{line_number} is missing: {fields}"
                )

            wrong_types = [
                field
                for field in ("image", "question", "answer")
                if not isinstance(record[field], str)
            ]
            if wrong_types:
                fields = ", ".join(wrong_types)
                raise ValueError(
                    f"{dataset_path}:{line_number} must use text for: {fields}"
                )

            examples.append(
                Example(
                    image=image_dir / record["image"],
                    question=str(record["question"]).strip(),
                    answer=str(record["answer"]).strip(),
                )
            )

    if not examples:
        raise ValueError(f"{dataset_path} has no examples")

    return examples


def find_data_problems(examples: Iterable[Example]) -> list[str]:
    """Return readable problems instead of failing on the first one."""
    problems = []

    for index, example in enumerate(examples, start=1):
        if not example.image.is_file():
            problems.append(f"Row {index}: image not found at {example.image}")
        if not example.question:
            problems.append(f"Row {index}: question is empty")
        if not example.answer:
            problems.append(f"Row {index}: answer is empty")

    return problems


def open_rgb_image(path: Path) -> Image.Image:
    """Open an image and give the model a consistent RGB copy."""
    with Image.open(path) as image:
        return image.convert("RGB")


class ImageTextDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]
