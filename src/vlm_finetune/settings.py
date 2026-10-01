from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"
SAMPLE_IMAGE_DIR = SAMPLE_DIR / "images"
SAMPLE_TRAIN_FILE = SAMPLE_DIR / "train.jsonl"
SAMPLE_EVAL_FILE = SAMPLE_DIR / "eval.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_PROMPT = "Describe this outfit for a fashion product page."

# These are calm starter values, not magic values. Change one thing at a time.
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_MAX_LENGTH = 1024

# Qwen2-VL turns each 28 by 28 pixel patch into a visual token. These bounds keep
# normal photos useful without letting one huge image swallow the whole sequence.
MIN_IMAGE_PIXELS = 256 * 28 * 28
MAX_IMAGE_PIXELS = 768 * 28 * 28
