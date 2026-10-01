# Fine-tune a vision-language model, one small step at a time

Hey, welcome. This repo is a hands-on path for teaching a vision-language model (VLM) how to speak the language of your own domain.

The example here is fashion. The same flow works for product photos, medical equipment, plants, warehouse parts, or any other image collection where a general model needs more specific answers.

We use Qwen2-VL 2B and LoRA because they make a friendly starting point. The project stays intentionally small: five lessons, one reusable Python package, and plain JSONL data.

> The two included samples only show you the data format. They are nowhere near enough for a useful fine-tune. Bring a real dataset before judging the result.

## What you will learn

By the end, you will know how to:

- check whether your machine is ready
- run the base model and capture a baseline
- shape your own image and text pairs
- fine-tune with QLoRA without changing every model weight
- compare the base model with your adapter on held-out images

You do not need to understand every detail before starting. Each lesson is short, and the deeper notes are there when you want them.

## The path

| Step | Run | What it teaches |
| --- | --- | --- |
| 0 | `python lessons/00_check_setup.py` | Check Python, PyTorch, CUDA, and GPU memory |
| 1 | `python lessons/01_run_baseline.py` | Ask the untouched model about one image |
| 2 | `python lessons/02_check_data.py` | Catch missing images and messy records early |
| 3 | `python lessons/03_train_lora.py` | Train a small adapter on your domain |
| 4 | `python lessons/04_compare_models.py` | Save base and tuned answers side by side |

Start with [the walkthrough](docs/walkthrough.md) when you are ready to run things.

## Quick setup

Python 3.11 or newer is a good place to start. Create an environment, use the [PyTorch install picker](https://pytorch.org/get-started/locally/) to install the right build for your CUDA setup, then install this project:

```bash
python -m venv .venv
source .venv/bin/activate

# Run the PyTorch command you copied from the install picker first.
pip install -e .
```

The model is downloaded from Hugging Face the first time you use it. If the download asks for a token, run `huggingface-cli login` once.

## Your data in one glance

Every line in a dataset file is one image, one question, and the answer you want the model to learn:

```json
{"image":"sample_001.jpg","question":"Describe this outfit for a product page.","answer":"A short turquoise dress with a red geometric print, fitted waist, and flared skirt."}
```

Images live beside the sample data in `data/samples/images/`. Training and evaluation examples are kept separate so you do not test the model on pictures it already saw.

Read [bringing your own data](docs/your-data.md) before building a real dataset. Label quality matters much more than clever training settings.

## Repo map

```text
practical-vlm-finetuning/
├── lessons/                 # the five files to follow in order
├── src/vlm_finetune/        # reusable loading, data, training, and evaluation code
├── data/samples/            # two tiny examples that document the format
├── docs/                    # friendly explanations and practical advice
└── outputs/                 # created locally and ignored by git
```

## A few honest expectations

- A fine-tune changes behavior and vocabulary. It does not magically add clean knowledge that is missing from your labels.
- Keep a test set that training never touches.
- Start with 50 to 100 carefully checked examples to test the idea. Most real projects need hundreds or thousands.
- Compare against the base model. A lower training loss does not always mean better answers.
- The default 4-bit setup is meant to lower memory use, but you still need a CUDA GPU for training.

If something goes sideways, [the troubleshooting page](docs/troubleshooting.md) covers the common bumps without sending you down a rabbit hole.

## Make it yours

Fashion is just the demo. Change the prompt, replace the two JSONL files and images, then keep the rest of the flow. The code has no fashion-only training logic hiding inside it.

That is the whole idea: understand the path once, then reuse it for a domain you actually care about.
