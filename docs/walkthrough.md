# The full walkthrough

This page connects the five lessons. Take them in order the first time. After that, jump around as much as you like.

## 0. See what your machine can handle

```bash
python lessons/00_check_setup.py
```

This prints your Python and PyTorch versions, whether CUDA is ready, and how much GPU memory is visible. It does not download a model.

No GPU right now? You can still read the project, build your JSONL files, and run the data check. Baseline inference can run on CPU in theory, but a VLM will be painfully slow. The training lesson stops early unless CUDA is available.

## 1. Get a baseline before changing anything

```bash
python lessons/01_run_baseline.py
```

The first run downloads `Qwen/Qwen2-VL-2B-Instruct`, so it takes longer. The script opens one image, turns the image and question into the model's chat format, and prints only the new answer tokens.

Try another image or prompt like this:

```bash
python lessons/01_run_baseline.py \
  --image /path/to/item.jpg \
  --prompt "List the visible parts and their condition."
```

Save a few baseline answers somewhere. They are your reality check later. Without them, it is easy to celebrate a tuned model that did not actually improve.

## 2. Check the data before spending GPU time

```bash
python lessons/02_check_data.py
```

This checks three simple things:

- every JSON object has `image`, `question`, and `answer`
- the text fields are not empty
- each named image really exists

Point it at your own files when you are ready:

```bash
python lessons/02_check_data.py \
  --data data/processed/train.jsonl \
  --images data/raw/images
```

The check cannot tell whether an answer is accurate or well written. That part still needs a careful human pass. [The data guide](your-data.md) has a practical checklist.

## 3. Train a LoRA adapter

```bash
python lessons/03_train_lora.py
```

Here is what happens under the hood:

1. The base model loads in 4-bit form to save GPU memory.
2. LoRA adds a small set of trainable weights around the attention projections.
3. The image, question, and answer are packed into the model's chat format.
4. The question and image tokens are masked from the loss. The model learns from the answer.
5. Only the adapter and processor files are saved to `outputs/fashion-lora/adapter/`.

For your own run:

```bash
python lessons/03_train_lora.py \
  --data data/processed/train.jsonl \
  --images data/raw/images \
  --output outputs/my-domain-lora \
  --epochs 3 \
  --learning-rate 0.0002
```

Start with the defaults. If the answers become repetitive or forget useful details, lower the epochs or learning rate. If they barely move toward your desired style, inspect the labels first, then consider a longer run.

### What the main settings mean

- `epochs` is how many passes the trainer makes over the dataset.
- `learning-rate` controls how hard each update nudges the adapter.
- `batch-size` is how many examples fit on the GPU at once.
- `gradient-accumulation-steps` collects several small batches before an update.
- `max-length` caps the combined image, question, and answer token sequence.

Images also become tokens. The project keeps their resolution within a sensible range so a huge photo cannot take the whole sequence and push out the answer.

The last three live in `TrainSettings` in `src/vlm_finetune/training.py`. They stay out of the lesson command so the first run is not a wall of switches.

## 4. Compare on images training never saw

```bash
python lessons/04_compare_models.py
```

The script runs the base model first, frees it, then loads the same base with your adapter. Results land in `outputs/comparison.json` with the reference, base answer, and tuned answer together.

Read the answers. For a serious project, score each one against a short rubric such as:

- Is it factually correct?
- Does it use the domain vocabulary correctly?
- Does it follow the requested format?
- Does it invent details that are not visible?
- Is it more useful than the base answer?

Keep that rubric stable across experiments. A consistent human review beats a fancy metric that does not match the real job.

## Where to go next

Once this little loop works, the best next upgrades are usually more checked data, a larger held-out set, and a clear evaluation rubric. Add tracking dashboards and bigger config systems only when they solve a problem you actually have.
