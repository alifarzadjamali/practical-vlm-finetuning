# When something gets stuck

Here are the common problems, starting with the fixes that usually help.

## CUDA says false

Run `python lessons/00_check_setup.py`. If CUDA is false, the installed PyTorch build may not match your driver and CUDA setup. Use the install picker at [pytorch.org](https://pytorch.org/get-started/locally/) instead of guessing a CUDA wheel.

Data preparation still works without CUDA. Training does not.

## The GPU runs out of memory

Close other GPU jobs first. Then try these in order:

1. Reduce `DEFAULT_MAX_LENGTH` in `src/vlm_finetune/settings.py`.
2. Keep the batch size at 1 and increase gradient accumulation if needed.
3. Resize unusually large source images before training.
4. Pick a smaller VLM if your domain allows it.

The default training path already loads the model in 4-bit form and trains only LoRA weights.

## The model download fails

Check the internet connection and free disk space. If Hugging Face asks for access, run:

```bash
huggingface-cli login
```

The model cache can be much larger than the adapter saved by this repo.

## The data check reports a missing image

The `image` field is relative to the folder passed through `--images`. Watch for uppercase extensions and spelling differences, especially when moving data between Windows and Linux.

## Training loss falls, but answers get worse

That usually points to overfitting, inconsistent labels, or an evaluation set that is too small. Compare with the base model, inspect bad examples, and try fewer epochs. More training is not always better.

## The tuned answer looks exactly like the base answer

Make sure `lessons/04_compare_models.py` is pointing to the adapter created by your run. If it is, review whether the training examples clearly demonstrate a different vocabulary, format, or behavior. A tiny or vague dataset gives the adapter very little to learn.

## You changed models and the code broke

This repo is written for the Qwen2-VL model family. Another VLM may use a different model class, chat template, image processor, or LoRA target names. Treat a model swap as a small porting job, not just a new model ID.
