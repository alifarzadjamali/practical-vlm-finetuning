# Bring your own domain data

The model learns the pattern in your examples, including the messy parts. A clean dataset is the highest-value work in this whole project.

## Keep the format boring

Use one JSON object per line:

```json
{"image":"pump_0042.jpg","question":"What condition is this pump in?","answer":"The casing has light surface rust near the lower flange. No crack or active leak is visible."}
```

The image name is resolved inside the folder passed with `--images`. Keeping paths out of the records makes the dataset easy to move between machines.

JSONL also makes large datasets friendlier to review and edit. One broken row does not hide inside a huge nested document.

## Write answers like the model you want

Pick a style and stay with it. If some labels are terse tags, others are essays, and others contain guesses, the model gets mixed instructions.

A useful answer should be:

- true to what is visible
- specific about domain details that matter
- consistent in tone and structure
- free from private or sensitive information
- honest when the image does not support a conclusion

Do not pad answers just to make them longer. Clear labels win.

## Split by the thing that could leak

Random rows are not always a safe split. If several photos show the same product, patient, property, or scene, keep that whole group in one split. Otherwise the evaluation set may be a near copy of training.

A decent starting split is 80 percent training, 10 percent validation, and 10 percent test. This small repo uses train and eval only to keep the lesson focused. For a real experiment, use validation while choosing settings and touch the test set at the end.

## How much data?

There is no magic count, but these ranges are useful for planning:

- 20 to 50 examples: check that the pipeline and format work
- 50 to 200 examples: test whether the task responds to fine-tuning
- hundreds to thousands: build something you can evaluate seriously

Coverage matters as much as count. Include common cases, difficult cases, different lighting and angles, and examples where the correct answer is uncertain.

## A quick review routine

Before training, sample records from the start, middle, and end of the file. Open each image beside its answer. Check names, units, spelling, omissions, and claims that cannot be seen.

Then run:

```bash
python lessons/02_check_data.py \
  --data data/processed/train.jsonl \
  --images data/raw/images
```

Keep the raw source untouched. Put cleaned JSONL files in `data/processed/`, which git ignores by default.
