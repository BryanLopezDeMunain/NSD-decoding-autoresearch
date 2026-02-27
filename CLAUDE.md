# NSD decoding

The goal of this small project is to train models for visual category decoding from fMRI data using the natural scenes dataset.

I've prepared a huggingface format dataset in `datasets/nsd_flat_cococlip`. Each sample has the following fields:

- `subject_id`: NSD subject ID (0, ..., 7)
- `trial_id`: NSD trial ID
- `nsd_id`: NSD stimulus ID
- `activity`: visual cortex fMRI activity represented in a flat map format (shape `(1, 215, 200)`)
- `target`: target category ID (0, ..., 23)

A model should predict the target category given the activity map, very simple.

I created a dummy starter script `src/nsd_decoding/nsd_flat_cococlip_decoding_v0.py`. I would like you to iterate as much as you like and see what is the best test accuracy you can achieve, subject to some constraints:

- single H100 GPU only
- wall time at most 1hr
- no additional data

I would like you to track progress carefully, and keep a markdown report of results in `RESULTS.md`. Good luck, see how well you can score!
