# JigNet: learning-based jigsaw puzzle solver

A Siamese CNN (**JigNet**) predicts the spatial relation of two 128x128 tiles
(`NO_MATCH / LEFT / RIGHT / UP / DOWN`). All pairwise probabilities of a shuffled puzzle are fed to an
**ILP** (PuLP) that finds the globally best g x g arrangement. Trained on 646 SDXL-Turbo images (4x4 grid) and
evaluated on held-out synthetic images and on a 250-image COCO subset, for several grid sizes.

Live demo: Hugging Face Space (`app.py`).

## Layout

```
configs/      one YAML per experiment (+ _common.yaml shared defaults)
data/         generate.py (SDXL), prepare.py (split/tile/pairs/checks), dataset.py, tiling.py, scenes.txt
models/       jignet.py, checkpoint.py, inference.py
solver/       scores.py (probs -> edge scores), ilp.py (PuLP model), board.py (metrics)
utils/        config, seeding, run folders, metrics, constants
scripts/      prepare_coco.py, compute_stats.py
train.py      python train.py --config configs/train.yaml
eval.py       python eval.py  --config configs/eval_*.yaml     (mode: pairs | network | pipeline)
app.py        Gradio demo
notebooks/    demo.ipynb (demo only)
tests/        pytest sanity tests
```

## Install

```bash
pip install -r requirements.txt          # core
pip install -r requirements-gen.txt      # only to regenerate the SDXL images
```

## Reproduce

Every step is one command; add `--set key=value` to override anything in a config
(e.g. `--set data.root=/content/drive/MyDrive/Jigsaw_Puzzle_Project/jigsaw_dataset`).

| Step | Command | Output |
|---|---|---|
| 1. generate images (GPU) | `python -m data.generate --config configs/generate_dataset.yaml` | `dataset/raw/...` + `generation_manifest.csv` |
| 2. split, tile, build pairs, run integrity checks | `python -m data.prepare --config configs/prepare_dataset.yaml` | `dataset/jigsaw_dataset/{images,tiles,metadata}` |
| 3. COCO subset | `python scripts/prepare_coco.py --src <coco/val2017> --dst dataset/COCO/resized_images` | 250 x 512x512 images + file list |
| 4. train | `python train.py --config configs/train.yaml` | `runs/jignet_exp2/` (checkpoints, `last.pth`, curves, metrics.csv) |
| 5. pair-level test report | `python eval.py --config configs/eval_pairs_test.yaml` | confusion matrix, report |
| 6. network, grids 4/5/6 | `python eval.py --config configs/eval_network_test.yaml` (and `_coco`) | `runs/eval/<name>/grid_*/` |
| 7. full pipeline, grids 3/4/5/6 | `python eval.py --config configs/eval_pipeline_test.yaml` (and `_coco`) | per-image results, CIs, `summary.csv` |
| 8. demo | `python app.py` | Gradio app |

`make prepare train eval-all test` does the same through the Makefile. Each run folder stores the config
actually used, `environment.json` (library versions, GPU, git commit) and all metrics, so every number in a
paper can be traced to a command and a commit.

Colab: mount Drive, `pip install -r requirements.txt`, and pass `--set data.root=...` (and
`checkpoint=...` for eval) pointing to your Drive folders.

## Using your existing data / model

* **Weights**: `eval.py` and `app.py` load the old `jignet_weights.pth` (bare state dict) unchanged:
  `python eval.py --config configs/eval_pipeline_test.yaml --set checkpoint=path/to/jignet_weights.pth`.
* **Dataset**: to keep the exact split and pairs the model was trained on, do **not** rebuild. Point
  `data.root` at the existing folder and, because the old `pairs.csv` stores absolute Colab paths, set
  `data.legacy_prefix=/content/drive/MyDrive/Jigsaw_Puzzle_Project/jigsaw_dataset`.
  A fresh `data.prepare` run cannot recreate the old split/negatives bit-for-bit (the notebook did not seed
  the negative sampling and split `os.listdir` order); to reuse the old split set
  `split.existing_csv` to the old `images_splits.csv`.

## Evaluation protocol (what the numbers mean)

* **pairs** - accuracy / macro-F1 on the pair manifest (`NO_MATCH` is 2/3 of the pairs, so look at balanced accuracy and macro-F1).
* **network** - every ordered tile pair of each full image is classified; reports top-1/top-3, balanced accuracy,
  macro-F1, relationship accuracy (true adjacent pairs recovered) and false-adjacent rate, with image-level bootstrap 95% CIs.
  Tiling `resize_image`: the image is resized to (g*128)^2 and cut.
* **pipeline** - tiles are shuffled (seeded), scored, solved by the ILP and compared with the original: perfect-reconstruction
  rate, absolute tile accuracy, neighbour accuracy, optimality proven, runtime. `model_errors` counts failures where the ILP
  found a board at least as good as the ground truth (the *scores* are wrong); `search_errors` counts boards worse than the
  ground truth (the *solver* stopped early). Tiling `resize_tiles`: (h//g)x(w//g) tiles are resized to 128.
