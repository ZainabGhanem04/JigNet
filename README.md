# 🧩 Deep Learning Approach to Solving Jigsaw Puzzles

A learning-based approach for reconstructing a jigsaw puzzle from shuffled image tiles.

The project explores whether a neural network can learn **spatial relationships between image tiles** and use those learned relationships to reconstruct an entire image. The system combines a **Siamese convolutional neural network** for local tile-pair prediction with **Integer Linear Programming (ILP)** for globally consistent puzzle reconstruction.

* **Project type:** Learning Project
* **Task:** Jigsaw puzzle reconstruction
* **Input:** Shuffled image tiles
* **Output:** Reconstructed image
* **Deep Learning:** Siamese CNN
* **Solver:** Integer Linear Programming (ILP)
* **Dataset:** AI-generated images using SDXL Turbo
* **Framework:** PyTorch

---

### Try the demo

👉 **[Hugging Face Space](https://huggingface.co/spaces/Zainab04/JigNet)**

The interface is intended primarily as a demonstration of the trained model and its inference behavior.



## 📌 Overview

Solving a jigsaw puzzle requires more than determining whether two pieces look similar. A solver needs to understand **which pieces belong next to each other and in which direction**.

This project approaches the problem as a combination of:

1. **Local visual learning** — a neural network learns the spatial relationship between two tiles.
2. **Global optimization** — an Integer Linear Programming solver uses the network's predictions to find a globally consistent arrangement of all tiles.

The complete pipeline is:

```text
Original Image
      │
      ▼
Split Image into 4 × 4 Tiles
      │
      ▼
Generate Positive & Negative Tile Pairs
      │
      ▼
Siamese CNN
      │
      ▼
Pairwise Spatial Predictions
      │
      ▼
Integer Linear Programming
      │
      ▼
Reconstructed Image
```

The project was developed primarily as a **learning exercise**: starting from dataset construction, designing and training a neural network from scratch, evaluating its behavior, and finally connecting its predictions to a combinatorial optimization algorithm.

---

# 🎯 Project Goal

The main goal is to investigate whether a relatively simple learning-based system can learn enough **local spatial information** to reconstruct a shuffled image.

More specifically, the neural network is trained to answer questions such as:

> Given two image tiles, how are they spatially related?

The network predicts the relationship between two tiles, allowing the solver to determine whether one tile belongs:

* above another tile,
* below it,
* to its left,
* to its right,
* or has no valid direct neighboring relationship.

The predictions are then passed to an ILP solver instead of simply choosing the highest-scoring neighbor independently.

---

# 🧠 System Architecture

The project consists of two major components.

## 1. Siamese Neural Network

Each candidate pair of tiles is passed through the same feature extractor.

```text
Tile A ───────► Shared CNN ─────► Feature A ──┐
                                               │
                                               ├──► Concatenate ─► MLP ─► Classes
                                               │
Tile B ───────► Shared CNN ─────► Feature B ──┘
```

The two branches share weights, meaning that the same visual feature extractor processes both tiles.

The resulting feature vectors are concatenated and passed through fully connected layers to predict the spatial relationship between the two tiles.

### Why a Siamese architecture?

The task is fundamentally a **comparison problem**.

Instead of learning an absolute representation for a single tile, the network needs to determine how two tiles relate to one another.

Weight sharing also means that both tiles are represented in the same feature space.

---

## 2. Integer Linear Programming Solver

The neural network provides local pairwise predictions, but local decisions alone can easily produce contradictory arrangements. May individually look plausible, but they cannot necessarily form a valid puzzle configuration.

The ILP solver converts the pairwise predictions into a **global optimization problem**.

The objective is to find an arrangement of all puzzle pieces that maximizes the overall compatibility according to the neural network predictions while satisfying the puzzle constraints.

This gives the final reconstructed puzzle.

---

# 🗂️ Repository Structure

The repository contains the notebooks used throughout the complete experiment.

```text
.
├── notebooks/
│   ├── dataset_generater.ipynb
│   ├── prepare_dataset.ipynb
│   ├── JigNet.ipynb
│   └── JigNet_solver.ipynb
│
├── samples/
│   └── ...
│
├── README.md
└── ...
```


# 📓 Notebooks

The notebooks are organized according to the project pipeline.

## 1. Generate the Image Dataset

**Notebook:** `dataset_generater.ipynb`

The initial dataset consists of AI-generated images created using **SDXL Turbo**.

The notebook contains the process used to generate the original images that were later transformed into jigsaw puzzles.

The generated images provide a controlled way to create a relatively large collection of images with known ground-truth spatial relationships.

---

## 2. Prepare the Puzzle Dataset

**Notebook:** `prepare_dataset.ipynb`

This notebook transforms the original images into training examples for the neural network.

The main steps are:

```text
Original Image
      ↓
4 × 4 grid
      ↓
16 image tiles
      ↓
Generate neighboring tile pairs
      ↓
Generate negative pairs
      ↓
Train / Validation / Test split
```

For each original image, the image is divided into a **4 × 4 grid**, producing:

```text
16 tiles
```

The dataset then contains pairs of tiles with labels describing their spatial relationship.

### Positive examples

Pairs that are actual neighbors in the original image.

### Negative examples

Pairs that do not have the specified spatial relationship.

Both easy and harder negative examples are included to make the classification task more challenging.

---

# 🤖 Network Training

**Notebook:** `JigNet.ipynb`

This notebook contains:

* network architecture
* dataset loading
* training
* validation
* evaluation
* visualization of predictions
* analysis of the trained model

The model receives two tiles:

```text
Tile A + Tile B
```

and predicts their spatial relationship.

The output probabilities are used later by the puzzle solver.

---

# 🧩 ILP Solver

**Notebook:** `JigNet_solver.ipynb`

The ILP notebook takes the network predictions and attempts to reconstruct the complete puzzle.

The solver receives the shuffled tiles and the pairwise relationship scores produced by the trained network.

It then searches for an arrangement satisfying the puzzle constraints while maximizing the compatibility between neighboring tiles.

This separation between **learning** and **global optimization** is an important part of the project.

---

# 💾 Dataset and Model Weights

The complete dataset and experiment files are stored externally because the generated dataset is too large to keep directly in the Git repository.

The Google Drive directory contains:

* original generated images
* dataset splits
* metadata
* prepared dataset information
* trained network weights

**Dataset & weights:**
[Google Drive — Dataset and Model Files](https://drive.google.com/drive/folders/1MJpww_Z5NIvIe4TmYUoedSCpAcyxf4fW?usp=sharing)

---

# 🚀 Reproducing the Experiment

The complete experiment can be reproduced by following the notebooks in order.

You can generate you own dataset using `dataset_generater.ipynb` or obtain the original one from the Google Drive link above.

## Step 1 — Prepare the training data

Run:

```text
prepare_dataset.ipynb
```

## Step 2 — Train the network

Run:

```text
JigNet.ipynb
```

## Step 3 — Reconstruct a puzzle

Run:

```text
JigNet_solver.ipynb
```
---

# 🧪 Using the Trained Model

The trained model can also be used on a new image.

---

# 🌍 Generalization to Real Images

One of the most interesting observations from the experiment was that the network was able to reconstruct **real images despite being trained on AI-generated images**.

This was not explicitly expected from the dataset design.

The training images were generated using SDXL Turbo, while inference was also tested on images that were not part of the generated training distribution.

In general, the model was still able to identify useful spatial relationships between tiles and reconstruct the image.


# ⚠️ Failure Cases

The main failure cases observed during inference occurred when the image contained **large uniform regions**, particularly very wide white or dark areas.

The network had very few examples of such cases in the training dataset.

---

# 📊 Results

The project was evaluated at both the **network level** and the **puzzle reconstruction level**.

## Network Evaluation

The neural network was evaluated based on its ability to classify the spatial relationship between two tiles.

The evaluation notebook contains the detailed results and analysis.

Add your final network results here:

| Metric              | Result |
| ------------------- | -----: |
| Macro F1-Score on Validation |  `90.88%` |
| Weighted F1-Score on Validation      |  `93.15%` |
| Macro F1-Score on Test        |   `91.79%` |
| Weighted F1-Score on Test        |   `93.86%` |

---

## Puzzle Reconstruction

The final system was evaluated by comparing the reconstructed arrangement with the original image.

The main criterion was whether the complete set of tiles could be placed into their correct positions.

Add your final reconstruction results here:

| Evaluation                      |    Result |
| ------------------------------- | --------: |
| Correctly reconstructed puzzles | `XX / XX` |
| Reconstruction accuracy         |     `XX%` |

---
