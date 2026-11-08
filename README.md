# 🧩 Deep Learning Approach to Solving Jigsaw Puzzles

A learning-based approach for reconstructing a jigsaw puzzle from shuffled image tiles.

The project explores whether a neural network can learn **spatial relationships between image tiles** and use those learned relationships to reconstruct an entire image. The system combines a **Siamese convolutional neural network** for local tile-pair prediction with **Integer Linear Programming (ILP)** for globally consistent puzzle reconstruction.

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

---

1. Sample from the original AI generated dataset
<table>
  <tr>
    <td>
      <img src="demo/original_1.webp" alt="Dragon Image" width="300" />
    </td>
    <td>
      <img src="demo/shuffled_1.webp" alt="Shuffled Dragon Image" width="300" />
    </td>
    <td>
      <img src="demo/reconstructed_1.png" alt="Reconstructed Dragon Image" width="300"/>
    </td>
  </tr>
</table>
2. Sample for a landscape image
<table>
  <tr>
    <td>
      <img src="demo/original_2.jpeg" alt="City scene" width="300" />
    </td>
    <td>
      <img src="demo/shuffled_2.webp" alt="Shuffled City scene" width="300" />
    </td>
    <td>
      <img src="demo/reconstructed_2.webp" alt="Reconstructed City scene" width="300"/>
    </td>
  </tr>
</table>
3. Sample from COCO dataset
<table>
  <tr>
    <td>
      <img src="demo/original_3.jpg" alt="Man skiing" width="300" />
    </td>
    <td>
      <img src="demo/shuffled_3.webp" alt="Shuffled Image" width="300" />
    </td>
    <td>
      <img src="demo/reconstruted_3.webp" alt="Reconstructed Image" width="300"/>
    </td>
  </tr>
</table>

# Repository Structure

The repository contains the notebooks used throughout the complete experiment.

```text
.
├── notebooks/
│   ├── dataset_generater.ipynb
│   ├── prepare_dataset.ipynb
│   ├── JigNet.ipynb
│   └── JigNet_Evaluation.ipynb
│   └── JigSolver.ipynb
│   └── COCO_prep.ipynb
│
├── final_model/
│   └── jignet_models.py
│   └── jignet_weights.pth
│
├── eval_results/
  └── ...
│
├── samples/
│   └── ...
│
├── README.md
```


# Notebooks

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


# Network Training

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

# ILP Solver

**Notebook:** `JigSolver.ipynb`

The ILP notebook takes the network predictions and attempts to reconstruct the complete puzzle.

The solver receives the shuffled tiles and the pairwise relationship scores produced by the trained network.

It then searches for an arrangement satisfying the puzzle constraints while maximizing the compatibility between neighboring tiles.

---

# Dataset and Model Weights

The complete dataset and experiment files are stored externally because the generated dataset is too large to keep directly in the Git repository.

The Google Drive directory contains:

* original generated images
* dataset splits
* metadata
* prepared dataset information
* trained network weights
* notebooks
* evalution results and report


[Google Drive — Dataset and Model Files](https://drive.google.com/drive/folders/1MJpww_Z5NIvIe4TmYUoedSCpAcyxf4fW?usp=sharing)


# Generalization to Real Images

One of the most interesting observations from the experiment was that the network was able to reconstruct **real images despite being trained on AI-generated images**.

This was not explicitly expected from the dataset design.

The training images were generated using SDXL Turbo, while inference was also tested on a subset of COCO benchmark (250 images).

In general, the model was still able to identify useful spatial relationships between tiles and reconstruct the image.


# ⚠️ Failure Cases

The main failure cases observed during inference occurred when the image contained **large uniform regions**, particularly very wide white or dark areas.

The network had very few examples of such cases in the training dataset.

---

# Results

The project was evaluated at both the **network level** and the **puzzle reconstruction level** on original test set and COCO subset for different grid sizes (3, 4, 5, 6).

## Network Evaluation

The neural network was evaluated based on its ability to classify the spatial relationship between two tiles.

The Drive link contains the detailed results and analysis. I'll write a full report soon.

