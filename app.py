"""
Jigsaw puzzle solver demo (Gradio / Hugging Face Spaces)

Pipeline
    1. user uploads an image
    2. image is made square (center crop) and resized to 512x512
    3. image is split into a 4x4 grid (16 tiles of 128x128) and the tiles are shuffled
    4. JigNet predicts, for every ordered pair of tiles, a softmax over
       [NO_MATCH, LEFT, RIGHT, UP, DOWN]
    5. pairwise scores are fed to the ILP (PuLP) which finds the best 4x4 arrangement
    6. the arrangement is stitched back into the final image

Run locally:   python app.py            (weights: jignet_weights.pth, or set JIGNET_WEIGHTS=runs/jignet_exp2/last.pth)
On HF Spaces:  upload app.py, requirements.txt, data/tiling.py (+ data/__init__.py), models/, solver/, utils/
               and the weights file next to app.py.
"""
import os
import random

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

from data.tiling import split_grid, stitch
from models.checkpoint import load_jignet
from models.inference import predict_probs_matrix, prepare_tiles
from solver import build_pairwise_scores, harden, solve_jigsaw_ilp
from utils.constants import MEAN, STD, TILE_SIZE

WEIGHTS_PATH = os.environ.get("JIGNET_WEIGHTS", "jignet_weights.pth")
GRID_SIZE = 4
NUM_TILES = GRID_SIZE * GRID_SIZE      # 16
IMAGE_SIZE = GRID_SIZE * TILE_SIZE     # 512

# True  -> feed the real softmax probabilities to the ILP (what all reported benchmarks use)
# False -> one-hot argmax predictions (ablation)
USE_SOFT_PROBS = True
ILP_TIME_LIMIT_SEC = int(os.environ.get("ILP_TIME_LIMIT_SEC", 120))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_jignet(WEIGHTS_PATH, device)


def prepare_image(img: Image.Image) -> Image.Image:
    """Fix EXIF rotation, force RGB, center-crop to square, resize to 512x512."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.fit(img, (IMAGE_SIZE, IMAGE_SIZE), method=Image.LANCZOS, centering=(0.5, 0.5))


def split_into_tiles(img: Image.Image):
    """512x512 image -> list of 16 uint8 RGB arrays (128x128x3), row-major."""
    return [t.copy() for t in split_grid(np.asarray(img), GRID_SIZE, TILE_SIZE, TILE_SIZE)]


def shuffle_step(image):
    """Upload -> preprocess -> split -> shuffle. Returns (preview, shuffled grid, state, cleared result, cleared status)."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    prepared = prepare_image(image)
    tiles = split_into_tiles(prepared)

    perm = list(range(NUM_TILES))          # perm[k] = original index of the tile shown in cell k
    while perm == list(range(NUM_TILES)):  # never return an already-solved puzzle
        random.shuffle(perm)

    shuffled_tiles = [tiles[p] for p in perm]
    state = {"tiles": shuffled_tiles, "perm": perm}
    return prepared, Image.fromarray(stitch(shuffled_tiles, grid_size=GRID_SIZE, gap=4)), state, None, ""


def solve_step(state):
    """Run JigNet on all pairs, solve the ILP, stitch the result."""
    if not state:
        raise gr.Error("Upload an image and press 'Shuffle tiles' first.")

    tiles, perm = state["tiles"], state["perm"]

    probs_matrix = predict_probs_matrix(model, prepare_tiles(tiles, MEAN, STD, device))
    if not USE_SOFT_PROBS:
        probs_matrix = harden(probs_matrix)
    result = solve_jigsaw_ilp(build_pairwise_scores(probs_matrix), GRID_SIZE, ILP_TIME_LIMIT_SEC)
    if result.board is None:
        raise gr.Error(f"The ILP did not return a valid arrangement (status: {result.status}).")

    order = [result.board[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    solved = Image.fromarray(stitch(tiles, order, grid_size=GRID_SIZE))

    correct = sum(perm[s] == k for k, s in enumerate(order))
    msg = f"Tiles in their original position: {correct}/{NUM_TILES}"
    if correct == NUM_TILES:
        msg += "  ✅ perfectly reconstructed"
    return solved, msg


with gr.Blocks(title="JigNet Jigsaw Solver") as demo:
    gr.Markdown(
        "# 🧩 JigNet Jigsaw Solver\n"
        "Upload an image → it is cropped to a square, resized to 512×512 and cut into a "
        "4×4 grid. The tiles are shuffled, then **JigNet** (pairwise tile network) and an "
        "**ILP solver** put them back together."
    )

    state = gr.State()

    with gr.Row():
        inp = gr.Image(type="pil", label="1. Upload an image", height=400)
        prepared_out = gr.Image(label="2. Preprocessed (512×512)", height=400, interactive=False)

    shuffle_btn = gr.Button("Shuffle tiles", variant="secondary")
    shuffled_out = gr.Image(label="3. Shuffled tiles (4×4)", height=520, interactive=False)
    solve_btn = gr.Button("Solve puzzle", variant="primary")
    solved_out = gr.Image(label="4. Reconstructed image", height=520, interactive=False)
    status = gr.Textbox(label="Result", interactive=False)

    shuffle_btn.click(shuffle_step, inputs=inp, outputs=[prepared_out, shuffled_out, state, solved_out, status])
    solve_btn.click(solve_step, inputs=state, outputs=[solved_out, status])

if __name__ == "__main__":
    demo.launch()
