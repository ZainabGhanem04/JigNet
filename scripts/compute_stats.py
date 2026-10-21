"""Channel mean/std of the training pairs (same definition as the training notebook:
statistics over tile_a and tile_b of every training PAIR, so tiles are weighted by how often they occur).

    python scripts/compute_stats.py --config configs/train.yaml
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import JigsawPairDataset
from utils.config import load_config, parse_args


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    ds = JigsawPairDataset(str(__import__("pathlib").Path(cfg.data.root) / cfg.data.pairs_csv), "train",
                           cfg.data.root, transforms.ToTensor(), cfg.data.get("legacy_prefix"))
    loader = DataLoader(ds, batch_size=128, num_workers=cfg.data.get("num_workers", 2), shuffle=False)
    s, sq, n = torch.zeros(3), torch.zeros(3), 0
    for a, b, _ in loader:
        x = torch.cat([a, b]).flatten(2)
        s += x.sum((0, 2)); sq += x.pow(2).sum((0, 2)); n += x.size(0) * x.size(2)
    mean = s / n
    std = (sq / n - mean.pow(2)).sqrt()
    print("mean:", mean.tolist())
    print("std: ", std.tolist())


if __name__ == "__main__":
    main()
