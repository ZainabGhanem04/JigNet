"""Generate the synthetic image dataset with SDXL Turbo.

    python -m data.generate --config configs/generate_dataset.yaml

Every image is saved as <out_dir>/img_<scene_idx>_<k>.png and recorded in generation_manifest.csv
(prompt, seed, sampler settings) so the dataset can be regenerated and described in a paper.
"""
from __future__ import annotations

import csv
from pathlib import Path

from utils.config import load_config, parse_args, save_config


def load_scenes(path: str, dedupe: bool = False):
    scenes = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if dedupe:
        scenes = list(dict.fromkeys(scenes))
    return scenes


def main():
    args = parse_args("Generate a dataset with SDXL Turbo")
    cfg = load_config(args.config, args.overrides)

    import torch
    from diffusers import AutoPipelineForText2Image   # pip install -r requirements-gen.txt

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "generation_config.yaml")

    scenes = load_scenes(cfg.scenes_file, cfg.get("dedupe_scenes", False))
    print(f"{len(scenes)} scenes x {cfg.images_per_scene} image(s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(cfg.model_id, torch_dtype=dtype).to(device)
    pipe.enable_attention_slicing()

    manifest_path = out_dir / "generation_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "scene_idx", "k", "seed", "prompt", "steps", "guidance_scale", "model_id"])
        for i, scene in enumerate(scenes):
            prompt = cfg.base_prompt.format(scene=scene).strip()
            for k in range(cfg.images_per_scene):
                seed = int(cfg.seed) + i * 1000 + k          # one reproducible seed per image
                filename = f"img_{i}_{k}.png"
                path = out_dir / filename
                if path.exists() and cfg.get("skip_existing", True):
                    continue
                gen = torch.Generator(device=device).manual_seed(seed)
                image = pipe(prompt=prompt, num_inference_steps=cfg.steps,
                             guidance_scale=cfg.guidance_scale, generator=gen).images[0]
                image.save(path)
                writer.writerow([filename, i, k, seed, prompt, cfg.steps, cfg.guidance_scale, cfg.model_id])
                f.flush()
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
