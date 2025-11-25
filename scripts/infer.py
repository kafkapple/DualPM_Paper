print("[DEBUG] Script starting...", flush=True)
import hydra
print("[DEBUG] Hydra imported", flush=True)
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
print("[DEBUG] Importing torch...", flush=True)
import torch
print("[DEBUG] Torch imported", flush=True)
from dualpm_paper.pointmaps import PointmapModule
print("[DEBUG] PointmapModule imported", flush=True)
import trimesh
print("[DEBUG] All imports done", flush=True)


@hydra.main(config_path="../configs", config_name="infer", version_base="1.3")
def main(cfg: DictConfig):
    print(f"[DEBUG] Config:\n{OmegaConf.to_yaml(cfg)}")

    weights_path = Path(cfg.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    print(f"[DEBUG] Weights found: {weights_path}")

    # Check directories
    feat_dir = Path(cfg.feat_dir)
    mask_dir = Path(cfg.mask_dir)
    print(f"[DEBUG] feat_dir exists: {feat_dir.exists()}, path: {feat_dir}")
    print(f"[DEBUG] mask_dir exists: {mask_dir.exists()}, path: {mask_dir}")

    if feat_dir.exists():
        feat_files = list(feat_dir.glob("*_feat.png"))
        print(f"[DEBUG] Found {len(feat_files)} feat files")
        if feat_files:
            print(f"[DEBUG] Sample feat files: {[f.name for f in feat_files[:3]]}")

    if mask_dir.exists():
        mask_files = list(mask_dir.glob("*_mask.png"))
        print(f"[DEBUG] Found {len(mask_files)} mask files")
        if mask_files:
            print(f"[DEBUG] Sample mask files: {[f.name for f in mask_files[:3]]}")

    dataset = hydra.utils.instantiate(cfg.dataset)
    print(f"[DEBUG] Dataset size: {len(dataset)}")
    module: PointmapModule = hydra.utils.instantiate(cfg.module)

    # map to CPU
    state_dicts = torch.load(weights_path, map_location="cpu")
    if "model_state" in state_dicts:
        model_weight_dict = state_dicts["model_state"]
    else:
        model_weight_dict = state_dicts

    module.model.load_state_dict(
        model_weight_dict,
        strict=False,
    )

    module.model.to(cfg.device)
    module.model.eval()
    module.device = cfg.device

    loader = hydra.utils.instantiate(
        cfg.dataloader, dataset=dataset, collate_fn=dataset.collate_fn
    )

    if len(dataset) == 0:
        print("[ERROR] Dataset is empty! Check:")
        print("  1. feat_dir and mask_dir paths are correct")
        print("  2. Files follow naming pattern: {id}_feat.png, {id}_mask.png")
        print("  3. IDs in feat_dir and mask_dir must match")
        return

    print(f"[INFO] Starting inference on {len(dataset)} samples...")

    def _loop():
        for batch in loader:
            file_ids, images, masks, feats = batch
            with torch.inference_mode():
                canon, posed, seq_mask = module.predict(
                    feats,
                    masks,
                    device=cfg.device,
                    confidence_threshold=cfg.confidence_threshold,
                )
            canon, posed, seq_mask = (t.clone().cpu() for t in (canon, posed, seq_mask))
            yield from zip(file_ids, canon, posed, seq_mask, strict=True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for file_id, canon, posed, seq_mask in _loop():
        canon_pointcloud = canon[seq_mask]
        reconstruction_pointcloud = posed[seq_mask]

        trimesh.PointCloud(vertices=canon_pointcloud).export(
            output_dir / f"{file_id}_canon.ply"
        )
        trimesh.PointCloud(vertices=reconstruction_pointcloud).export(
            output_dir / f"{file_id}_rec.ply"
        )
        processed += 1
        if processed % 10 == 0:
            print(f"[INFO] Processed {processed}/{len(dataset)} samples")

    print(f"[INFO] Done! Processed {processed} samples. Output: {output_dir}")


if __name__ == "__main__":
    main()
