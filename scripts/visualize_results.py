#!/usr/bin/env python3
"""
Visualize inference results using Rerun web viewer.

Usage:
    python scripts/visualize_results.py --output_dir /path/to/outputs

    Then open browser at http://localhost:9876
"""

import argparse
from pathlib import Path
import numpy as np
import trimesh
import rerun as rr


def load_ply(path: Path) -> np.ndarray:
    """Load PLY file and return vertices."""
    mesh = trimesh.load(path)
    return np.array(mesh.vertices)


def get_sample_ids(output_dir: Path) -> list[str]:
    """Get all unique sample IDs from output directory."""
    canon_files = list(output_dir.glob("*_canon.ply"))
    ids = sorted([f.stem.replace("_canon", "") for f in canon_files])
    return ids


def visualize_sample(sample_id: str, output_dir: Path):
    """Log a single sample to rerun."""
    canon_path = output_dir / f"{sample_id}_canon.ply"
    rec_path = output_dir / f"{sample_id}_rec.ply"

    if not canon_path.exists() or not rec_path.exists():
        print(f"[WARN] Missing files for {sample_id}")
        return False

    canon_pts = load_ply(canon_path)
    rec_pts = load_ply(rec_path)

    # Log canonical pointcloud (left side, blue)
    rr.log(
        "canonical",
        rr.Points3D(
            canon_pts,
            colors=np.tile([100, 149, 237], (len(canon_pts), 1)),  # Cornflower blue
            radii=0.005,
        ),
    )

    # Log reconstruction pointcloud (right side, offset, orange)
    # Offset to the right for side-by-side view
    offset = np.array([1.5, 0, 0])  # Offset on X axis
    rr.log(
        "reconstruction",
        rr.Points3D(
            rec_pts + offset,
            colors=np.tile([255, 140, 0], (len(rec_pts), 1)),  # Dark orange
            radii=0.005,
        ),
    )

    # Log info
    rr.log("info/sample_id", rr.TextLog(f"Sample: {sample_id}"))
    rr.log("info/canon_points", rr.TextLog(f"Canon points: {len(canon_pts)}"))
    rr.log("info/rec_points", rr.TextLog(f"Rec points: {len(rec_pts)}"))

    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize DualPM inference results")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory with PLY files")
    parser.add_argument("--port", type=int, default=9876, help="Web viewer port")
    parser.add_argument("--sample", type=str, default=None, help="Specific sample ID to view")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"[ERROR] Output directory not found: {output_dir}")
        return

    sample_ids = get_sample_ids(output_dir)
    print(f"[INFO] Found {len(sample_ids)} samples")

    if len(sample_ids) == 0:
        print("[ERROR] No samples found!")
        return

    # Initialize rerun with web viewer
    rr.init("DualPM Results", spawn=False)
    rr.serve(open_browser=False, web_port=args.port)

    print(f"\n{'='*50}")
    print(f"🌐 Open browser at: http://localhost:{args.port}")
    print(f"{'='*50}\n")

    # If specific sample requested
    if args.sample:
        if args.sample in sample_ids:
            rr.set_time_sequence("sample", 0)
            visualize_sample(args.sample, output_dir)
            print(f"[INFO] Showing sample: {args.sample}")
        else:
            print(f"[ERROR] Sample {args.sample} not found")
            print(f"[INFO] Available samples: {sample_ids[:10]}...")
        input("\nPress Enter to exit...")
        return

    # Visualize all samples with timeline
    print("[INFO] Loading samples... (use timeline in Rerun to browse)")
    for i, sample_id in enumerate(sample_ids):
        rr.set_time_sequence("sample", i)
        if visualize_sample(sample_id, output_dir):
            if (i + 1) % 10 == 0:
                print(f"[INFO] Loaded {i + 1}/{len(sample_ids)} samples")

    print(f"\n[INFO] All {len(sample_ids)} samples loaded!")
    print("[INFO] Use the timeline slider in Rerun to browse samples")
    print("[INFO] Blue = Canonical, Orange = Reconstruction (offset)")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
