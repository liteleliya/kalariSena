"""
csv_to_npz.py
=============
Convert a retargeted G1 CSV (raw or preprocessed) into the .npz format that
`scripts/visualize_motion_npz.py` (and our meshcat viewer) expects.

Output q layout per frame (Pinocchio FreeFlyer convention, nq = 36):

    [ px, py, pz,      # root position   (meters)
      qx, qy, qz, qw,  # root orientation (quaternion, xyzw)
      j1 ... j29 ]     # joint angles    (radians)

Units in the CSV: translations in centimeters, all angles in degrees.

Run:
    .\\.venv\\Scripts\\python.exe csv_to_npz.py \
        --csv "Data/Urumi_Sword_retarget_g1_step1.csv" \
        --out "Data/Urumi_Sword_retarget_g1_step1.npz"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


CM_TO_M = 0.01
DEG_TO_RAD = np.pi / 180.0

ROOT_POS_COLS = ["root_translateX", "root_translateY", "root_translateZ"]
ROOT_ROT_COLS = ["root_rotateX", "root_rotateY", "root_rotateZ"]


def main():
    parser = argparse.ArgumentParser(description="CSV -> NPZ for G1 motion.")
    parser.add_argument("--csv", required=True, help="Input motion CSV.")
    parser.add_argument("--out", default=None, help="Output .npz path.")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--motion-id", default=None,
                        help="Metadata id (default: CSV file stem).")
    parser.add_argument("--family", default="urumi_sword",
                        help="Motion family/category metadata.")
    parser.add_argument("--euler-order", default="XYZ",
                        help="Euler order of root_rotate columns (default XYZ).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    joint_cols = [c for c in df.columns if c.endswith("_dof")]

    T = len(df)
    nq = 7 + len(joint_cols)
    print(f"Frames: {T}, joints: {len(joint_cols)}, nq: {nq}")

    # Root position: cm -> m
    root_pos = df[ROOT_POS_COLS].to_numpy(dtype=float) * CM_TO_M

    # Root orientation: Euler degrees -> quaternion (xyzw)
    euler_deg = df[ROOT_ROT_COLS].to_numpy(dtype=float)
    quat_xyzw = Rotation.from_euler(
        args.euler_order, euler_deg, degrees=True
    ).as_quat()  # scipy returns (x, y, z, w)

    # Joints: degrees -> radians
    joints_rad = df[joint_cols].to_numpy(dtype=float) * DEG_TO_RAD

    # Assemble q
    q = np.concatenate([root_pos, quat_xyzw, joints_rad], axis=1)
    assert q.shape == (T, nq), q.shape

    out_path = Path(args.out) if args.out else csv_path.with_suffix(".npz")
    motion_id = args.motion_id or csv_path.stem

    np.savez(
        out_path,
        q=q,
        fps=np.float64(args.fps),
        motion_id=motion_id,
        family=args.family,
    )
    print(f"Saved -> {out_path}")
    print(f"  q shape: {q.shape}")
    print(f"  root pos range (m): min={root_pos.min(0)}, max={root_pos.max(0)}")
    print(f"  first quat (xyzw): {quat_xyzw[0]} |norm|={np.linalg.norm(quat_xyzw[0]):.4f}")


if __name__ == "__main__":
    main()
