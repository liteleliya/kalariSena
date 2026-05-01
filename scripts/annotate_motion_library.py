"""Annotate retargeted motion library with contacts, phase, and family labels."""

# TODO: Run Step 1 diagnostics and paste the real NPZ keys here.
# NPZ keys (from: python -c "import numpy as np; ...")
# TODO

# Step 3 CSV diagnostics (from: python -c "import pandas as pd; ...")
# File: cloud_outputs/otta_kaal_cloudgpu_20260417/ks/ks_retarget_g1.csv
# Columns: ['Frame', 'root_translateX', 'root_translateY', 'root_translateZ',
#           'root_rotateX', 'root_rotateY', 'root_rotateZ',
#           'left_hip_pitch_joint_dof', 'left_hip_roll_joint_dof',
#           'left_hip_yaw_joint_dof', 'left_knee_joint_dof',
#           'left_ankle_pitch_joint_dof', 'left_ankle_roll_joint_dof',
#           'right_hip_pitch_joint_dof', 'right_hip_roll_joint_dof',
#           'right_hip_yaw_joint_dof', 'right_knee_joint_dof',
#           'right_ankle_pitch_joint_dof', 'right_ankle_roll_joint_dof',
#           'waist_yaw_joint_dof', 'waist_roll_joint_dof', 'waist_pitch_joint_dof',
#           'left_shoulder_pitch_joint_dof', 'left_shoulder_roll_joint_dof',
#           'left_shoulder_yaw_joint_dof', 'left_elbow_joint_dof',
#           'left_wrist_roll_joint_dof', 'left_wrist_pitch_joint_dof',
#           'left_wrist_yaw_joint_dof', 'right_shoulder_pitch_joint_dof',
#           'right_shoulder_roll_joint_dof', 'right_shoulder_yaw_joint_dof',
#           'right_elbow_joint_dof', 'right_wrist_roll_joint_dof',
#           'right_wrist_pitch_joint_dof', 'right_wrist_yaw_joint_dof']

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception as exc:
    raise SystemExit("Missing dependency: pandas. Install it before running this script.") from exc

try:
    from scipy.spatial.transform import Rotation

    _HAVE_SCIPY_ROT = True
except Exception:
    _HAVE_SCIPY_ROT = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics.pinocchio_wrapper import PinocchioWrapper

VALID_FAMILIES = [
    "stable_stance",
    "single_support",
    "translational",
    "rotational",
    "explosive_strike",
    "compound",
    "recovery_critical",
]


def _motion_id_from_path(csv_path: Path) -> str:
    stem = csv_path.stem
    for suffix in ("_retarget_g1_from_bvh", "_retarget_g1"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=1)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(norm > 0, q / norm, q)
    w = np.clip(q[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    axis = np.zeros((q.shape[0], 3), dtype=np.float64)
    mask = sin_half > 1e-8
    axis[mask] = q[mask, 1:] / sin_half[mask, None]
    return axis * angle[:, None]


def _euler_to_quat_xyzw(
    euler: np.ndarray, order: str, degrees: bool
) -> np.ndarray:
    if _HAVE_SCIPY_ROT:
        rot = Rotation.from_euler(order, euler, degrees=degrees)
        return rot.as_quat()

    if order.lower() not in ("xyz", "zyx"):
        raise ValueError("Fallback euler conversion supports only 'xyz' or 'zyx'.")

    angles = np.deg2rad(euler) if degrees else euler
    rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, sx = np.cos(rx * 0.5), np.sin(rx * 0.5)
    cy, sy = np.cos(ry * 0.5), np.sin(ry * 0.5)
    cz, sz = np.cos(rz * 0.5), np.sin(rz * 0.5)

    if order.lower() == "xyz":
        w = cx * cy * cz - sx * sy * sz
        x = sx * cy * cz + cx * sy * sz
        y = cx * sy * cz - sx * cy * sz
        z = cx * cy * sz + sx * sy * cz
    else:  # zyx
        w = cz * cy * cx + sz * sy * sx
        x = cz * cy * sx - sz * sy * cx
        y = cz * sy * cx + sz * cy * sx
        z = sz * cy * cx - cz * sy * sx
    return np.stack([x, y, z, w], axis=1)


def _angular_velocity_from_quat(q_xyzw: np.ndarray, dt: float) -> np.ndarray:
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
    q_prev = q_wxyz[:-1]
    q_next = q_wxyz[1:]
    q_rel = _quat_mul(q_next, _quat_conjugate(q_prev))
    axis_angle = _quat_to_axis_angle(q_rel)
    omega = axis_angle / dt
    if omega.shape[0] == 0:
        return np.zeros((q_xyzw.shape[0], 3), dtype=np.float64)
    return np.vstack([omega[:1], omega])


def _median_filter_bool(data: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return data
    pad = kernel // 2
    padded = np.pad(data.astype(np.int32), ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(data, dtype=bool)
    for i in range(data.shape[0]):
        window = padded[i : i + kernel]
        out[i] = np.median(window, axis=0) >= 0.5
    return out


def _compute_phase(num_frames: int) -> np.ndarray:
    phase = (np.arange(num_frames) * 10.0 / max(1, num_frames)).astype(int)
    return np.clip(phase, 0, 9)


def _load_families(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    families: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        families[key.strip()] = value.strip()
    return families


def _write_families_stub(path: Path, motion_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fill in the family for each motion manually.",
        "# Valid families: stable_stance, single_support, translational,",
        "#                 rotational, explosive_strike, compound, recovery_critical",
    ]
    for motion_id in sorted(motion_ids):
        lines.append(f"{motion_id}: unlabeled")
    path.write_text("\n".join(lines) + "\n")


def _append_missing_families(path: Path, missing: list[str]) -> None:
    if not missing:
        return
    with path.open("a", encoding="utf-8") as f:
        for motion_id in sorted(missing):
            f.write(f"{motion_id}: unlabeled\n")


def _find_csvs(root: Path, include_from_bvh: bool) -> list[Path]:
    csvs = [Path(p) for p in root.glob("**/*_retarget_g1.csv")]
    if include_from_bvh:
        csvs.extend(Path(p) for p in root.glob("**/*_retarget_g1_from_bvh.csv"))
    return sorted(set(csvs))


def _load_npz_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as npz:
        for k in npz.files:
            data[k] = npz[k]
    return data


def _build_q_dq(
    df: "pd.DataFrame",
    fps: float,
    pos_scale: float,
    angles_deg: bool,
    euler_order: str,
    quat_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    required = [
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in CSV")

    joint_cols = [c for c in df.columns if c not in required and c != "Frame"]

    root_pos = df[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy()
    root_pos = root_pos * float(pos_scale)

    root_euler = df[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy()
    root_quat_xyzw = _euler_to_quat_xyzw(root_euler, euler_order, degrees=angles_deg)

    joint_pos = df[joint_cols].to_numpy()
    if angles_deg:
        joint_pos = np.deg2rad(joint_pos)

    dt = 1.0 / float(fps)
    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    root_ang_vel = _angular_velocity_from_quat(root_quat_xyzw, dt)
    joint_vel = np.gradient(joint_pos, dt, axis=0)

    if quat_order.lower() == "xyzw":
        quat = root_quat_xyzw
    elif quat_order.lower() == "wxyz":
        quat = np.concatenate([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]], axis=1)
    else:
        raise ValueError("quat_order must be 'xyzw' or 'wxyz'")

    q = np.concatenate([root_pos, quat, joint_pos], axis=1)
    dq = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=1)
    return q, dq, root_pos, root_quat_xyzw, joint_pos, joint_cols


def _process_csv(
    csv_path: Path,
    wrapper: PinocchioWrapper,
    args: argparse.Namespace,
    families: dict[str, str],
    output_dir: Path,
    npz_source_dir: Path | None,
) -> tuple[str, str]:
    df = pd.read_csv(csv_path)
    motion_id = _motion_id_from_path(csv_path)

    q, dq, root_pos, root_quat_xyzw, joint_pos, joint_cols = _build_q_dq(
        df,
        fps=args.fps,
        pos_scale=args.pos_scale,
        angles_deg=args.angles_deg,
        euler_order=args.root_euler_order,
        quat_order=args.quat_order,
    )

    if q.shape[1] != wrapper.model.nq:
        raise ValueError(
            f"q shape {q.shape} does not match model.nq={wrapper.model.nq}. "
            "Check URDF and CSV mapping."
        )
    if dq.shape[1] != wrapper.model.nv:
        raise ValueError(
            f"dq shape {dq.shape} does not match model.nv={wrapper.model.nv}. "
            "Check URDF and CSV mapping."
        )

    foot_positions = np.zeros((q.shape[0], 2, 3), dtype=np.float64)
    for t in range(q.shape[0]):
        lf, rf = wrapper.get_foot_positions(q[t], dq[t])
        foot_positions[t, 0] = lf
        foot_positions[t, 1] = rf

    contacts = np.stack(
        [
            foot_positions[:, 0, 2] < args.contact_z_threshold,
            foot_positions[:, 1, 2] < args.contact_z_threshold,
        ],
        axis=1,
    )
    contacts = _median_filter_bool(contacts, kernel=args.median_kernel)

    phase = _compute_phase(q.shape[0])

    family = families.get(motion_id, "unlabeled")
    if family == "unlabeled":
        print(f"[warn] Motion {motion_id} not labeled in {args.families_config}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{motion_id}.npz"

    base = {}
    if npz_source_dir is not None:
        candidate = npz_source_dir / f"{motion_id}.npz"
        base = _load_npz_if_exists(candidate)
    if not base and output_path.exists():
        base = _load_npz_if_exists(output_path)

    payload = dict(base)
    payload.update(
        {
            "q": q,
            "dq": dq,
            "root_pos": root_pos,
            "root_quat_xyzw": root_quat_xyzw,
            "joint_pos": joint_pos,
            "joint_cols": np.array(joint_cols, dtype=np.bytes_),
            "contacts": contacts.astype(bool),
            "phase": phase.astype(np.int32),
            "family": np.array(family.encode("utf-8")),
            "fps": np.array(float(args.fps), dtype=np.float32),
            "motion_id": np.array(motion_id.encode("utf-8")),
        }
    )

    np.savez(output_path, **payload)
    return motion_id, family


def _write_splits(
    ids_by_family: dict[str, list[str]],
    splits_dir: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for family, ids in sorted(ids_by_family.items()):
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            print(f"[warn] Family {family} has {n} motions; assigning to train only.")
            train_ids.extend(ids)
            continue
        n_train = max(1, int(round(n * 0.8)))
        n_val = max(1, int(round(n * 0.1)))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_val = max(1, n_val - 1)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val : n_train + n_val + n_test])

    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (splits_dir / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
    (splits_dir / "test_ids.txt").write_text("\n".join(test_ids) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", type=str, default="cloud_outputs")
    parser.add_argument("--output-dir", type=str, default="data/motions_retargeted")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--families-config", type=str, default="configs/motion_families.yaml")
    parser.add_argument("--npz-source-dir", type=str, default=None)
    parser.add_argument("--urdf", type=str, default="assets/unitree_g1/g1.urdf")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--angles-deg", action="store_true", default=True)
    parser.add_argument("--angles-rad", action="store_true", default=False)
    parser.add_argument("--root-euler-order", type=str, default="xyz")
    parser.add_argument("--quat-order", type=str, default="xyzw")
    parser.add_argument("--contact-z-threshold", type=float, default=0.05)
    parser.add_argument("--median-kernel", type=int, default=5)
    parser.add_argument("--include-from-bvh", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.angles_rad:
        args.angles_deg = False

    csv_root = Path(args.csv_root)
    families_path = Path(args.families_config)
    csvs = _find_csvs(csv_root, include_from_bvh=args.include_from_bvh)
    if args.limit:
        csvs = csvs[: args.limit]

    if not csvs:
        raise SystemExit(f"No retarget CSVs found under {csv_root}")

    motion_ids = [_motion_id_from_path(p) for p in csvs]
    families = _load_families(families_path)
    if not families:
        _write_families_stub(families_path, motion_ids)
        families = _load_families(families_path)

    missing = [mid for mid in motion_ids if mid not in families]
    if missing:
        _append_missing_families(families_path, missing)
        families = _load_families(families_path)

    wrapper = PinocchioWrapper(args.urdf)

    output_dir = Path(args.output_dir)
    npz_source_dir = Path(args.npz_source_dir) if args.npz_source_dir else None

    ids_by_family: dict[str, list[str]] = {}
    for csv_path in csvs:
        motion_id, family = _process_csv(
            csv_path,
            wrapper,
            args,
            families,
            output_dir,
            npz_source_dir,
        )
        ids_by_family.setdefault(family, []).append(motion_id)

    _write_splits(ids_by_family, Path(args.splits_dir), seed=args.seed)


def _smoke_test(args: argparse.Namespace) -> None:
    args.limit = 1
    main(args)
    output_dir = Path(args.output_dir)
    csvs = _find_csvs(Path(args.csv_root), include_from_bvh=args.include_from_bvh)
    motion_id = _motion_id_from_path(csvs[0])
    out_path = output_dir / f"{motion_id}.npz"
    assert out_path.exists(), f"Expected {out_path} to be created"
    with np.load(out_path, allow_pickle=True) as npz:
        assert "contacts" in npz.files
        contacts = npz["contacts"]
        contact_ratio = float(contacts.mean())
    print(f"Smoke test contact ratio: {contact_ratio:.3f}")


if __name__ == "__main__":
    args = _parse_args()
    if args.smoke_test:
        _smoke_test(args)
    else:
        main(args)
