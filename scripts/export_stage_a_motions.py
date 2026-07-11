"""Export Kalari retargeted NPZ files to unitree_rl_mjlab tracking NPZ files.

Stage A uses unitree_rl_mjlab's existing G1 tracking task. That task expects
motion files with simulated body trajectories:

  joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

Our retargeted files already contain root and joint references. This script
replays those references through the mjlab/MuJoCo robot model and records the
body trajectory arrays required by the tracking command loader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MJLAB_ROOT = PROJECT_ROOT / "unitree_rl_mjlab"

G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _import_mjlab_deps() -> dict[str, Any]:
    if not MJLAB_ROOT.exists():
        raise SystemExit(
            f"Missing unitree_rl_mjlab at {MJLAB_ROOT}. Run scripts/install_subprojects.sh first."
        )
    sys.path.insert(0, str(MJLAB_ROOT))
    try:
        import torch
        from mjlab.entity import Entity
        from mjlab.scene import Scene
        from mjlab.sim.sim import Simulation, SimulationCfg
        from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    except Exception as exc:
        if "mjENBL_MULTICCD" in str(exc):
            raise SystemExit(
                "Stage A dependencies are installed, but the active MuJoCo Python package is "
                "not compatible with mujoco-warp 3.5.0. Pin MuJoCo to the 3.5.x family in "
                "this same conda environment:\n"
                "  python -m pip install --force-reinstall 'mujoco>=3.5.0,<3.6.0' 'mujoco-warp==3.5.0'\n"
                "Then verify:\n"
                "  python -c \"import mujoco; print(mujoco.__version__, hasattr(mujoco.mjtEnableBit, 'mjENBL_MULTICCD'))\"\n"
                f"Original import error: {exc}"
            ) from exc
        raise SystemExit(
            "Missing Stage A training dependencies. Install unitree_rl_mjlab first:\n"
            "  cd unitree_rl_mjlab\n"
            "  pip install -e .\n"
            f"Original import error: {exc}"
        ) from exc

    return {
        "torch": torch,
        "Entity": Entity,
        "Scene": Scene,
        "Simulation": Simulation,
        "SimulationCfg": SimulationCfg,
        "unitree_g1_flat_tracking_env_cfg": unitree_g1_flat_tracking_env_cfg,
    }


def _decode_scalar(value: np.ndarray, fallback: str) -> str:
    if value.shape != ():
        return fallback
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)


def _load_reference_motion(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        files = set(data.files)
        if "joint_pos" not in files:
            raise ValueError(f"{path} is missing required key 'joint_pos'")

        if "root_pos" in files:
            root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        elif "q" in files:
            root_pos = np.asarray(data["q"][:, :3], dtype=np.float32)
        else:
            raise ValueError(f"{path} is missing 'root_pos' or 'q'")

        if "root_quat_xyzw" in files:
            root_quat_xyzw = np.asarray(data["root_quat_xyzw"], dtype=np.float32)
        elif "q" in files:
            root_quat_xyzw = np.asarray(data["q"][:, 3:7], dtype=np.float32)
        else:
            raise ValueError(f"{path} is missing 'root_quat_xyzw' or 'q'")

        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        if joint_pos.shape[1] != len(G1_JOINT_NAMES):
            raise ValueError(
                f"{path} joint_pos has {joint_pos.shape[1]} joints; expected {len(G1_JOINT_NAMES)}"
            )

        if "joint_vel" in files:
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        elif "dq" in files:
            joint_vel = np.asarray(data["dq"][:, 6 : 6 + len(G1_JOINT_NAMES)], dtype=np.float32)
        else:
            fps_for_grad = float(data["fps"]) if "fps" in files else 50.0
            joint_vel = np.gradient(joint_pos, 1.0 / fps_for_grad, axis=0).astype(np.float32)

        if "dq" in files:
            root_lin_vel = np.asarray(data["dq"][:, :3], dtype=np.float32)
            root_ang_vel = np.asarray(data["dq"][:, 3:6], dtype=np.float32)
        else:
            fps_for_grad = float(data["fps"]) if "fps" in files else 50.0
            root_lin_vel = np.gradient(root_pos, 1.0 / fps_for_grad, axis=0).astype(np.float32)
            root_ang_vel = np.zeros_like(root_lin_vel)

        fps = float(data["fps"]) if "fps" in files else 50.0
        motion_id = _decode_scalar(data["motion_id"], path.stem) if "motion_id" in files else path.stem

    if not (
        root_pos.shape[0]
        == root_quat_xyzw.shape[0]
        == joint_pos.shape[0]
        == joint_vel.shape[0]
        == root_lin_vel.shape[0]
        == root_ang_vel.shape[0]
    ):
        raise ValueError(f"{path} has inconsistent trajectory lengths")

    return {
        "motion_id": motion_id,
        "fps": fps,
        "root_pos": root_pos,
        "root_quat_wxyz": _xyzw_to_wxyz(root_quat_xyzw),
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }


def _export_one(input_path: Path, output_path: Path, device: str) -> None:
    deps = _import_mjlab_deps()
    torch = deps["torch"]
    Scene = deps["Scene"]
    Simulation = deps["Simulation"]
    SimulationCfg = deps["SimulationCfg"]
    unitree_g1_flat_tracking_env_cfg = deps["unitree_g1_flat_tracking_env_cfg"]

    ref = _load_reference_motion(input_path)

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / ref["fps"]

    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    robot = scene["robot"]
    joint_indexes = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]

    log: dict[str, list[np.ndarray] | list[float] | str] = {
        "fps": [ref["fps"]],
        "source_motion_id": ref["motion_id"],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    root_pos = torch.tensor(ref["root_pos"], dtype=torch.float32, device=device)
    root_quat = torch.tensor(ref["root_quat_wxyz"], dtype=torch.float32, device=device)
    root_lin_vel = torch.tensor(ref["root_lin_vel"], dtype=torch.float32, device=device)
    root_ang_vel = torch.tensor(ref["root_ang_vel"], dtype=torch.float32, device=device)
    joint_pos_ref = torch.tensor(ref["joint_pos"], dtype=torch.float32, device=device)
    joint_vel_ref = torch.tensor(ref["joint_vel"], dtype=torch.float32, device=device)

    scene.reset()
    for t in range(root_pos.shape[0]):
        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = root_pos[t : t + 1]
        root_states[:, 3:7] = root_quat[t : t + 1]
        root_states[:, 7:10] = root_lin_vel[t : t + 1]
        root_states[:, 10:13] = root_ang_vel[t : t + 1]
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indexes] = joint_pos_ref[t : t + 1]
        joint_vel[:, joint_indexes] = joint_vel_ref[t : t + 1]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())  # type: ignore[union-attr]
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())  # type: ignore[union-attr]
        log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())  # type: ignore[union-attr]
        log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())  # type: ignore[union-attr]
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())  # type: ignore[union-attr]
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())  # type: ignore[union-attr]

    payload: dict[str, Any] = {}
    for key, value in log.items():
        if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            payload[key] = np.stack(value, axis=0)
        else:
            payload[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"Wrote {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/motions_retargeted")
    parser.add_argument("--output-dir", type=Path, default=MJLAB_ROOT / "src/assets/motions/g1/kalari")
    parser.add_argument("--motion-id", action="append", default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    paths = sorted(input_dir.glob("*.npz"))
    if args.motion_id:
        wanted = set(args.motion_id)
        paths = [path for path in paths if path.stem in wanted]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No input NPZ files found in {input_dir}")

    for input_path in paths:
        _export_one(input_path, output_dir / input_path.name, device=args.device)


if __name__ == "__main__":
    main()
