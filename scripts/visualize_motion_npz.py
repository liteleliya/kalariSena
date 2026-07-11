import argparse
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _decode_scalar(value: np.ndarray, fallback: str = "") -> str:
    if getattr(value, "shape", None) != ():
        return fallback or str(value)
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _optional_array(motion: "np.lib.npyio.NpzFile", key: str, fallback: str = "") -> np.ndarray:
    if key in motion.files:
        return motion[key]
    return np.array(fallback)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--npz", required=True)
    parser.add_argument("--urdf", default=str(PROJECT_ROOT / "assets/unitree_g1/g1_29dof.urdf"))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--validate-only", action="store_true")

    args = parser.parse_args()

    motion = np.load(args.npz, allow_pickle=True)

    q = motion["q"]

    fps = float(motion["fps"])

    print("Motion ID:", _decode_scalar(_optional_array(motion, "motion_id")))
    print("Family:", _decode_scalar(_optional_array(motion, "family")))
    if "source_csv" in motion.files:
        print("Source CSV:", _decode_scalar(motion["source_csv"]))
    if "preprocessed" in motion.files:
        print("Preprocessed:", bool(motion["preprocessed"]))

    print("q shape:", q.shape)
    print("FPS:", fps)

    if q.ndim != 2:
        raise ValueError(f"Expected q to be [T, nq], got shape {q.shape}")
    if q.shape[1] != 36:
        raise ValueError(f"Expected G1 FreeFlyer q width 36, got {q.shape[1]}")

    quat_norm = np.linalg.norm(q[:, 3:7], axis=1)
    print("Quat norm range:", float(quat_norm.min()), "->", float(quat_norm.max()))
    if not np.allclose(quat_norm, 1.0, atol=1e-3):
        raise ValueError("Root quaternion norms are not close to 1.0")

    if "dq" in motion.files:
        print("dq shape:", motion["dq"].shape)
    if "contacts" in motion.files:
        contacts = motion["contacts"]
        print("contact ratio:", float(contacts.mean()))
    if "phase" in motion.files:
        print("phase range:", int(motion["phase"].min()), "->", int(motion["phase"].max()))

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    mesh_dir = urdf_path.parent

    if args.validate_only:
        print("Validation passed.")
        return

    try:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer
    except Exception as exc:
        raise SystemExit(
            "Visualization requires pinocchio and meshcat. Install pipeline dependencies first.\n"
            f"Original import error: {exc}"
        ) from exc

    print("Loading FreeFlyer G1...")

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path),
        str(mesh_dir),
        pin.JointModelFreeFlyer()
    )

    # print("Visual geometries:", len(visual_model.geometryObjects))

    # for g in visual_model.geometryObjects[:10]:
    #     print("Name:", g.name)
    #     print("Mesh:", g.meshPath)
    #     print()

    print("Model nq:", model.nq)
    print("Model nv:", model.nv)

    assert model.nq == q.shape[1]

    viz = MeshcatVisualizer(
        model,
        collision_model,
        visual_model
    )

    viz.initViewer(open=True)
    viz.loadViewerModel()

    # q0 = pin.neutral(model)

    # print("Neutral q:", q0[:10])

    # viz.display(q0)

    # input("Press Enter...")

    print("First q:", q[0][:10])

    print("root pos:", q[0][:3])
    print("quat:", q[0][3:7])

    print("quat norm:", np.linalg.norm(q[0][3:7]))

    print("Min root:", np.min(q[:, :3], axis=0))
    print("Max root:", np.max(q[:, :3], axis=0))

    # input("Enter")

    T = len(q)

    print("Frames:", T)
    print("Duration:", T / fps, "seconds")

    while True:

        for t in range(T):

            viz.display(q[t])

            time.sleep(1.0 / fps)

        if not args.loop:
            break


if __name__ == "__main__":
    main()
