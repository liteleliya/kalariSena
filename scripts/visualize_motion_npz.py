import argparse
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--npz", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--loop", action="store_true")

    args = parser.parse_args()

    motion = np.load(args.npz, allow_pickle=True)

    q = motion["q"]

    fps = float(motion["fps"])

    print("Motion ID:", motion["motion_id"])
    print("Family:", motion["family"])

    print("q shape:", q.shape)
    print("FPS:", fps)

    urdf_path = Path(args.urdf)
    mesh_dir = urdf_path.parent

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