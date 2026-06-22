"""
visualize_motion_meshcat.py
===========================
Drop-in equivalent of `scripts/visualize_motion_npz.py`, but using
`yourdfpy` + `meshcat` instead of `pinocchio` (which has no Windows pip wheel).

It reads the SAME .npz format produced by `csv_to_npz.py`:

    q          : (T, 36) FreeFlyer config -> [pos(3), quat xyzw(4), joints(29)]
    fps        : playback rate
    motion_id  : metadata string
    family     : metadata string

A browser window opens automatically and the G1 plays back the motion.

Run:
    .\\.venv\\Scripts\\python.exe visualize_motion_meshcat.py \
        --npz "Data/Urumi_Sword_retarget_g1_step1.npz" \
        --urdf "unitree_ros/robots/g1_description/g1_29dof.urdf" --loop
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import yourdfpy
import meshcat
import meshcat.geometry as mg


def base_transform(pos, quat_xyzw):
    """Build a 4x4 world transform for the floating base."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--color", default="0x4477aa",
                        help="Hex robot color, e.g. 0x4477aa.")
    args = parser.parse_args()

    # --- load motion ------------------------------------------------------
    motion = np.load(args.npz, allow_pickle=True)
    q = motion["q"]
    fps = float(motion["fps"])
    print("Motion ID:", motion["motion_id"])
    print("Family:", motion["family"])
    print("q shape:", q.shape)
    print("FPS:", fps)

    # --- load robot -------------------------------------------------------
    print("Loading G1 (yourdfpy)...")
    urdf = yourdfpy.URDF.load(
        str(Path(args.urdf)),
        load_meshes=True,
        build_collision_scene_graph=False,
    )
    scene = urdf.scene

    n_joints = len(urdf.actuated_joint_names)
    nq = 7 + n_joints
    assert q.shape[1] == nq, f"q has {q.shape[1]} cols, expected {nq}"

    # --- start meshcat ----------------------------------------------------
    print("Opening meshcat in browser...")
    vis = meshcat.Visualizer()
    vis.open()

    # Send each link mesh to the viewer once. We keep the node->path mapping
    # so we can re-pose them every frame.
    color = int(args.color, 16)
    material = mg.MeshLambertMaterial(color=color)
    node_names = list(scene.graph.nodes_geometry)
    for name in node_names:
        geom_T, geom_name = scene.graph[name]
        mesh = scene.geometry[geom_name]
        vis["robot"][name].set_object(
            mg.TriangularMeshGeometry(mesh.vertices, mesh.faces), material
        )
    print(f"Loaded {len(node_names)} link meshes.")

    # --- some quick stats (mirrors the original script) -------------------
    print("First q:", q[0][:10])
    print("root pos:", q[0][:3])
    print("quat:", q[0][3:7], "norm:", np.linalg.norm(q[0][3:7]))
    print("Min root:", np.min(q[:, :3], axis=0))
    print("Max root:", np.max(q[:, :3], axis=0))

    T = len(q)
    print("Frames:", T, "Duration:", T / fps, "seconds")

    # --- playback loop ----------------------------------------------------
    while True:
        for t in range(T):
            pos = q[t, 0:3]
            quat = q[t, 3:7]              # xyzw
            joints = q[t, 7:]

            urdf.update_cfg(joints)       # forward kinematics
            T_base = base_transform(pos, quat)

            for name in node_names:
                node_T, _ = scene.graph[name]
                vis["robot"][name].set_transform(T_base @ node_T)

            time.sleep(1.0 / fps)

        if not args.loop:
            break


if __name__ == "__main__":
    main()
