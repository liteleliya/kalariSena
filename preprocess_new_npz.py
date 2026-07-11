import numpy as np
import pinocchio as pin
from scipy.signal import argrelmin, argrelmax

URDF_PATH    = "/Users/aayush/Documents/Pragya AI Research/Experiments/unitree_ros/robots/g1_description/g1_29dof.urdf"  # adjust path
PACKAGE_DIRS = ["/Users/aayush/Documents/Pragya AI Research/Experiments/unitree_ros/robots/g1_description"]
G            = 9.81
VEL_THRESH   = 0.01

# ── Load model once ────────────────────────────────────────────────────────
model, _, _ = pin.buildModelsFromUrdf(
    URDF_PATH, PACKAGE_DIRS, pin.JointModelFreeFlyer()
)
data = model.createData()

def z_min_at_frame(root_pos, root_quat, q_joints):
    """
    Compute the lowest point of the ENTIRE robot body at this pose,
    using real forward kinematics — not just root height.
    """
    q_full = np.concatenate([root_pos, root_quat, q_joints])
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)

    min_z = np.inf
    for frame_id in range(model.nframes):
        z = data.oMf[frame_id].translation[2]
        if z < min_z:
            min_z = z
    return min_z


def correct_root_height_proper(root_pos, root_quat, q_joints, vel_thresh=VEL_THRESH):
    """
    Proper Algorithm 1 implementation using real zmin(qt) via FK.
    """
    T = len(root_pos)
    P = root_pos[:, 2].copy()          # root z trajectory
    P_dot = np.diff(P, append=P[-1])
    P_hat = np.zeros(T)

    # zmin at every frame BEFORE correction
    zmin_arr = np.array([
        z_min_at_frame(root_pos[t], root_quat[t], q_joints[t])
        for t in range(T)
    ])

    # Initialize: shift root so lowest body part touches ground at frame 0
    P_hat[0] = P[0] - zmin_arr[0]

    local_max = argrelmax(P, order=3)[0]
    local_min = argrelmin(P, order=3)[0]

    for t in range(1, T):
        if t in local_min:
            # Try snapping: shift root so zmin touches ground
            candidate = P[t] - zmin_arr[t]
            if zmin_arr[t] >= 0:
                P_hat[t] = candidate
                continue
        P_hat[t] = P_hat[t-1] + (P_dot[t-1] if P_dot[t-1] > vel_thresh else 0.0)

    # Parabolic reconstruction for jump segments
    for t_s in local_max:
        candidates = local_min[local_min > t_s]
        if len(candidates) == 0:
            continue
        t_e = candidates[0]
        y0, y1 = P_hat[t_s], P_hat[t_e]
        if y0 <= y1:
            continue
        T_flight = np.sqrt(2 * (y0 - y1) / G)
        N = t_e - t_s - 1
        for k in range(1, N + 1):
            dt = (k / (N + 1)) * T_flight
            P_hat[t_s + k] = y0 - 0.5 * G * dt ** 2

    # Final ground penetration check using REAL zmin, recomputed with new root
    root_pos_corrected = root_pos.copy()
    root_pos_corrected[:, 2] = P_hat
    for t in range(T):
        zm = z_min_at_frame(root_pos_corrected[t], root_quat[t], q_joints[t])
        if zm < 0:
            root_pos_corrected[t, 2] -= zm   # push root up by the penetration amount

    return root_pos_corrected[:, 2]


# ── Test on one file ──────────────────────────────────────────────────────
motion = np.load("/Users/aayush/Documents/Pragya AI Research/Experiments/kalariSena/data/motions_retargeted_raw/Urumi_Sword.npz", allow_pickle=True)
root_pos  = motion["root_pos"].copy()
root_quat = motion["root_quat_xyzw"]
q_ref     = motion["q"]

print("Before correction — root Z range:", root_pos[:,2].min(), "->", root_pos[:,2].max())

# Check actual zmin BEFORE any correction — this tells us how bad it is
zmin_before = np.array([
    z_min_at_frame(root_pos[t], root_quat[t], q_ref[t])
    for t in range(len(root_pos))
])
print("Body zmin BEFORE correction:", zmin_before.min(), "->", zmin_before.max())
print("Frames with body below ground:", np.sum(zmin_before < 0), "/", len(zmin_before))

corrected_z = correct_root_height_proper(root_pos, root_quat, q_ref)
root_pos[:, 2] = corrected_z

zmin_after = np.array([
    z_min_at_frame(root_pos[t], root_quat[t], q_ref[t])
    for t in range(len(root_pos))
])
print("\nBody zmin AFTER correction:", zmin_after.min(), "->", zmin_after.max())
print("Frames with body below ground:", np.sum(zmin_after < 0), "/", len(zmin_after))

# Create a dictionary containing every key from the original file
motion_dict = {k: motion[k] for k in motion.files}

# Replace root_pos with corrected version
motion_dict["root_pos"] = root_pos

output_path = "/Users/aayush/Documents/Pragya AI Research/Experiments/kalariSena/data/motions_retargeted_preprocessed/Urumi_Sword.npz"

np.savez_compressed(output_path, **motion_dict)

print(f"\nSaved corrected motion to:\n{output_path}")
print("Keys:", list(motion_dict.keys()))













