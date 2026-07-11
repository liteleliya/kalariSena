"""
calibrate_thresholds.py
=======================
Pick the preprocessing thresholds FROM YOUR DATA instead of guessing.

For your motion file it computes the three distributions that the thresholds
actually act on, and suggests a value for each using Otsu's method (the classic
way to find the natural split between two clusters in a histogram):

    Step 1  tau               <- histogram of |raw vertical velocity|
    Step 2  contact_threshold <- histogram of lowest-foot clearance after Step 1
    Step 3  tau3              <- histogram of |vertical velocity| after Step 2

It prints a suggestion table and saves a 3-panel figure.

Run:
    .\\.venv\\Scripts\\python.exe calibrate_thresholds.py \
        --csv "Data/Urumi_Sword_retarget_g1.csv" \
        --urdf "unitree_ros/robots/g1_description/g1_29dof.urdf"
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yourdfpy

from pre_process import (
    load_data,
    build_joint_matrix,
    compute_z_offsets,
    root_height_drift_correction,
    minimum_body_height_constraint,
    CM_TO_M,
    FPS,
    GRAVITY,
)


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Find the 1-D threshold that best splits ``values`` into two clusters.

    This is Otsu's method: it picks the cutoff that maximizes the variance
    *between* the two groups (equivalently, minimizes the variance within them).
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(float)
    total = p.sum()
    if total == 0:
        return float(centers[0])
    p = p / total
    omega = np.cumsum(p)                 # class 0 probability up to each bin
    mu = np.cumsum(p * centers)          # class 0 cumulative mean
    mu_t = mu[-1]                        # global mean
    denom = omega * (1.0 - omega)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    if np.all(np.isnan(sigma_b2)):
        return float(np.median(v))
    return float(centers[int(np.nanargmax(sigma_b2))])


def pct_line(name, values):
    """Pretty percentile summary string."""
    q = np.percentile(values, [50, 75, 90, 99])
    return (f"  {name}: p50={q[0]:.4f}  p75={q[1]:.4f}  "
            f"p90={q[2]:.4f}  p99={q[3]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Data-driven threshold calibration.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--out-png", default="outputs/threshold_calibration.png")
    parser.add_argument("--fps", type=float, default=FPS,
                        help="Motion frame rate used when reporting m/frame thresholds in m/s.")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)

    # --- load + forward kinematics ----------------------------------------
    print("Loading CSV + URDF...")
    df, joint_cols = load_data(csv_path)
    urdf = yourdfpy.URDF.load(
        str(Path(args.urdf)), load_meshes=True, build_collision_scene_graph=False,
    )
    joints_rad = build_joint_matrix(df, urdf, joint_cols)
    P_raw = df["root_translateZ"].to_numpy(dtype=float) * CM_TO_M

    print("Computing z_min(q) via forward kinematics...")
    z_offset = compute_z_offsets(urdf, joints_rad)

    # --- run Steps 1-2 to get the signals each threshold acts on ----------
    P1, _ = root_height_drift_correction(P_raw, z_offset, fps=args.fps, gravity=GRAVITY)
    P2, _ = minimum_body_height_constraint(P1, z_offset)

    vel_raw = np.abs(np.diff(P_raw))        # Step 1 tau acts here
    clearance1 = P1 + z_offset              # Step 2 contact_threshold acts here
    vel_step2 = np.abs(np.diff(P2))         # Step 3 tau3 acts here

    # --- Otsu suggestions --------------------------------------------------
    tau_sugg = otsu_threshold(vel_raw)
    contact_sugg = otsu_threshold(clearance1)
    tau3_sugg = otsu_threshold(vel_step2)

    fps = args.fps
    print("\n================ THRESHOLD CALIBRATION ================")
    print(f"(units: velocity in m/frame; multiply by fps={fps:.0f} for m/s)\n")
    print(pct_line("|raw velocity|     ", vel_raw))
    print(pct_line("foot clearance(S1) ", clearance1))
    print(pct_line("|velocity| after S2", vel_step2))
    print("\n  Threshold            current     suggested (Otsu)")
    print(f"  Step1 tau            0.0015      {tau_sugg:.4f}   m/frame "
          f"(= {tau_sugg * fps:.3f} m/s)")
    print(f"  Step2 contact_thresh 0.0400      {contact_sugg:.4f}   m")
    print(f"  Step3 tau3           0.0030      {tau3_sugg:.4f}   m/frame "
          f"(= {tau3_sugg * fps:.3f} m/s)")
    print("======================================================\n")

    # --- figure ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    def hist_panel(ax, data, current, suggested, title, xlabel, xclip):
        ax.hist(np.clip(data, 0, xclip), bins=80, color="tab:blue", alpha=0.7)
        ax.axvline(current, color="tab:red", lw=2, ls="--",
                   label=f"current = {current:g}")
        ax.axvline(suggested, color="tab:green", lw=2,
                   label=f"suggested = {suggested:.4f}")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("frame count (log)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    hist_panel(axes[0], vel_raw, 0.0015, tau_sugg,
               "Step 1  tau  <-  |raw vertical velocity|",
               "|velocity| [m/frame]", np.percentile(vel_raw, 99))
    hist_panel(axes[1], clearance1, 0.04, contact_sugg,
               "Step 2  contact_threshold  <-  lowest-foot clearance after Step 1",
               "clearance above floor [m]", np.percentile(clearance1, 99))
    hist_panel(axes[2], vel_step2, 0.003, tau3_sugg,
               "Step 3  tau3  <-  |vertical velocity| after Step 2",
               "|velocity| [m/frame]", np.percentile(vel_step2, 99))

    fig.suptitle(f"Threshold calibration for {csv_path.name}", fontsize=13)
    fig.tight_layout()
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved calibration figure -> {out_path}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
