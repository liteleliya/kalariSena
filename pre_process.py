"""
pre_process.py
==============
Preprocessing pipeline for retargeted Unitree G1 motion data.

We build this up STEP BY STEP. Each step has its own function and its own
visualization so we can see exactly what changed.

STEP 1: Root Node Height Drift Correction with Parabolic Jump Reconstruction
        (Algorithm 1 from the paper).
STEP 2: Minimum Body Height Constraint  (Section 3.2.2, equations 3-5).
        Runs in continuation, on top of the Step 1 result.
STEP 3: Temporal Propagation with Velocity Thresholding (Section 3.2.3, eq 6-7).
        Runs in continuation, on top of the Step 2 result.
STEP 4: Jump Phase Handling via Local Extrema (Section 3.2.4, eq 8).
        Detects true jump segments and skips squat/stand look-alikes.
STEP 5: Parabolic Reconstruction of Airborne Root Motion (Section 3.2.5, eq 9-12).
        Rebuilds each jump arc as a gravity parabola.
STEP 6: Final Ground Penetration Correction (Section 3.2.6, eq 13).
        Last safeguard: no body part below the floor anywhere.
STEP 7: Post-Processing via Savitzky-Golay Smoothing (Section 3.3).
        Smooths every channel (root + joints) while preserving peaks.

See docs/DATASET.md, docs/ROOT_HEIGHT_DRIFT.md, docs/MIN_BODY_HEIGHT.md,
docs/TEMPORAL_PROPAGATION.md, docs/JUMP_DETECTION.md,
docs/PARABOLIC_RECONSTRUCTION.md, docs/FINAL_PENETRATION.md and
docs/SAVGOL_SMOOTHING.md for the beginner-friendly explanations.

Run:
    .\\.venv\\Scripts\\python.exe pre_process.py \
        --csv "Data/Urumi_Sword_retarget_g1.csv" \
        --urdf "unitree_ros/robots/g1_description/g1_29dof.urdf"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

import yourdfpy


# ---------------------------------------------------------------------------
# Constants describing the dataset (see DATASET.md)
# ---------------------------------------------------------------------------

FPS = 30.0            # frames per second of the capture
GRAVITY = 9.81        # m/s^2, used for the parabolic jump reconstruction
CM_TO_M = 0.01        # CSV translations are in centimeters -> meters
DEG_TO_RAD = np.pi / 180.0  # CSV angles are in degrees -> radians

# The 6 root columns (pelvis position + orientation).
ROOT_COLS = [
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path):
    """Read the CSV and return the dataframe plus the list of joint columns.

    The joint columns are everything that ends with '_dof'. They appear in the
    same order as the actuated joints inside the G1 URDF.
    """
    df = pd.read_csv(csv_path)
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    return df, joint_cols


def build_joint_matrix(df: pd.DataFrame, urdf: "yourdfpy.URDF", joint_cols):
    """Return a (T, 29) array of joint angles in RADIANS, ordered exactly like
    the URDF's actuated joints.

    The CSV columns are named '<joint>_dof'; the URDF joints are named
    '<joint>'. We line them up by name so ordering can never go wrong.
    """
    # Map URDF joint name -> CSV column name (append the '_dof' suffix).
    ordered_cols = []
    for joint_name in urdf.actuated_joint_names:
        csv_col = joint_name + "_dof"
        if csv_col not in joint_cols:
            raise KeyError(f"URDF joint '{joint_name}' has no CSV column '{csv_col}'")
        ordered_cols.append(csv_col)

    joints_deg = df[ordered_cols].to_numpy(dtype=float)
    return joints_deg * DEG_TO_RAD


# ---------------------------------------------------------------------------
# Forward kinematics helper:  z_min(q)
# ---------------------------------------------------------------------------

def compute_z_offsets(urdf: "yourdfpy.URDF", joints_rad: np.ndarray) -> np.ndarray:
    """For every frame, compute z_min(q): the height of the LOWEST point of the
    whole robot body, measured RELATIVE to the pelvis (pelvis at origin).

    This is a negative number (the feet are below the pelvis). With it we can
    place the foot exactly on the floor by setting  pelvis_height = -z_min(q).
    """
    T = joints_rad.shape[0]
    offsets = np.empty(T, dtype=float)
    for t in range(T):
        # Bend the virtual robot into this pose (pelvis stays at the origin).
        urdf.update_cfg(joints_rad[t])
        # scene.bounds is a 2x3 array: row 0 = (xmin, ymin, zmin).
        offsets[t] = urdf.scene.bounds[0][2]
        if (t + 1) % 200 == 0:
            print(f"  FK z_min: {t + 1}/{T} frames")
    return offsets


# ---------------------------------------------------------------------------
# STEP 1: Root Node Height Drift Correction  (Algorithm 1)
# ---------------------------------------------------------------------------

def root_height_drift_correction(
    P: np.ndarray,
    z_offset: np.ndarray,
    fps: float = FPS,
    gravity: float = GRAVITY,
    tau: float = 0.0015,
    peak_prominence: float = 0.02,
):
    """Correct the pelvis-height trajectory P (in METERS).

    Parameters
    ----------
    P            : (T,) raw pelvis height per frame [m].
    z_offset     : (T,) z_min(q) per frame [m] (lowest body point vs pelvis).
    tau          : velocity threshold [m/frame]; rises slower than this are
                   treated as drift and ignored.
    peak_prominence : how pronounced a peak/valley must be to count as an
                   apex/contact [m].

    Returns
    -------
    P_hat   : (T,) corrected pelvis height [m].
    info    : dict with contact/apex frame indices (for plotting).
    """
    T = len(P)

    # Grounded height for each frame = pelvis height that puts the foot on z=0.
    grounded = -z_offset                      # P̂ during contact

    # (Part A) velocity and the contact / apex frames -----------------------
    velocity = np.zeros(T)
    velocity[:-1] = P[1:] - P[:-1]            # Ṗ(t) = P(t+1) - P(t)

    minima, _ = find_peaks(-P, prominence=peak_prominence)   # contacts
    maxima, _ = find_peaks(P, prominence=peak_prominence)     # apexes
    contact_set = set(minima.tolist())

    # (Part B) rebuild the height frame by frame ----------------------------
    P_hat = np.empty(T)
    P_hat[0] = grounded[0]                    # ground the first frame

    for t in range(1, T):
        if t in contact_set:
            # Foot contact -> snap pelvis so the lowest point sits on the floor.
            P_hat[t] = grounded[t]
        else:
            # Carry previous height; add the previous rise ONLY if it was a real
            # upward motion (faster than tau). Otherwise hold -> removes drift.
            rise = velocity[t - 1] if velocity[t - 1] > tau else 0.0
            P_hat[t] = P_hat[t - 1] + rise

    # (Part C) replace each jump's descent with a gravity parabola ----------
    for t_s in maxima:
        # nearest contact strictly after this apex = the landing frame
        later = minima[minima > t_s]
        if len(later) == 0:
            continue
        t_e = int(later[0])
        N = t_e - t_s - 1                     # frames strictly between apex/landing
        if N <= 0:
            continue
        drop = P_hat[t_s] - P_hat[t_e]        # how far it falls [m]
        if drop <= 0:
            continue                          # not a real fall, skip
        T_flight = np.sqrt(2.0 * drop / gravity)   # airtime [s]
        for k in range(1, N + 1):
            dt = (k / (N + 1)) * T_flight     # time since apex [s]
            P_hat[t_s + k] = P_hat[t_s] - 0.5 * gravity * dt * dt

    # (Part D) safety net: never let the foot go below the floor ------------
    foot_world = P_hat + z_offset
    penetrating = foot_world < 0.0
    P_hat[penetrating] = grounded[penetrating]

    info = {
        "contacts": minima,
        "apexes": maxima,
        "grounded": grounded,
    }
    return P_hat, info


# ---------------------------------------------------------------------------
# STEP 2: Minimum Body Height Constraint  (Section 3.2.2, equations 3-5)
# ---------------------------------------------------------------------------

def minimum_body_height_constraint(
    P: np.ndarray,
    z_offset: np.ndarray,
    contact_threshold: float = 0.04,
):
    """Step 2: enforce the minimum body height constraint on a height trajectory.

    Runs *after* Step 1, on the Step-1 corrected pelvis height ``P``.

    Guarantee produced (paper eq. 5):
        z_min(q_hat) == 0   during ground-contact phases   (exact contact)
        z_min(q_hat) >= 0   during airborne phases         (clearance, no penetration)

    Parameters
    ----------
    P                 : (T,) Step-1 corrected pelvis height [m].
    z_offset          : (T,) z_min relative to the pelvis [m] (lowest body part).
                        Reused from Step 1 because the joints are unchanged.
    contact_threshold : eps [m]; a frame is a ground-contact phase when the
                        lowest body part is within this band of the floor.

    Returns
    -------
    P_hat : (T,) Step-2 corrected pelvis height [m].
    info  : dict with the contact mask and the before/after lowest-part curves.
    """
    # Eq. (3): height of the lowest body part in the world (after Step 1).
    world_min_before = P + z_offset

    # Pelvis height that places the lowest body part exactly on z=0 (eq. 4 form).
    grounded = -z_offset

    # Classify every frame as contact (near floor) or airborne (clear of floor).
    contact_mask = world_min_before <= contact_threshold
    airborne_mask = ~contact_mask

    P_hat = P.copy()

    # Contact phases -> Eq. (4): align lowest body part to the floor (z_min = 0).
    P_hat[contact_mask] = grounded[contact_mask]

    # Airborne phases -> Eq. (5): only lift if it would penetrate (z_min < 0).
    penetrating = airborne_mask & (world_min_before < 0.0)
    P_hat[penetrating] = grounded[penetrating]

    info = {
        "contact_mask": contact_mask,
        "world_min_before": world_min_before,
        "world_min_after": P_hat + z_offset,
    }
    return P_hat, info


# ---------------------------------------------------------------------------
# STEP 3: Temporal Propagation with Velocity Thresholding (Section 3.2.3, eq 6-7)
# ---------------------------------------------------------------------------

def temporal_propagation_velocity_threshold(
    P: np.ndarray,
    contact_mask: np.ndarray,
    tau: float = 0.003,
):
    """Step 3: temporally smooth the height by gating out sub-threshold jitter.

    Runs *after* Step 2, on the Step-2 corrected pelvis height ``P``.

    Logic (paper eq. 6-7):
        - eq. 6: velocity by forward difference  v(t) = P(t+1) - P(t)
        - eq. 7: propagate from the previous *corrected* frame, adding this
          frame's motion only when it is bigger than tau; otherwise hold.
        - reliable ground-contact frames stay anchored at their Step-2 value.

    The paper's test is one-sided ( v > tau ), aimed at upward drift inside
    Algorithm 1. As a standalone smoothing pass we gate on the magnitude
    ( |v| > tau ) so genuine downward motion is preserved too.

    Parameters
    ----------
    P            : (T,) Step-2 corrected pelvis height [m].
    contact_mask : (T,) bool; True on reliable ground-contact frames (anchors).
    tau          : velocity threshold [m/frame]; |v| <= tau is treated as jitter.

    Returns
    -------
    P_hat : (T,) Step-3 corrected pelvis height [m].
    info  : dict with velocity, the +/-tau band and the suppressed-frame mask.
    """
    T = len(P)

    # eq. (6): first-order forward difference velocity.
    velocity = np.zeros(T)
    velocity[:-1] = P[1:] - P[:-1]

    P_hat = P.copy()
    suppressed = np.zeros(T, dtype=bool)   # frames whose jitter we gated to 0

    for t in range(1, T):
        if contact_mask[t]:
            # Reliable contact -> keep the Step-2 anchor (feet on the floor).
            P_hat[t] = P[t]
            continue
        v = velocity[t - 1]
        if abs(v) > tau:
            P_hat[t] = P_hat[t - 1] + v        # real motion -> propagate it
        else:
            P_hat[t] = P_hat[t - 1]            # jitter -> hold (eq. 7 '0' case)
            suppressed[t] = True

    info = {
        "velocity": velocity,
        "tau": tau,
        "suppressed": suppressed,
        "contact_mask": contact_mask,
    }
    return P_hat, info


# ---------------------------------------------------------------------------
# STEP 4: Jump Phase Handling via Local Extrema  (Section 3.2.4, eq 8)
# ---------------------------------------------------------------------------

def detect_jump_segments(
    P: np.ndarray,
    z_offset: np.ndarray,
    airborne_threshold: float = 0.06,
    min_air_frames: int = 2,
    peak_prominence: float = 0.02,
):
    """Step 4: find the TRUE jump segments and skip squat/stand look-alikes.

    A frame is 'airborne' when the lowest body part is clearly off the floor
    (clearance > airborne_threshold). Contiguous airborne runs are real jumps;
    everything else (squat/stand, where the feet stay down) is skipped -- this
    is our objective version of the paper's manually defined skip set.

    Parameters
    ----------
    P                  : (T,) Step-3 corrected pelvis height [m].
    z_offset           : (T,) z_min relative to the pelvis [m].
    airborne_threshold : clearance [m] above which the body counts as airborne.
    min_air_frames     : ignore airborne blips shorter than this many frames.
    peak_prominence    : prominence [m] for the displayed local extrema.

    Returns
    -------
    segments : list of dicts {t_s, apex, t_e} (take-off, apex, landing frames).
    info     : dict with clearance, airborne mask, extrema and the skip set.
    """
    T = len(P)
    clearance = P + z_offset
    airborne = clearance > airborne_threshold

    # Local extrema of the height curve (eq. 8 context: P'=0 with P''<0 / P''>0).
    maxima, _ = find_peaks(P, prominence=peak_prominence)
    minima, _ = find_peaks(-P, prominence=peak_prominence)

    # Walk the airborne mask and collect contiguous runs -> jump segments.
    segments = []
    i = 0
    while i < T:
        if not airborne[i]:
            i += 1
            continue
        start = i
        while i < T and airborne[i]:
            i += 1
        end = i - 1                          # inclusive end of the airborne run
        if end - start + 1 < min_air_frames:
            continue
        t_s = max(start - 1, 0)              # last contact before take-off
        t_e = min(end + 1, T - 1)            # first contact after landing
        apex = start + int(np.argmax(P[start:end + 1]))
        segments.append({"t_s": t_s, "apex": apex, "t_e": t_e,
                         "run": (start, end)})

    # Skip set = extrema that do NOT fall inside any jump segment.
    in_jump = np.zeros(T, dtype=bool)
    for s in segments:
        in_jump[s["t_s"]:s["t_e"] + 1] = True
    skipped_extrema = np.array(
        [e for e in np.concatenate([maxima, minima]) if not in_jump[e]], dtype=int
    )

    info = {
        "clearance": clearance,
        "airborne": airborne,
        "airborne_threshold": airborne_threshold,
        "maxima": maxima,
        "minima": minima,
        "skipped_extrema": skipped_extrema,
        "segments": segments,
    }
    return segments, info


# ---------------------------------------------------------------------------
# STEP 5: Parabolic Reconstruction of Airborne Root Motion (Sec 3.2.5, eq 9-12)
# ---------------------------------------------------------------------------

def parabolic_airborne_reconstruction(
    P: np.ndarray,
    segments,
    gravity: float = GRAVITY,
):
    """Step 5: replace each jump arc with a physics parabola (eq. 9-12).

    For each segment, the apex height y0 and landing height y1 are kept fixed
    (eq. 9). The descent apex->landing is the free-fall curve y0 - 1/2 g t^2
    (eq. 10) with airtime T = sqrt(2(y0-y1)/g) (eq. 11), sampled uniformly in
    time (eq. 12). The ascent take-off->apex is the symmetric mirror.

    Parameters
    ----------
    P        : (T,) Step-3 corrected pelvis height [m].
    segments : list of {t_s, apex, t_e} from detect_jump_segments.
    gravity  : g [m/s^2].

    Returns
    -------
    P_hat : (T,) pelvis height with jump arcs reconstructed [m].
    info  : dict with the reconstructed-frame mask (for plotting).
    """
    P_hat = P.copy()
    reconstructed = np.zeros(len(P), dtype=bool)

    for seg in segments:
        t_s, apex, t_e = seg["t_s"], seg["apex"], seg["t_e"]
        y_apex = P[apex]

        # --- descent: apex -> landing (eq. 9-12) --------------------------
        y1 = P[t_e]
        drop = y_apex - y1
        Nd = t_e - apex - 1
        if drop > 0 and Nd > 0:
            T_fall = np.sqrt(2.0 * drop / gravity)     # eq. 11
            for k in range(1, Nd + 1):
                dt = (k / (Nd + 1)) * T_fall           # eq. 12
                P_hat[apex + k] = y_apex - 0.5 * gravity * dt * dt   # eq. 10
                reconstructed[apex + k] = True

        # --- ascent: take-off -> apex (symmetric mirror) ------------------
        y0 = P[t_s]
        rise = y_apex - y0
        Na = apex - t_s - 1
        if rise > 0 and Na > 0:
            T_rise = np.sqrt(2.0 * rise / gravity)
            for k in range(1, Na + 1):
                # time measured backward from the apex
                dt = ((Na + 1 - k) / (Na + 1)) * T_rise
                P_hat[t_s + k] = y_apex - 0.5 * gravity * dt * dt
                reconstructed[t_s + k] = True

    info = {"reconstructed": reconstructed, "segments": segments}
    return P_hat, info


# ---------------------------------------------------------------------------
# STEP 6: Final Ground Penetration Correction  (Section 3.2.6, eq 13)
# ---------------------------------------------------------------------------

def final_ground_penetration_correction(
    P: np.ndarray,
    z_offset: np.ndarray,
):
    """Step 6: clamp any remaining ground penetration (eq. 13).

    For every frame where the lowest body part is below the floor
    (clearance = P + z_offset < 0), lift the root to the grounded height so the
    lowest part sits exactly on z = 0. Frames already on/above the floor are
    left untouched.

    Returns
    -------
    P_hat : (T,) pelvis height with all penetration removed [m].
    info  : dict with before/after clearance and the penetrating-frame mask.
    """
    clearance_before = P + z_offset
    penetrating = clearance_before < 0.0

    P_hat = P.copy()
    P_hat[penetrating] = -z_offset[penetrating]      # grounded height

    info = {
        "clearance_before": clearance_before,
        "clearance_after": P_hat + z_offset,
        "penetrating": penetrating,
    }
    return P_hat, info


# ---------------------------------------------------------------------------
# STEP 7: Post-Processing via Savitzky-Golay Smoothing  (Section 3.3)
# ---------------------------------------------------------------------------

def savitzky_golay_smoothing(
    df: pd.DataFrame,
    joint_cols,
    window: int = None,
    polyorder: int = 3,
):
    """Step 7: smooth every motion channel with a Savitzky-Golay filter (Sec 3.3).

    Each channel (root position, root rotation, and all joint angles) is filtered
    independently along time. SG fits a local polynomial of order ``polyorder``
    over a sliding window of ``window`` samples, which removes jitter while
    preserving peaks (jump apexes / contact dips).

    Parameters
    ----------
    df         : the corrected pose dataframe (root_translateZ already set to the
                 Step-6 height, in centimeters).
    joint_cols : the list of joint columns ('*_dof').
    window     : odd window length. Default ~ 1/10 of the frames (paper rule).
    polyorder  : polynomial order of the local fit.

    Returns
    -------
    df_out : a copy of df with all motion channels smoothed.
    info   : dict with the window/polyorder actually used and the channel list.
    """
    T = len(df)
    if window is None:
        window = T // 10
    if window % 2 == 0:                      # window must be odd
        window += 1
    window = max(window, polyorder + 2 + (polyorder % 2 == 0))   # > polyorder, odd-safe
    if window % 2 == 0:
        window += 1
    window = min(window, T if T % 2 == 1 else T - 1)            # cannot exceed T

    channels = ROOT_COLS + list(joint_cols)
    df_out = df.copy()
    for c in channels:
        df_out[c] = savgol_filter(df[c].to_numpy(dtype=float), window, polyorder)

    info = {"window": window, "polyorder": polyorder, "channels": channels}
    return df_out, info


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_step1(P_raw, P_hat, z_offset, info, fps, out_path: Path, show: bool):
    """Two-panel figure proving the drift was removed."""
    T = len(P_raw)
    t_axis = np.arange(T) / fps

    foot_before = P_raw + z_offset
    foot_after = P_hat + z_offset

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1: pelvis height -------------------------------------------------
    ax1.plot(t_axis, P_raw, color="tab:red", alpha=0.6, label="raw pelvis height")
    ax1.plot(t_axis, P_hat, color="tab:blue", lw=2, label="corrected pelvis height")
    ax1.scatter(info["contacts"] / fps, P_hat[info["contacts"]],
                color="green", zorder=5, s=25, label="contacts (minima)")
    ax1.scatter(info["apexes"] / fps, P_hat[info["apexes"]],
                color="orange", zorder=5, s=25, label="apexes (maxima)")
    ax1.set_ylabel("pelvis height [m]")
    ax1.set_title("Step 1 — Root Node Height Drift Correction")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: lowest-foot world height -------------------------------------
    ax2.axhline(0.0, color="black", lw=1.2, label="floor (z = 0)")
    ax2.plot(t_axis, foot_before, color="tab:red", alpha=0.6,
             label="lowest foot BEFORE")
    ax2.plot(t_axis, foot_after, color="tab:blue", lw=2,
             label="lowest foot AFTER")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("lowest body point [m]")
    ax2.set_title("Ground contact: before (float/penetration) vs after (on floor)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved visualization -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_step2(P1, P2, z_offset, info, fps, out_path: Path, show: bool):
    """Two-panel figure showing the minimum body height constraint (eq. 5)."""
    T = len(P1)
    t_axis = np.arange(T) / fps

    world_min_before = info["world_min_before"]   # lowest body part after Step 1
    world_min_after = info["world_min_after"]     # lowest body part after Step 2
    contact_mask = info["contact_mask"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1: lowest body part height, before vs after, with contact shading.
    ax1.fill_between(t_axis, 0, 1, where=contact_mask, transform=ax1.get_xaxis_transform(),
                     color="green", alpha=0.08, label="contact phase")
    ax1.axhline(0.0, color="black", lw=1.2, label="floor (z = 0)")
    ax1.plot(t_axis, world_min_before, color="tab:blue", alpha=0.6,
             label="lowest body part after Step 1")
    ax1.plot(t_axis, world_min_after, color="tab:green", lw=2,
             label="lowest body part after Step 2")
    ax1.set_ylabel("lowest body point [m]")
    ax1.set_title("Step 2 — Minimum Body Height Constraint  (=0 contact, >=0 airborne)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: zoom on the floor band to reveal the Step 2 effect.
    # During contact phases Step 1 leaves a few-cm wiggle; Step 2 flattens it
    # to exactly 0. The jumps are clipped away by the zoomed y-range.
    ax2.fill_between(t_axis, 0, 1, where=contact_mask, transform=ax2.get_xaxis_transform(),
                     color="green", alpha=0.08, label="contact phase")
    ax2.axhline(0.0, color="black", lw=1.2, label="floor (z = 0)")
    ax2.plot(t_axis, world_min_before, color="tab:blue", alpha=0.7,
             label="after Step 1 (wiggles above floor)")
    ax2.plot(t_axis, world_min_after, color="tab:green", lw=2,
             label="after Step 2 (flat on floor in contact)")
    ax2.set_ylim(-0.01, 0.06)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("lowest body point [m]")
    ax2.set_title("Zoom near the floor: Step 2 pins contact phases to exactly 0")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved visualization -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_step3(P2, P3, info, fps, out_path: Path, show: bool):
    """Two-panel figure showing velocity thresholding (eq. 6-7)."""
    T = len(P2)
    t_axis = np.arange(T) / fps

    velocity = info["velocity"]
    tau = info["tau"]
    suppressed = info["suppressed"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1: velocity with the +/-tau gate band (the mechanism).
    # A frame t is gated based on the velocity entering it, v(t-1); plot those
    # tested samples so every suppressed marker sits inside the band.
    gated_vel_idx = np.clip(np.where(suppressed)[0] - 1, 0, T - 1)
    ax1.axhspan(-tau, tau, color="grey", alpha=0.25, label=f"jitter band |v| <= {tau}")
    ax1.axhline(0.0, color="black", lw=0.8)
    ax1.plot(t_axis, velocity, color="tab:purple", lw=1.0, label="vertical velocity v(t)")
    ax1.scatter(t_axis[gated_vel_idx], velocity[gated_vel_idx], color="red", s=10,
                zorder=5, label="suppressed (gated to 0)")
    ax1.set_ylabel("velocity [m/frame]")
    ax1.set_ylim(-0.05, 0.05)
    ax1.set_title("Step 3 — Temporal Propagation with Velocity Thresholding")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: pelvis height before vs after (the effect).
    ax2.plot(t_axis, P2, color="tab:blue", alpha=0.6, label="pelvis height after Step 2")
    ax2.plot(t_axis, P3, color="tab:orange", lw=2, label="pelvis height after Step 3")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("pelvis height [m]")
    ax2.set_title("Pelvis height: Step 2 vs Step 3 (de-jittered, temporally smooth)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved visualization -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_step4(P, info, fps, out_path: Path, show: bool):
    """Two-panel figure: detected jump segments vs skipped squat/stand extrema."""
    T = len(P)
    t_axis = np.arange(T) / fps

    clearance = info["clearance"]
    thr = info["airborne_threshold"]
    maxima = info["maxima"]
    minima = info["minima"]
    skipped = info["skipped_extrema"]
    segments = info["segments"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1: root height with extrema and shaded jump segments.
    for j, s in enumerate(segments):
        ax1.axvspan(s["t_s"] / fps, s["t_e"] / fps, color="orange", alpha=0.12,
                    label="jump segment" if j == 0 else None)
    ax1.plot(t_axis, P, color="tab:blue", lw=1.5, label="root height (after Step 3)")
    ax1.scatter(maxima / fps, P[maxima], marker="^", color="tab:orange", s=40,
                zorder=5, label="local maxima (apex)")
    ax1.scatter(minima / fps, P[minima], marker="o", color="tab:green", s=25,
                zorder=5, label="local minima (landing)")
    if len(skipped) > 0:
        ax1.scatter(skipped / fps, P[skipped], marker="x", color="grey", s=45,
                    zorder=6, label="skipped (squat/stand)")
    ax1.set_ylabel("root height [m]")
    ax1.set_title(f"Step 4 — Jump Phase Detection  ({len(segments)} true jump segments)")
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: foot clearance with the airborne threshold -> justifies detection.
    for s in segments:
        ax2.axvspan(s["t_s"] / fps, s["t_e"] / fps, color="orange", alpha=0.12)
    ax2.axhline(0.0, color="black", lw=1.0, label="floor (z = 0)")
    ax2.axhline(thr, color="tab:red", lw=1.5, ls="--",
                label=f"airborne threshold = {thr} m")
    ax2.plot(t_axis, clearance, color="tab:purple", lw=1.2, label="lowest-foot clearance")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("clearance [m]")
    ax2.set_title("Why: clearance rises above the threshold only during real jumps")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved visualization -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_step5(P_before, P_after, info, fps, out_path: Path, show: bool):
    """Two-panel figure: jump arcs replaced by gravity parabolas (+ a zoom)."""
    T = len(P_before)
    t_axis = np.arange(T) / fps
    segments = info["segments"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

    # Panel 1: full height, before vs after, shaded jump segments.
    for j, s in enumerate(segments):
        ax1.axvspan(s["t_s"] / fps, s["t_e"] / fps, color="orange", alpha=0.12,
                    label="jump segment" if j == 0 else None)
    ax1.plot(t_axis, P_before, color="tab:blue", alpha=0.6, label="before Step 5")
    ax1.plot(t_axis, P_after, color="tab:red", lw=1.8, label="after Step 5 (parabolic)")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("root height [m]")
    ax1.set_title("Step 5 — Parabolic Reconstruction of Airborne Root Motion")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: zoom on the tallest jump (largest apex-above-endpoints arc).
    if segments:
        def arc_height(s):
            return P_before[s["apex"]] - max(P_before[s["t_s"]], P_before[s["t_e"]])
        s = max(segments, key=arc_height)
        a, b = s["t_s"], s["t_e"]
        pad = max(2, (b - a) // 3)
        lo, hi = max(0, a - pad), min(T - 1, b + pad)
        seg_t = np.arange(lo, hi + 1) / fps
        ax2.axvspan(a / fps, b / fps, color="orange", alpha=0.12, label="jump segment")
        ax2.plot(seg_t, P_before[lo:hi + 1], color="tab:blue", alpha=0.6,
                 marker="o", ms=3, label="before (noisy)")
        ax2.plot(seg_t, P_after[lo:hi + 1], color="tab:red", lw=2,
                 marker="o", ms=3, label="after (parabola)")
        ax2.scatter([s["apex"] / fps], [P_after[s["apex"]]], color="black", s=50,
                    zorder=6, label="apex")
        ax2.set_title("Zoom on the tallest jump: noisy arc -> clean gravity parabola")
    else:
        ax2.text(0.5, 0.5, "No airborne jump segments detected",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Zoom (no jumps found)")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("root height [m]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved visualization -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_step6(info, fps, out_path: Path, show: bool):
    """One-panel figure: final ground-penetration clamp (eq. 13)."""
    clearance_before = info["clearance_before"]
    clearance_after = info["clearance_after"]
    penetrating = info["penetrating"]
    T = len(clearance_before)
    t_axis = np.arange(T) / fps

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(0.0, color="black", lw=1.2, label="floor (z = 0)")
    ax.plot(t_axis, clearance_before, color="tab:blue", alpha=0.6,
            label="lowest-foot clearance before")
    ax.plot(t_axis, clearance_after, color="tab:green", lw=1.8,
            label="lowest-foot clearance after")
    if np.any(penetrating):
        ax.scatter(t_axis[penetrating], clearance_before[penetrating],
                   color="red", s=20, zorder=5,
                   label=f"penetrating frames ({int(np.sum(penetrating))})")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("clearance [m]")
    ax.set_title("Step 6 — Final Ground Penetration Correction  (z_min >= 0 everywhere)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved visualization -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_step7(df_before, df_after, joint_cols, fps, info, out_path: Path, show: bool):
    """Two-panel figure: SG smoothing on the root height and on a joint angle."""
    T = len(df_before)
    t_axis = np.arange(T) / fps
    window = info["window"]
    polyorder = info["polyorder"]

    z_before = df_before["root_translateZ"].to_numpy(dtype=float) * CM_TO_M
    z_after = df_after["root_translateZ"].to_numpy(dtype=float) * CM_TO_M

    joint = "left_knee_joint_dof" if "left_knee_joint_dof" in joint_cols else joint_cols[0]
    j_before = df_before[joint].to_numpy(dtype=float)
    j_after = df_after[joint].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

    # Panel 1: root height before vs after (full).
    ax1.plot(t_axis, z_before, color="tab:blue", alpha=0.6, label="before SG")
    ax1.plot(t_axis, z_after, color="tab:orange", lw=1.8, label="after SG")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("root height [m]")
    ax1.set_title(f"Step 7 — Savitzky-Golay Smoothing  (window={window}, poly={polyorder})")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: a joint angle, zoomed, to show per-joint de-jitter.
    n = min(T, int(8 * fps))                 # first ~8 seconds
    ax2.plot(t_axis[:n], j_before[:n], color="tab:blue", alpha=0.5,
             marker="o", ms=2, label=f"{joint} before")
    ax2.plot(t_axis[:n], j_after[:n], color="tab:red", lw=2,
             label=f"{joint} after")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("joint angle [deg]")
    ax2.set_title("Zoom on a joint: jitter removed, swing peaks preserved")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved visualization -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="G1 motion preprocessing (Steps 1-5).")
    parser.add_argument("--csv", required=True, help="Input retargeted CSV.")
    parser.add_argument("--urdf", required=True, help="G1 29-DOF URDF path.")
    parser.add_argument("--out-csv", default=None,
                        help="Where to write the Step 1 corrected CSV "
                             "(default: <input>_step1.csv).")
    parser.add_argument("--out-csv2", default=None,
                        help="Where to write the Step 2 corrected CSV "
                             "(default: <input>_step2.csv).")
    parser.add_argument("--out-csv3", default=None,
                        help="Where to write the Step 3 corrected CSV "
                             "(default: <input>_step3.csv).")
    parser.add_argument("--out-csv5", default=None,
                        help="Where to write the final Step 5 corrected CSV "
                             "(default: <input>_step5.csv).")
    parser.add_argument("--out-csv7", default=None,
                        help="Where to write the FINAL Step 7 smoothed CSV "
                             "(default: <input>_step7.csv).")
    parser.add_argument("--out-png", default="outputs/step1_root_height_drift.png",
                        help="Where to write the Step 1 visualization PNG.")
    parser.add_argument("--out-png2", default="outputs/step2_min_body_height.png",
                        help="Where to write the Step 2 visualization PNG.")
    parser.add_argument("--out-png3", default="outputs/step3_temporal_propagation.png",
                        help="Where to write the Step 3 visualization PNG.")
    parser.add_argument("--out-png4", default="outputs/step4_jump_detection.png",
                        help="Where to write the Step 4 visualization PNG.")
    parser.add_argument("--out-png5", default="outputs/step5_parabolic_reconstruction.png",
                        help="Where to write the Step 5 visualization PNG.")
    parser.add_argument("--out-png6", default="outputs/step6_final_penetration.png",
                        help="Where to write the Step 6 visualization PNG.")
    parser.add_argument("--out-png7", default="outputs/step7_savgol_smoothing.png",
                        help="Where to write the Step 7 visualization PNG.")
    parser.add_argument("--tau", type=float, default=0.0015,
                        help="Velocity threshold [m/frame] for drift removal.")
    parser.add_argument("--tau3", type=float, default=0.003,
                        help="Step 3 velocity threshold [m/frame] for jitter gating.")
    parser.add_argument("--contact-threshold", type=float, default=0.04,
                        help="Step 2 contact band [m]: lowest part within this of "
                             "the floor counts as a ground-contact phase.")
    parser.add_argument("--airborne-threshold", type=float, default=0.06,
                        help="Step 4 clearance [m] above which a frame is airborne "
                             "(a real jump rather than a squat/stand).")
    parser.add_argument("--sg-window", type=int, default=31,
                        help="Step 7 Savitzky-Golay window (odd). Paper rule is "
                             "~frames/10; default 31 keeps contacts crisp on long clips.")
    parser.add_argument("--sg-polyorder", type=int, default=3,
                        help="Step 7 Savitzky-Golay polynomial order.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open the plot window (just save the PNG).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    urdf_path = Path(args.urdf)

    # 1. Load data ----------------------------------------------------------
    print("Loading CSV...")
    df, joint_cols = load_data(csv_path)
    print(f"  frames: {len(df)},  joint columns: {len(joint_cols)}")

    # 2. Load the robot model ----------------------------------------------
    print("Loading G1 URDF...")
    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        load_meshes=True,
        build_collision_scene_graph=False,
    )
    assert len(urdf.actuated_joint_names) == len(joint_cols), \
        "URDF actuated joints != CSV joint columns"

    # 3. Convert units ------------------------------------------------------
    joints_rad = build_joint_matrix(df, urdf, joint_cols)
    P_raw = df["root_translateZ"].to_numpy(dtype=float) * CM_TO_M   # pelvis height [m]

    # 4. Forward kinematics: z_min(q) per frame ----------------------------
    print("Computing z_min(q) via forward kinematics...")
    z_offset = compute_z_offsets(urdf, joints_rad)

    # 5. STEP 1: correct the height ----------------------------------------
    print("Running Root Node Height Drift Correction...")
    P_hat, info = root_height_drift_correction(
        P_raw, z_offset, fps=FPS, gravity=GRAVITY, tau=args.tau,
    )

    # 6. Report -------------------------------------------------------------
    pen_before = np.sum((P_raw + z_offset) < -1e-4)
    pen_after = np.sum((P_hat + z_offset) < -1e-4)
    print(f"  contacts detected : {len(info['contacts'])}")
    print(f"  apexes detected   : {len(info['apexes'])}")
    print(f"  ground-penetrating frames  before: {pen_before}  after: {pen_after}")

    # 7. Write Step 1 corrected CSV (height back to centimeters) ------------
    out_csv = Path(args.out_csv) if args.out_csv else \
        csv_path.with_name(csv_path.stem + "_step1.csv")
    df_out = df.copy()
    df_out["root_translateZ"] = P_hat / CM_TO_M
    df_out.to_csv(out_csv, index=False)
    print(f"Saved Step 1 corrected CSV -> {out_csv}")

    # 8. Visualize Step 1 ---------------------------------------------------
    visualize_step1(P_raw, P_hat, z_offset, info, FPS,
                    Path(args.out_png), show=not args.no_show)

    # 9. STEP 2: minimum body height constraint (on top of Step 1) ---------
    print("Running Minimum Body Height Constraint...")
    P_hat2, info2 = minimum_body_height_constraint(
        P_hat, z_offset, contact_threshold=args.contact_threshold,
    )
    n_contact = int(np.sum(info2["contact_mask"]))
    min_clear_before = float(np.min(info2["world_min_before"]))
    min_clear_after = float(np.min(info2["world_min_after"]))
    print(f"  contact-phase frames : {n_contact}/{len(P_hat2)}")
    print(f"  min lowest-part height  after Step 1: {min_clear_before:+.4f} m")
    print(f"  min lowest-part height  after Step 2: {min_clear_after:+.4f} m (should be >= 0)")

    # 10. Write Step 2 corrected CSV ---------------------------------------
    out_csv2 = Path(args.out_csv2) if args.out_csv2 else \
        csv_path.with_name(csv_path.stem + "_step2.csv")
    df_out2 = df.copy()
    df_out2["root_translateZ"] = P_hat2 / CM_TO_M
    df_out2.to_csv(out_csv2, index=False)
    print(f"Saved Step 2 corrected CSV -> {out_csv2}")

    # 11. Visualize Step 2 --------------------------------------------------
    visualize_step2(P_hat, P_hat2, z_offset, info2, FPS,
                    Path(args.out_png2), show=not args.no_show)

    # 12. STEP 3: temporal propagation with velocity thresholding ----------
    print("Running Temporal Propagation with Velocity Thresholding...")
    P_hat3, info3 = temporal_propagation_velocity_threshold(
        P_hat2, info2["contact_mask"], tau=args.tau3,
    )
    n_suppressed = int(np.sum(info3["suppressed"]))
    print(f"  jitter frames suppressed : {n_suppressed}/{len(P_hat3)}")

    # 13. Write Step 3 corrected CSV ---------------------------------------
    out_csv3 = Path(args.out_csv3) if args.out_csv3 else \
        csv_path.with_name(csv_path.stem + "_step3.csv")
    df_out3 = df.copy()
    df_out3["root_translateZ"] = P_hat3 / CM_TO_M
    df_out3.to_csv(out_csv3, index=False)
    print(f"Saved Step 3 corrected CSV -> {out_csv3}")

    # 14. Visualize Step 3 --------------------------------------------------
    visualize_step3(P_hat2, P_hat3, info3, FPS,
                    Path(args.out_png3), show=not args.no_show)

    # 15. STEP 4: detect true jump segments (skip squats/stands) -----------
    print("Running Jump Phase Detection...")
    segments, info4 = detect_jump_segments(
        P_hat3, z_offset, airborne_threshold=args.airborne_threshold,
    )
    print(f"  true jump segments : {len(segments)}")
    for s in segments:
        print(f"    take-off f{s['t_s']}  apex f{s['apex']}  landing f{s['t_e']}")
    visualize_step4(P_hat3, info4, FPS, Path(args.out_png4), show=not args.no_show)

    # 16. STEP 5: parabolic reconstruction of the jump arcs ----------------
    print("Running Parabolic Reconstruction...")
    P_hat5, info5 = parabolic_airborne_reconstruction(P_hat3, segments, gravity=GRAVITY)
    n_recon = int(np.sum(info5["reconstructed"]))
    print(f"  frames reconstructed as parabola : {n_recon}/{len(P_hat5)}")

    # 17. Write final (Step 5) corrected CSV -------------------------------
    out_csv5 = Path(args.out_csv5) if args.out_csv5 else \
        csv_path.with_name(csv_path.stem + "_step5.csv")
    df_out5 = df.copy()
    df_out5["root_translateZ"] = P_hat5 / CM_TO_M
    df_out5.to_csv(out_csv5, index=False)
    print(f"Saved FINAL (Step 5) corrected CSV -> {out_csv5}")

    # 18. Visualize Step 5 --------------------------------------------------
    visualize_step5(P_hat3, P_hat5, info5, FPS,
                    Path(args.out_png5), show=not args.no_show)

    # 19. STEP 6: final ground-penetration clamp ---------------------------
    print("Running Final Ground Penetration Correction...")
    P_hat6, info6 = final_ground_penetration_correction(P_hat5, z_offset)
    print(f"  penetrating frames fixed : {int(np.sum(info6['penetrating']))}")
    print(f"  min clearance after Step 6: {float(np.min(info6['clearance_after'])):+.4f} m")
    visualize_step6(info6, FPS, Path(args.out_png6), show=not args.no_show)

    # 20. STEP 7: Savitzky-Golay smoothing of every channel ----------------
    print("Running Savitzky-Golay Smoothing...")
    df_step6 = df.copy()
    df_step6["root_translateZ"] = P_hat6 / CM_TO_M          # final corrected height
    df_step7, info7 = savitzky_golay_smoothing(
        df_step6, joint_cols, window=args.sg_window, polyorder=args.sg_polyorder,
    )
    print(f"  SG window={info7['window']}  polyorder={info7['polyorder']}  "
          f"channels={len(info7['channels'])}")

    # 21. Write FINAL (Step 7) smoothed CSV --------------------------------
    out_csv7 = Path(args.out_csv7) if args.out_csv7 else \
        csv_path.with_name(csv_path.stem + "_step7.csv")
    df_step7.to_csv(out_csv7, index=False)
    print(f"Saved FINAL (Step 7) smoothed CSV -> {out_csv7}")

    # 22. Visualize Step 7 --------------------------------------------------
    visualize_step7(df_step6, df_step7, joint_cols, FPS, info7,
                    Path(args.out_png7), show=not args.no_show)


if __name__ == "__main__":
    main()
