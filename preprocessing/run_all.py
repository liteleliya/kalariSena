"""
run_all.py
==========
ONE-FILE ORCHESTRATOR for the complete motion pipeline.

Flip the booleans in the TOGGLES block to choose which stages run:

    1) PRE-PROCESS    raw CSV  -> Steps 1-7 corrected CSVs + graphs
    2) CSV -> NPZ     final corrected CSV -> .npz (viewer format)
    3) VISUALIZE      play the .npz on the G1 in the meshcat 3D viewer

Run it with the project's virtual-env Python (from the repo root):

    .\\.venv\\Scripts\\python.exe preprocessing\\run_all.py
"""

import subprocess
import sys
from pathlib import Path

# Locations: this file lives in <repo>/preprocessing/ ; data lives at <repo>/.
HERE = Path(__file__).resolve().parent     # .../preprocessing
ROOT = HERE.parent                         # repo root

# ============================ TOGGLES ===============================
# Turn each stage on/off here. They run top-to-bottom.
RUN_PREPROCESS = True     # raw CSV -> Steps 1-7 corrected CSVs + PNG graphs
RUN_CSV_TO_NPZ = True     # final (Step 7) CSV -> .npz
RUN_VISUALIZE  = True      # open the meshcat 3D viewer on the final motion

SHOW_PLOTS  = False        # True = pop up matplotlib windows during pre-process
LOOP_VIEWER = True         # True = loop the motion in the 3D viewer
# ===================================================================

# ============================ CONFIG ===============================
INPUT_CSV = ROOT / "Data/Urumi_Sword_retarget_g1.csv"
URDF      = ROOT / "unitree_ros/robots/g1_description/g1_29dof.urdf"
FPS       = 30

# The pipeline's final output is the Step 7 (Savitzky-Golay) CSV.
FINAL_CSV = INPUT_CSV.with_name(INPUT_CSV.stem + "_step7.csv")
NPZ_OUT   = INPUT_CSV.with_name(INPUT_CSV.stem + "_step7.npz")
# ===================================================================

PY = sys.executable        # the same (venv) Python that runs this file


def run(script: str, *script_args):
    """Run one pipeline script (found next to this file) with the venv Python.

    The working directory is the repo root so that any relative output paths
    (e.g. ``outputs/``) land in the right place.
    """
    cmd = [PY, str(HERE / script), *[str(a) for a in script_args]]
    print("\n" + "=" * 70)
    print(">>> " + " ".join(cmd))
    print("=" * 70)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    if RUN_PREPROCESS:
        args = ["--csv", INPUT_CSV, "--urdf", URDF]
        if not SHOW_PLOTS:
            args.append("--no-show")
        run("pre_process.py", *args)

    if RUN_CSV_TO_NPZ:
        run("csv_to_npz.py", "--csv", FINAL_CSV, "--out", NPZ_OUT, "--fps", FPS)

    if RUN_VISUALIZE:
        args = ["--npz", NPZ_OUT, "--urdf", URDF]
        if LOOP_VIEWER:
            args.append("--loop")
        run("visualize_motion_meshcat.py", *args)

    print("\nAll selected stages finished.")


if __name__ == "__main__":
    main()
