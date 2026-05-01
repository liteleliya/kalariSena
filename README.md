# KalariSena Pipeline (Teammate Install Guide)

This repo lets you run a full pipeline from an input MP4 to a Unitree G1 retarget
and then prepare training data for the physics-aware stages (B–F).

The code is runnable end-to-end **once** the G1 URDF, Euler conventions, and a
sample NPZ are available. Until then, the TODO markers tell you exactly where to
paste diagnostics.

## Prerequisites

Minimum:
- macOS, Linux, or Windows (see note below)
- Python 3.10+
- Git

For GEM-X retargeting:
- NVIDIA GPU + CUDA (GEM-X default)
- Optional: ONNX Runtime path for macOS (no CUDA)

For training (later stages):
- MuJoCo dependencies for unitree_rl_mjlab

Repo expectations:
- The folders GEM-X/ and unitree_rl_mjlab/ should exist at repo root.
  If they are not present, clone them separately or add as submodules.

Windows note:
- The pipeline can run on Windows, but GEM-X + CUDA + MuJoCo are most reliable on
  Linux. For Windows users, we recommend WSL2 + Ubuntu with NVIDIA GPU passthrough.

## Install (subprojects)

```bash
chmod +x scripts/install_subprojects.sh
bash scripts/install_subprojects.sh
```

This pulls:
- GEM-X (NVlabs)
- unitree_rl_mjlab (Unitree)

## Install (pipeline only)

```bash
chmod +x scripts/install_pipeline.sh
bash scripts/install_pipeline.sh
source .venv/bin/activate
```

This installs:
- numpy, pandas, pyyaml, scipy, shapely, pin (Pinocchio)

## Install (GEM-X retargeting)

GEM-X is included as a subfolder and has its own setup steps. Follow:
- GEM-X/README.md
- GEM-X/docs/INSTALL.md

If you are on macOS without CUDA, check:
- GEM-X/docs/INSTALL_MACOS.md

## One-go pipeline run

```bash
python scripts/run_pipeline.py \
  --video /path/to/input.mp4 \
  --urdf /path/to/g1.urdf \
  --root-euler-order xyz \
  --angles-deg
```

What this does:
1. Runs GEM-X retargeting (GPU required unless you use ONNX).
2. Writes retarget outputs under cloud_outputs/<motion_id>/
3. Runs motion annotation and writes data/motions_retargeted/*.npz

Skip steps if needed:

```bash
python scripts/run_pipeline.py --video /path/to/input.mp4 --skip-gemx
python scripts/run_pipeline.py --video /path/to/input.mp4 --skip-annotate
```

## Required diagnostics (must be filled)

Before training, you **must** fill the TODO blocks with real diagnostics:

- src/dynamics/pinocchio_wrapper.py
- scripts/annotate_motion_library.py

Run the diagnostics listed in those files and paste their outputs at the top.
This locks in:
- model.nq, model.nv, joint names
- exact left/right foot frame names
- real NPZ keys and shapes
- real CSV column names

## Checklist

Use this checklist when setting up on a new machine:

- [ ] Install subprojects (scripts/install_subprojects.sh)
- [ ] Install pipeline deps (scripts/install_pipeline.sh)
- [ ] Install GEM-X deps (GEM-X/docs/INSTALL.md)
- [ ] Put G1 URDF on disk and note its path
- [ ] Determine Euler order and units for root_rotateXYZ
- [ ] Run diagnostics and paste results into TODO blocks
- [ ] Run scripts/run_pipeline.py on a test MP4
- [ ] Fill configs/motion_families.yaml with real labels
- [ ] Wire training scripts to your MuJoCo PPO loop (stubs are TODO)

## Training scripts

These are stubs until your PPO loop is wired:
- scripts/train_com.py
- scripts/train_momentum.py
- scripts/train_fall.py
- scripts/train_recovery.py
- scripts/train_switch.py

## Repo layout

```
src/
  dynamics/
  envs/
  rewards/
  switch/
scripts/
  run_pipeline.py
  annotate_motion_library.py
  train_*.py
configs/
  com.yaml, momentum.yaml, fall.yaml, recovery.yaml, switch.yaml
```

## Git hygiene

Outputs are ignored by the root .gitignore. Do not commit:
- cloud_outputs/
- data/motions_retargeted/
- data/splits/
- large media files or logs

## License

Follow the licenses of bundled subprojects (GEM-X, unitree_rl_mjlab) for any
redistribution or deployment.
