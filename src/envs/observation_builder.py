from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import yaml
except Exception:
    yaml = None


@dataclass
class ObsConfig:
    proprioception: bool = True
    reference: bool = True
    future_window: int = 2
    phase_embed_dim: int = 4
    family_embed_dim: int = 7
    com_support: bool = False
    momentum: bool = False
    momentum_history: int = 0
    switch_features: bool = False


class ObservationBuilder:
    def __init__(self, cfg: ObsConfig):
        self.cfg = cfg
        self._obs_dim: int | None = None

    @staticmethod
    def _phase_embed(phase: int, dim: int) -> np.ndarray:
        if dim <= 0:
            return np.zeros(0, dtype=np.float32)
        angle = 2.0 * np.pi * float(phase) / 10.0
        feats = []
        pairs = dim // 2
        for k in range(1, pairs + 1):
            feats.extend([np.sin(k * angle), np.cos(k * angle)])
        if dim % 2 == 1:
            feats.append(np.sin((pairs + 1) * angle))
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _family_one_hot(family: int, dim: int) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        if 0 <= int(family) < dim:
            vec[int(family)] = 1.0
        return vec

    def build(
        self,
        prop: dict,
        ref: dict,
        physics: dict,
        switch: dict,
    ) -> tuple[np.ndarray, dict[str, slice]]:
        parts: list[np.ndarray] = []
        slices: dict[str, slice] = {}
        offset = 0

        def _append(name: str, arr: np.ndarray) -> None:
            nonlocal offset
            arr = np.asarray(arr, dtype=np.float32).ravel()
            parts.append(arr)
            slices[name] = slice(offset, offset + arr.size)
            offset += arr.size

        if self.cfg.proprioception:
            q = np.asarray(prop["q"], dtype=np.float32).ravel()
            dq = np.asarray(prop["dq"], dtype=np.float32).ravel()
            base_orient = np.asarray(prop["base_orient"], dtype=np.float32).ravel()
            base_angvel = np.asarray(prop["base_angvel"], dtype=np.float32).ravel()
            _append("proprioception", np.concatenate([q, dq, base_orient, base_angvel]))

        if self.cfg.reference:
            q_ref = np.asarray(ref["q_ref"], dtype=np.float32).ravel()
            dq_ref = np.asarray(ref["dq_ref"], dtype=np.float32).ravel()
            _append("reference", np.concatenate([q_ref, dq_ref]))

            if self.cfg.future_window > 0:
                future = np.asarray(ref["future_refs"], dtype=np.float32).reshape(
                    self.cfg.future_window, -1
                )
                _append("future_ref", future.ravel())

            phase = int(ref["phase"])
            _append("phase_embed", self._phase_embed(phase, self.cfg.phase_embed_dim))

            family = int(ref["family"])
            _append("family_embed", self._family_one_hot(family, self.cfg.family_embed_dim))

        if self.cfg.com_support:
            com_margin = float(physics["com_margin"])
            cp_margin = float(physics["cp_margin"])
            support_center = np.asarray(physics["support_center"], dtype=np.float32).ravel()
            support_area = np.array([float(physics["support_area"])], dtype=np.float32)
            capture_point = np.asarray(physics["capture_point"], dtype=np.float32).ravel()
            _append(
                "com_support",
                np.concatenate([support_center, support_area, capture_point, [com_margin, cp_margin]]),
            )

        if self.cfg.momentum:
            hg = np.asarray(physics["hg"], dtype=np.float32).ravel()
            _append("momentum", hg)

        if self.cfg.momentum_history > 0:
            hg_hist = np.asarray(
                physics.get("hg_history", np.zeros((self.cfg.momentum_history, 6))),
                dtype=np.float32,
            )
            _append("momentum_history", hg_hist.ravel())

        if self.cfg.switch_features:
            base_height = float(switch["base_height"])
            torso_pitch = float(switch["torso_pitch"])
            torso_roll = float(switch["torso_roll"])
            contact_count = float(switch["contact_count"])
            slip_flag = float(switch["slip_flag"])
            mode = str(switch["mode"]).lower()
            mode_vec = np.array(
                [float(mode == "nominal"), float(mode == "fall"), float(mode == "recovery")],
                dtype=np.float32,
            )
            _append(
                "switch",
                np.array(
                    [base_height, torso_pitch, torso_roll, contact_count, slip_flag],
                    dtype=np.float32,
                ),
            )
            _append("switch_mode", mode_vec)

        obs = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        self._obs_dim = int(obs.size)
        return obs, slices

    @classmethod
    def from_config(cls, cfg_path: str) -> "ObservationBuilder":
        if yaml is None:
            raise SystemExit("Missing dependency: pyyaml. Install it before loading configs.")
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        obs_cfg = raw.get("obs", raw)
        cfg = ObsConfig(**obs_cfg)
        return cls(cfg)

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            raise ValueError("obs_dim is unknown until build() has been called once.")
        return self._obs_dim


if __name__ == "__main__":
    cfg = ObsConfig(
        proprioception=True,
        reference=True,
        future_window=2,
        phase_embed_dim=4,
        family_embed_dim=7,
        com_support=True,
        momentum=True,
        momentum_history=2,
        switch_features=True,
    )
    builder = ObservationBuilder(cfg)
    prop = {
        "q": np.zeros(10),
        "dq": np.zeros(9),
        "base_orient": np.zeros(4),
        "base_angvel": np.zeros(3),
    }
    ref = {
        "q_ref": np.zeros(10),
        "dq_ref": np.zeros(9),
        "future_refs": np.zeros((2, 10)),
        "phase": 0,
        "family": 0,
    }
    physics = {
        "com_margin": 0.1,
        "cp_margin": 0.1,
        "support_center": np.zeros(2),
        "support_area": 0.0,
        "capture_point": np.zeros(2),
        "hg": np.zeros(6),
        "hg_history": np.zeros((2, 6)),
    }
    switch = {
        "base_height": 0.5,
        "torso_pitch": 0.0,
        "torso_roll": 0.0,
        "contact_count": 2,
        "slip_flag": 0,
        "mode": "nominal",
    }
    obs, _ = builder.build(prop, ref, physics, switch)
    assert obs.shape[0] == builder.obs_dim
    print("ObservationBuilder smoke test passed.")
