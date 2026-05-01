from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    NOMINAL = "nominal"
    FALL = "fall"
    RECOVERY = "recovery"


@dataclass
class SwitchConfig:
    delta1: float = 0.05
    delta2: float = 15.0
    delta3: float = 0.20
    min_dwell_steps: int = 30


class ModeSwitch:
    """Switching law with hysteresis and transition logging."""

    def __init__(self, cfg: SwitchConfig):
        self.cfg = cfg
        self._mode = Mode.NOMINAL
        self._dwell_counter = 0
        self._transition_log: list[dict] = []

    def step(self, features: dict) -> Mode:
        cp_margin = float(features["cp_margin"])
        momentum_norm = float(features["momentum_norm"])
        base_height = float(features["base_height"])
        is_fallen = bool(features["is_fallen"])
        is_upright = bool(features["is_upright"])
        step_index = int(features["step_index"])

        prev = self._mode
        trigger = ""

        if self._mode == Mode.NOMINAL:
            if cp_margin <= self.cfg.delta1:
                self._mode = Mode.FALL
                trigger = "cp_margin"
            elif momentum_norm >= self.cfg.delta2:
                self._mode = Mode.FALL
                trigger = "momentum_norm"
        elif self._mode == Mode.FALL:
            if base_height <= self.cfg.delta3:
                self._mode = Mode.RECOVERY
                trigger = "base_height"
            elif is_fallen:
                self._mode = Mode.RECOVERY
                trigger = "is_fallen"
        elif self._mode == Mode.RECOVERY:
            if is_upright:
                self._dwell_counter += 1
                if self._dwell_counter >= self.cfg.min_dwell_steps:
                    self._mode = Mode.NOMINAL
                    trigger = "upright_dwell"
            else:
                self._dwell_counter = 0

        if prev != self._mode:
            self._transition_log.append(
                {"step": step_index, "from": prev.value, "to": self._mode.value, "trigger": trigger}
            )
            if self._mode != Mode.RECOVERY:
                self._dwell_counter = 0

        return self._mode

    def reset(self) -> None:
        self._mode = Mode.NOMINAL
        self._dwell_counter = 0
        self._transition_log = []

    @property
    def current_mode(self) -> Mode:
        return self._mode

    @property
    def transition_log(self) -> list[dict]:
        return list(self._transition_log)


if __name__ == "__main__":
    cfg = SwitchConfig(delta1=0.05, delta2=15.0, delta3=0.2, min_dwell_steps=2)
    switch = ModeSwitch(cfg)

    sequence = [
        {"cp_margin": 0.1, "momentum_norm": 0.0, "base_height": 1.0, "is_fallen": False, "is_upright": True, "step_index": 0},
        {"cp_margin": 0.0, "momentum_norm": 0.0, "base_height": 1.0, "is_fallen": False, "is_upright": True, "step_index": 1},
        {"cp_margin": 0.0, "momentum_norm": 0.0, "base_height": 0.1, "is_fallen": False, "is_upright": False, "step_index": 2},
        {"cp_margin": 0.1, "momentum_norm": 0.0, "base_height": 0.5, "is_fallen": False, "is_upright": True, "step_index": 3},
        {"cp_margin": 0.1, "momentum_norm": 0.0, "base_height": 0.5, "is_fallen": False, "is_upright": True, "step_index": 4},
    ]

    modes = [switch.step(step) for step in sequence]
    assert modes[0] == Mode.NOMINAL
    assert modes[1] == Mode.FALL
    assert modes[2] == Mode.RECOVERY
    assert modes[-1] == Mode.NOMINAL
    print("ModeSwitch smoke test passed.")
