from __future__ import annotations

from typing import Any

import numpy as np


def _tracking_joint(q: np.ndarray, q_ref: np.ndarray, W: float = 1.0) -> float:
    return -W * float(np.sum((q - q_ref) ** 2))


def _tracking_ee(ee_pos: np.ndarray, ee_ref: np.ndarray, W: float = 1.0) -> float:
    return -W * float(np.sum((ee_pos - ee_ref) ** 2))


def _action_smoothness(action: np.ndarray, prev_action: np.ndarray, W: float = 0.01) -> float:
    return -W * float(np.sum((action - prev_action) ** 2))


def _com_support_margin(com_margin: float, delta: float = 0.05, lam: float = 1.0) -> float:
    return -lam * max(0.0, delta - float(com_margin)) ** 2


def _capture_point_margin(cp_margin: float, delta: float = 0.05, lam: float = 1.0) -> float:
    return -lam * max(0.0, delta - float(cp_margin)) ** 2


def _momentum_magnitude(hg_angular: np.ndarray, W: float = 0.1) -> float:
    return -W * float(np.dot(hg_angular, hg_angular))


def _momentum_phase_weighted(hg_angular: np.ndarray, phase_weight: float, W: float = 0.1) -> float:
    return -W * float(phase_weight) * float(np.dot(hg_angular, hg_angular))


def _impact_penalty(impact_forces: dict, vulnerable_weights: dict) -> float:
    return -sum(float(vulnerable_weights.get(b, 0.0)) * float(f) ** 2 for b, f in impact_forces.items())


def _head_hit_penalty(head_contact: bool, lam: float = 5.0) -> float:
    return -lam * float(head_contact)


def _recovery_success(is_upright: bool, bonus: float = 10.0) -> float:
    return bonus * float(is_upright)


def _time_to_upright(step: int, lam: float = 0.01) -> float:
    return -lam * float(step)


class RewardBuilder:
    """Stateless reward computation. Returns (total, breakdown)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def compute(self, **kwargs: Any) -> tuple[float, dict[str, float]]:
        total = 0.0
        breakdown: dict[str, float] = {}

        for term_name, term_cfg in self.cfg.items():
            if term_name == "tracking_joint":
                W = float(term_cfg) if not isinstance(term_cfg, dict) else float(term_cfg.get("W", 1.0))
                value = _tracking_joint(kwargs["q"], kwargs["q_ref"], W=W)
            elif term_name == "tracking_ee":
                W = float(term_cfg) if not isinstance(term_cfg, dict) else float(term_cfg.get("W", 1.0))
                value = _tracking_ee(kwargs["ee_pos"], kwargs["ee_ref"], W=W)
            elif term_name == "action_smoothness":
                W = float(term_cfg) if not isinstance(term_cfg, dict) else float(term_cfg.get("W", 0.01))
                value = _action_smoothness(kwargs["action"], kwargs["prev_action"], W=W)
            elif term_name == "com_support_margin":
                cfg = term_cfg if isinstance(term_cfg, dict) else {"lam": float(term_cfg)}
                value = _com_support_margin(
                    kwargs["com_margin"],
                    delta=float(cfg.get("delta", 0.05)),
                    lam=float(cfg.get("lam", 1.0)),
                )
            elif term_name == "capture_point_margin":
                cfg = term_cfg if isinstance(term_cfg, dict) else {"lam": float(term_cfg)}
                value = _capture_point_margin(
                    kwargs["cp_margin"],
                    delta=float(cfg.get("delta", 0.05)),
                    lam=float(cfg.get("lam", 1.0)),
                )
            elif term_name == "momentum_magnitude":
                W = float(term_cfg) if not isinstance(term_cfg, dict) else float(term_cfg.get("W", 0.1))
                value = _momentum_magnitude(kwargs["hg_angular"], W=W)
            elif term_name == "momentum_phase_weighted":
                cfg = term_cfg if isinstance(term_cfg, dict) else {"W": float(term_cfg)}
                value = _momentum_phase_weighted(
                    kwargs["hg_angular"],
                    kwargs["phase_weight"],
                    W=float(cfg.get("W", 0.1)),
                )
            elif term_name == "impact_penalty":
                cfg = term_cfg if isinstance(term_cfg, dict) else {}
                weight = float(cfg.get("weight", 1.0))
                value = weight * _impact_penalty(
                    kwargs["impact_forces"],
                    cfg.get("vulnerable_weights", {}),
                )
            elif term_name == "head_hit_penalty":
                cfg = term_cfg if isinstance(term_cfg, dict) else {"lam": float(term_cfg)}
                value = _head_hit_penalty(kwargs["head_contact"], lam=float(cfg.get("lam", 5.0)))
            elif term_name == "recovery_success":
                cfg = term_cfg if isinstance(term_cfg, dict) else {"bonus": float(term_cfg)}
                value = _recovery_success(kwargs["is_upright"], bonus=float(cfg.get("bonus", 10.0)))
            elif term_name == "time_to_upright":
                cfg = term_cfg if isinstance(term_cfg, dict) else {"lam": float(term_cfg)}
                value = _time_to_upright(kwargs["step"], lam=float(cfg.get("lam", 0.01)))
            else:
                raise KeyError(f"Unknown reward term: {term_name}")

            breakdown[term_name] = float(value)
            total += float(value)

        return float(total), breakdown


if __name__ == "__main__":
    cfg = {
        "tracking_joint": 1.0,
        "tracking_ee": 0.5,
        "action_smoothness": 0.01,
        "com_support_margin": {"lam": 0.5, "delta": 0.05},
        "capture_point_margin": {"lam": 0.5, "delta": 0.05},
        "momentum_magnitude": 0.1,
        "momentum_phase_weighted": 0.1,
        "impact_penalty": {"vulnerable_weights": {"head": 5.0}},
        "head_hit_penalty": 5.0,
        "recovery_success": 10.0,
        "time_to_upright": 0.01,
    }
    rb = RewardBuilder(cfg)
    total, breakdown = rb.compute(
        q=np.zeros(3),
        q_ref=np.zeros(3),
        ee_pos=np.zeros(3),
        ee_ref=np.zeros(3),
        action=np.zeros(2),
        prev_action=np.zeros(2),
        com_margin=0.1,
        cp_margin=0.1,
        hg_angular=np.zeros(3),
        phase_weight=1.0,
        impact_forces={"head": 0.1},
        head_contact=False,
        is_upright=True,
        step=5,
    )
    assert abs(total - sum(breakdown.values())) < 1e-6
    print("RewardBuilder smoke test passed.")
