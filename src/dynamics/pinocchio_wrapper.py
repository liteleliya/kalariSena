"""Pinocchio-based kinematics/dynamics accessors.

This module is the sole source of truth for CoM, centroidal momentum, and foot
frame computations. All other modules must import from here.
"""

# TODO: Run the URDF diagnostics and paste the real outputs here.
# G1 URDF diagnostics (from: python -c "import pinocchio as pin; ...")
# nq=TODO, nv=TODO
# Foot frames found: TODO
# Full frame list: TODO

from __future__ import annotations

from typing import Iterable
import math

import numpy as np
import pinocchio as pin

try:  # Optional, for convex hulls
    from scipy.spatial import ConvexHull

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:  # Optional, for convex hulls and distance
    from shapely.geometry import MultiPoint, Point, Polygon

    _HAVE_SHAPELY = True
except Exception:
    _HAVE_SHAPELY = False

_G = 9.81
_SUPPORT_SQUARE_SIDE = 0.05


def _square_around_point(pt_xy: np.ndarray, side: float) -> np.ndarray:
    half = side * 0.5
    return np.array(
        [
            [pt_xy[0] - half, pt_xy[1] - half],
            [pt_xy[0] + half, pt_xy[1] - half],
            [pt_xy[0] + half, pt_xy[1] + half],
            [pt_xy[0] - half, pt_xy[1] + half],
        ],
        dtype=np.float64,
    )


def _rectangle_around_segment(p0: np.ndarray, p1: np.ndarray, width: float) -> np.ndarray:
    seg = p1 - p0
    norm = np.linalg.norm(seg)
    if norm < 1e-8:
        return _square_around_point(p0, width)
    perp = np.array([-seg[1], seg[0]], dtype=np.float64) / norm
    half = width * 0.5
    offset = perp * half
    return np.array([p0 + offset, p1 + offset, p1 - offset, p0 - offset], dtype=np.float64)


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    area = _polygon_area(poly)
    if area < 1e-8:
        return np.mean(poly, axis=0)
    x = poly[:, 0]
    y = poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = float(np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area))
    cy = float(np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area))
    return np.array([cx, cy], dtype=np.float64)


def _point_to_segment_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(pt - a))
    t = float(np.dot(pt - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def _point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    if poly.shape[0] < 3:
        return False
    signs = []
    for i in range(poly.shape[0]):
        p0 = poly[i]
        p1 = poly[(i + 1) % poly.shape[0]]
        edge = p1 - p0
        rel = pt - p0
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if abs(cross) < 1e-12:
            continue
        signs.append(math.copysign(1.0, cross))
    if not signs:
        return True
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _signed_distance_to_polygon(pt: np.ndarray, poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return -999.0
    min_dist = float("inf")
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        min_dist = min(min_dist, _point_to_segment_distance(pt, a, b))
    inside = _point_in_polygon(pt, poly)
    return min_dist if inside else -min_dist


def _convex_hull(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] == 1:
        return _square_around_point(points_xy[0], _SUPPORT_SQUARE_SIDE)
    if points_xy.shape[0] == 2:
        return _rectangle_around_segment(points_xy[0], points_xy[1], _SUPPORT_SQUARE_SIDE)
    if _HAVE_SHAPELY:
        hull = MultiPoint(points_xy).convex_hull
        if hull.geom_type == "Polygon":
            return np.array(hull.exterior.coords[:-1], dtype=np.float64)
        if hull.geom_type == "LineString":
            coords = np.array(hull.coords, dtype=np.float64)
            if coords.shape[0] == 2:
                return _rectangle_around_segment(coords[0], coords[1], _SUPPORT_SQUARE_SIDE)
        if hull.geom_type == "Point":
            coords = np.array(hull.coords, dtype=np.float64)
            return _square_around_point(coords[0], _SUPPORT_SQUARE_SIDE)
    if _HAVE_SCIPY:
        hull = ConvexHull(points_xy)
        return points_xy[hull.vertices]
    raise ImportError("Support polygon requires scipy or shapely.")


class PinocchioWrapper:
    """Lightweight Pinocchio wrapper for G1 kinematics and dynamics."""

    # Optional overrides if auto-detection fails.
    DEFAULT_LEFT_FOOT_FRAMES: list[str] = []
    DEFAULT_RIGHT_FOOT_FRAMES: list[str] = []

    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._frame_names = [f.name for f in self.model.frames]

        self.left_foot_frame = self._resolve_foot_frame("left", self.DEFAULT_LEFT_FOOT_FRAMES)
        self.right_foot_frame = self._resolve_foot_frame("right", self.DEFAULT_RIGHT_FOOT_FRAMES)
        if self.left_foot_frame is None or self.right_foot_frame is None:
            raise ValueError(
                "Foot frame names not found. Run the URDF diagnostics (Step 2) "
                "and update DEFAULT_LEFT_FOOT_FRAMES/DEFAULT_RIGHT_FOOT_FRAMES."
            )
        if self.left_foot_frame not in self._frame_names:
            raise ValueError(f"Left foot frame not found: {self.left_foot_frame}")
        if self.right_foot_frame not in self._frame_names:
            raise ValueError(f"Right foot frame not found: {self.right_foot_frame}")
        self.left_foot_id = self.model.getFrameId(self.left_foot_frame)
        self.right_foot_id = self.model.getFrameId(self.right_foot_frame)

    def _resolve_foot_frame(self, side: str, preferred: Iterable[str]) -> str | None:
        for name in preferred:
            if name in self._frame_names:
                return name
        tokens = ("ankle", "foot", "toe", "heel", "link")
        candidates = [
            name
            for name in self._frame_names
            if side in name.lower() and any(tok in name.lower() for tok in tokens)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            for tok in ("ankle", "foot", "toe", "heel"):
                filtered = [name for name in candidates if tok in name.lower()]
                if len(filtered) == 1:
                    return filtered[0]
        return None

    def _update_all(self, q: np.ndarray, dq: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        dq = np.asarray(dq, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"q has shape {q.shape[0]} but model.nq={self.model.nq}")
        if dq.shape[0] != self.model.nv:
            raise ValueError(f"dq has shape {dq.shape[0]} but model.nv={self.model.nv}")
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.centerOfMass(self.model, self.data, q, dq)
        pin.computeCentroidalMomentum(self.model, self.data, q, dq)

    def compute_com(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (com [3], com_vel [3])."""
        self._update_all(q, dq)
        com = np.array(self.data.com[0]).reshape(3)
        com_vel = np.array(self.data.vcom[0]).reshape(3)
        return com, com_vel

    def compute_centroidal_momentum(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Returns hg [6] — [linear_momentum (3), angular_momentum (3)]."""
        self._update_all(q, dq)
        return np.array(self.data.hg.vector).reshape(6)

    def get_frame_pose(self, frame_name: str, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Returns 4x4 SE3 homogeneous transform (np.ndarray)."""
        self._update_all(q, dq)
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= len(self.model.frames):
            raise ValueError(f"Frame not found: {frame_name}")
        pose = self.data.oMf[frame_id]
        try:
            mat = pose.homogeneous
        except AttributeError:
            mat = pose.toHomogeneousMatrix()
        return np.array(mat, dtype=np.float64)

    def get_foot_positions(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (left_foot_pos [3], right_foot_pos [3]) in world frame."""
        self._update_all(q, dq)
        lf = np.array(self.data.oMf[self.left_foot_id].translation).reshape(3)
        rf = np.array(self.data.oMf[self.right_foot_id].translation).reshape(3)
        return lf, rf

    def get_support_features(
        self, q: np.ndarray, dq: np.ndarray, contacts: np.ndarray
    ) -> dict:
        """Compute support polygon and stability features."""
        contacts = np.asarray(contacts, dtype=bool).reshape(-1)
        if contacts.shape[0] != 2:
            raise ValueError("contacts must be shape [2] -> (left, right)")

        self._update_all(q, dq)
        com = np.array(self.data.com[0]).reshape(3)
        com_vel = np.array(self.data.vcom[0]).reshape(3)

        lf = np.array(self.data.oMf[self.left_foot_id].translation).reshape(3)
        rf = np.array(self.data.oMf[self.right_foot_id].translation).reshape(3)

        points_xy = []
        if contacts[0]:
            points_xy.append(lf[:2])
        if contacts[1]:
            points_xy.append(rf[:2])

        if not points_xy:
            return {
                "support_polygon": np.zeros((0, 2), dtype=np.float64),
                "support_area": 0.0,
                "support_center": np.array([np.nan, np.nan], dtype=np.float64),
                "com_margin": -999.0,
                "capture_point": np.array([np.nan, np.nan], dtype=np.float64),
                "cp_margin": -999.0,
                "support_mode": "no_contact",
            }

        points_xy = np.array(points_xy, dtype=np.float64)
        if points_xy.shape[0] == 1:
            support_poly = _square_around_point(points_xy[0], _SUPPORT_SQUARE_SIDE)
            support_mode = "single_left" if contacts[0] else "single_right"
        elif points_xy.shape[0] == 2:
            support_poly = _rectangle_around_segment(
                points_xy[0], points_xy[1], _SUPPORT_SQUARE_SIDE
            )
            support_mode = "double"
        else:
            support_poly = _convex_hull(points_xy)
            support_mode = "double"

        support_area = _polygon_area(support_poly)
        support_center = _polygon_centroid(support_poly)

        omega = math.sqrt(_G / max(float(com[2]), 0.1))
        capture_point = com[:2] + com_vel[:2] / omega

        com_margin = _signed_distance_to_polygon(com[:2], support_poly)
        cp_margin = _signed_distance_to_polygon(capture_point, support_poly)

        return {
            "support_polygon": support_poly,
            "support_area": float(support_area),
            "support_center": support_center,
            "com_margin": float(com_margin),
            "capture_point": capture_point,
            "cp_margin": float(cp_margin),
            "support_mode": support_mode,
        }


if __name__ == "__main__":
    import pinocchio as pin

    wrapper = PinocchioWrapper("assets/unitree_g1/g1.urdf")

    q0 = pin.neutral(wrapper.model)
    dq0 = np.zeros(wrapper.model.nv)

    com, com_vel = wrapper.compute_com(q0, dq0)
    hg = wrapper.compute_centroidal_momentum(q0, dq0)
    lf, rf = wrapper.get_foot_positions(q0, dq0)
    contacts = np.array([True, True])
    feats = wrapper.get_support_features(q0, dq0, contacts)

    print("CoM:", com)
    print("CoM vel:", com_vel)
    print("hg:", hg)
    print("Left foot:", lf)
    print("Right foot:", rf)
    print("CP margin:", feats["cp_margin"])
    print("Support mode:", feats["support_mode"])

    assert com[2] > 0.3, f"CoM height suspiciously low: {com[2]}"
    assert com[2] < 1.5, f"CoM height suspiciously high: {com[2]}"
    assert feats["support_mode"] == "double"
    assert feats["cp_margin"] > 0
    print("All smoke tests passed.")
