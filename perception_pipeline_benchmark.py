# -*- coding: utf-8 -*-
"""
perception_pipeline_benchmark.py
================================================================================
Reference Implementation & Hardware/Synthetic Latency Profiler for Section 3:
1. End-to-End 5-Stage Pipeline Micro-Benchmark (Table 1 methodology).
2. Multi-Scale ROI ONNX Runtime / DirectML Latency Profiler (Table 2 methodology).
3. Foveated Dual-Threshold Confidence Recall (Section 3.4.1).
4. Sub-Part Bipartite Geometric Clustering & Occlusion Prior Synthesis (Section 3.4.2).
5. 3D Depth-Scale Metric Weighting & Hysteresis Sticky Target Selector (Section 3.4.3).
6. OS-Native Win32 SendInput Actuator Interface Specification (Dry-Run Safe Mode).
"""

import time
import math
import ctypes
from typing import List, Dict, Optional, Tuple
import numpy as np


# ==============================================================================
# 1. In-Memory ONNX Graph Builder & Live Inference Session Factory
# ==============================================================================
def _create_live_ort_session(resolution: int) -> Tuple[Optional[object], str]:
    """
    Constructs a weight-free synthetic CNN backbone in memory to benchmark
    ONNX Runtime execution overhead across ROI tensor resolutions.
    """
    try:
        import onnx
        from onnx import helper, TensorProto
        import onnxruntime as ort

        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, resolution, resolution])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, resolution // 4, resolution // 4])

        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((16, 3, 3, 3)).astype(np.float32) * 0.05
        w2 = rng.standard_normal((16, 16, 3, 3)).astype(np.float32) * 0.05

        init_w1 = helper.make_tensor("w1", TensorProto.FLOAT, w1.shape, w1.flatten().tolist())
        init_w2 = helper.make_tensor("w2", TensorProto.FLOAT, w2.shape, w2.flatten().tolist())

        conv1 = helper.make_node("Conv", ["input", "w1"], ["c1"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        relu1 = helper.make_node("Relu", ["c1"], ["r1"])
        conv2 = helper.make_node("Conv", ["r1", "w2"], ["c2"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
        relu2 = helper.make_node("Relu", ["c2"], ["output"])

        graph = helper.make_graph([conv1, relu1, conv2, relu2], f"SSVS_Synthetic_Backbone_{resolution}", [X], [Y], [init_w1, init_w2])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7

        avail = ort.get_available_providers()
        if "DmlExecutionProvider" in avail:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess = ort.InferenceSession(model.SerializeToString(), sess_options=sess_opts, providers=providers)
        return sess, sess.get_providers()[0]
    except Exception as exc:
        return None, f"NumPy-Fallback ({type(exc).__name__})"


def _run_synthetic_forward_pass(sess: Optional[object], tensor_nchw: np.ndarray) -> np.ndarray:
    if sess is not None:
        return sess.run(None, {"input": tensor_nchw})[0]
    s = tensor_nchw[:, :, ::2, ::2]
    return np.tanh(s[:, :, ::2, ::2])


# ==============================================================================
# 2. Foveated Dual-Threshold Confidence Recall (Section 3.4.1)
# ==============================================================================
def foveated_dual_threshold_filter(
    raw_detections: List[Dict],
    crosshair_pt: Tuple[float, float],
    tau_fovea: float = 0.35,
    tau_periph: float = 0.55,
    fovea_radius_px: float = 180.0,
) -> List[Dict]:
    cx, cy = crosshair_pt
    valid = []
    for det in raw_detections:
        dcx, dcy = det["center"]
        dist = float(math.hypot(dcx - cx, dcy - cy))
        thresh = tau_fovea if dist <= fovea_radius_px else tau_periph
        if float(det["conf"]) >= thresh:
            valid.append(det)
    return valid


# ==============================================================================
# 3. Sub-Part Bipartite Geometric Clustering & Prior Synthesis (Section 3.4.2)
# ==============================================================================
class CompositeTargetEntity:
    """
    Fuses primary body bounding box B_b and salient upper sub-part box B_h.
    Enforces upright aspect-ratio (h_b / w_b >= 1.45) and confidence (c_b >= 0.50)
    gating when synthesizing upper sub-region anchors under foveal occlusion.
    """
    def __init__(self, body_det: Optional[Dict] = None, upper_det: Optional[Dict] = None):
        if body_det is None and upper_det is None:
            raise ValueError("CompositeTargetEntity requires at least one valid detection.")
        self.body_det = body_det
        self.upper_det = upper_det

    @property
    def box(self) -> List[float]:
        if self.body_det and self.upper_det:
            b, h = self.body_det["box"], self.upper_det["box"]
            return [min(b[0], h[0]), min(b[1], h[1]), max(b[2], h[2]), max(b[3], h[3])]
        return self.body_det["box"] if self.body_det else self.upper_det["box"]

    @property
    def conf(self) -> float:
        if self.upper_det and self.body_det:
            return max(float(self.upper_det["conf"]), float(self.body_det["conf"]))
        return float(self.upper_det["conf"]) if self.upper_det else float(self.body_det["conf"])

    def get_anchor_point(self) -> Tuple[float, float]:
        if self.upper_det is not None:
            hx1, hy1, hx2, hy2 = self.upper_det["box"]
            return (hx1 + hx2) / 2.0, hy1 + max(1.0, hy2 - hy1) * 0.35

        bx1, by1, bx2, by2 = self.body_det["box"]
        bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
        ax = (bx1 + bx2) / 2.0
        # Enforce upright aspect ratio (h_b / w_b >= 1.45) and confidence (c_b >= 0.50) prior gate
        if (bh / bw) >= 1.45 and float(self.body_det.get("conf", 0.0)) >= 0.50:
            synth_h = min(bh * 0.25, max(14.0, bw * 0.38))
            ay = by1 + synth_h * 0.35
        else:
            ay = by1 + bh * 0.25
        return ax, ay


def cluster_bipartite_targets(detections: List[Dict]) -> List[CompositeTargetEntity]:
    """
    Executes global ascending-cost greedy bipartite matching between primary body
    boxes and upper salient sub-region boxes under axial alignment constraints:
      |x_h - x_b| <= 0.15 * w_b
    """
    uppers = [d for d in detections if d.get("is_upper_part", False)]
    bodies = [d for d in detections if not d.get("is_upper_part", False)]

    candidates = []
    for b_idx, b in enumerate(bodies):
        bx1, by1, bx2, by2 = b["box"]
        bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
        bcx = (bx1 + bx2) / 2.0
        b_neck_y = by1 + min(bh * 0.20, 25.0)

        for h_idx, h in enumerate(uppers):
            hx1, hy1, hx2, hy2 = h["box"]
            hw, hh = max(1.0, hx2 - hx1), max(1.0, hy2 - hy1)
            hcx, hcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0

            inter_w = max(0.0, min(bx2, hx2) - max(bx1, hx1))
            inter_h = max(0.0, min(by2, hy2) - max(by1, hy1))
            overlap_ratio = (inter_w * inter_h) / max(1.0, hw * hh)

            v_up_margin = max(bh * 0.45, hh * 1.50, bw * 0.65)
            # Exact axial shoulder alignment constraint: |x_h - x_b| <= 0.15 * w_b
            h_in_horiz = abs(hcx - bcx) <= (0.15 * bw)
            h_in_vert = (by1 - v_up_margin <= hcy <= by1 + bh * 0.90)

            if h_in_horiz and (overlap_ratio > 0.08 or h_in_vert):
                cost = abs(hcx - bcx) * 3.5 + abs(hcy - b_neck_y) * 1.2 - overlap_ratio * 35.0
                candidates.append((cost, b_idx, h_idx))

    candidates.sort(key=lambda x: x[0])
    matched_b, matched_h, b_to_h = set(), set(), {}
    for _, b_idx, h_idx in candidates:
        if b_idx not in matched_b and h_idx not in matched_h:
            matched_b.add(b_idx)
            matched_h.add(h_idx)
            b_to_h[b_idx] = h_idx

    entities = [
        CompositeTargetEntity(body_det=b, upper_det=uppers[b_to_h[i]] if i in b_to_h else None)
        for i, b in enumerate(bodies)
    ]
    for h_idx, h in enumerate(uppers):
        if h_idx not in matched_h:
            entities.append(CompositeTargetEntity(body_det=None, upper_det=h))
    return entities


# ==============================================================================
# 4. Depth-Scale Weighted & Hysteresis Sticky Selector (Section 3.4.3)
# ==============================================================================
class DepthWeightedStickySelector:
    def __init__(self, sticky_ratio: float = 1.35):
        self.sticky_ratio = float(sticky_ratio)
        self.locked_pt: Optional[Tuple[float, float]] = None

    def select(self, detections: List[Dict], crosshair_pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        entities = cluster_bipartite_targets(detections)
        if not entities:
            self.locked_pt = None
            return None

        cx, cy = crosshair_pt
        cands = []
        for ent in entities:
            ax, ay = ent.get_anchor_point()
            dist_raw = float(math.hypot(ax - cx, ay - cy))
            bx1, by1, bx2, by2 = ent.box
            bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)

            dx = max(0.0, bx1 - cx, cx - bx2)
            dy = max(0.0, by1 - cy, cy - by2)
            box_dist = float(math.hypot(dx, dy))
            gamma_in = 0.40 if box_dist == 0.0 else 1.0

            s_depth = max(0.40, min(2.5, max(bh, bw * 1.8) / 120.0))
            d_eff = gamma_in * (dist_raw + 1.6 * box_dist) / (s_depth ** 0.85)
            cands.append({"ax": ax, "ay": ay, "d_eff": d_eff})

        best = min(cands, key=lambda c: c["d_eff"])
        if self.locked_pt is not None:
            curr_locked = min(cands, key=lambda c: math.hypot(c["ax"] - self.locked_pt[0], c["ay"] - self.locked_pt[1]))
            if math.hypot(curr_locked["ax"] - self.locked_pt[0], curr_locked["ay"] - self.locked_pt[1]) < 45.0:
                if best is not curr_locked and not (best["d_eff"] * self.sticky_ratio < curr_locked["d_eff"]):
                    best = curr_locked

        self.locked_pt = (best["ax"], best["ay"])
        return self.locked_pt


# ==============================================================================
# 5. OS-Native Win32 SendInput Actuator Specification (Dry-Run Safe Mode)
# ==============================================================================
ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", _INPUT_UNION),
    ]


class Win32ActuatorDispatcher:
    """
    Demonstrates the OS-native SendInput structure marshaling, relative displacement
    dispatch, and sub-count residual accumulation (defaulting to dry_run=True so
    reviewer cursors are never moved unintentionally).
    """
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.rem_x = 0.0
        self.rem_y = 0.0

    def dispatch_relative(self, dx: float, dy: float) -> Tuple[int, int]:
        dx += self.rem_x
        dy += self.rem_y
        ix = math.trunc(dx) if abs(dx) >= 1.0 else 0
        iy = math.trunc(dy) if abs(dy) >= 1.0 else 0
        self.rem_x, self.rem_y = dx - ix, dy - iy

        pkt = _INPUT(type=0, union=_INPUT_UNION(mi=_MOUSEINPUT(dx=ix, dy=iy, mouseData=0, dwFlags=0x0001, time=0, dwExtraInfo=0)))
        _ = ctypes.sizeof(pkt)

        if not self.dry_run and (ix != 0 or iy != 0):
            if not hasattr(ctypes, "windll"):
                raise RuntimeError("Win32 SendInput is only supported on Windows platforms.")
            sent = ctypes.windll.user32.SendInput(1, ctypes.byref(pkt), ctypes.sizeof(_INPUT))
            if sent != 1:
                raise RuntimeError("Win32 SendInput dispatch failed.")
        return ix, iy


# ==============================================================================
# 6. Multi-Scale ROI Profiler (Table 2) & 5-Stage Micro-Benchmark (Table 1)
# ==============================================================================
def profile_multiscale_roi_latency(
    resolutions: Tuple[int, ...] = (256, 320, 416, 512, 640),
    n_warmup: int = 8,
    n_iters: int = 40,
) -> Dict[int, Dict[str, float]]:
    reference_rtx3080_dml = {
        256: {"fov_px": 70,  "mean_ms": 5.82,  "std_ms": 0.45},
        320: {"fov_px": 90,  "mean_ms": 8.24,  "std_ms": 0.76},
        416: {"fov_px": 120, "mean_ms": 11.45, "std_ms": 0.92},
        512: {"fov_px": 150, "mean_ms": 15.10, "std_ms": 1.15},
        640: {"fov_px": 180, "mean_ms": 21.30, "std_ms": 1.48},
    }

    _, sample_prov = _create_live_ort_session(256)
    print(f"=== Table 2 Profiler: Multi-Scale ROI Latency (Synthetic Backbone Backend: {sample_prov}) ===")
    print("| ROI SxS (px) | R_fov (px) | Paper Full-YOLO RTX 3080 Reference (ms) | Paper Max FPS | Live Synthetic Backbone Pass (ms) |")
    print("|:---:|:---:|:---:|:---:|:---:|")

    results = {}
    rng = np.random.default_rng(42)
    dummy_screen = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    cx, cy = 960, 540

    for s in resolutions:
        sess, _ = _create_live_ort_session(s)
        half = s // 2
        h_scr, w_scr = dummy_screen.shape[:2]
        y1 = max(0, min(h_scr - s, cy - half))
        x1 = max(0, min(w_scr - s, cx - half))
        for _ in range(n_warmup):
            roi = np.ascontiguousarray(dummy_screen[y1:y1 + s, x1:x1 + s].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            _run_synthetic_forward_pass(sess, roi)

        live_times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            roi = np.ascontiguousarray(dummy_screen[y1:y1 + s, x1:x1 + s].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            _run_synthetic_forward_pass(sess, roi)
            live_times.append((time.perf_counter() - t0) * 1000.0)

        live_mean = float(np.mean(live_times))
        live_std = float(np.std(live_times))
        ref = reference_rtx3080_dml[s]
        fps = 1000.0 / ref["mean_ms"]
        results[s] = {
            "fov_px": ref["fov_px"],
            "paper_mean_ms": ref["mean_ms"],
            "paper_std_ms": ref["std_ms"],
            "live_mean_ms": round(live_mean, 3),
            "live_std_ms": round(live_std, 3),
        }
        print(f"| {s}x{s} | {ref['fov_px']} | {ref['mean_ms']:.2f} ± {ref['std_ms']:.2f} | ~{int(round(fps))} | {live_mean:.3f} ± {live_std:.3f} |")

    return results


def profile_end_to_end_stages(n_iters: int = 100) -> Dict[str, Dict[str, float]]:
    from ssvs_core import SmoothSSVSController

    sess_320, backend_name = _create_live_ort_session(320)
    ctrl = SmoothSSVSController()
    selector = DepthWeightedStickySelector(sticky_ratio=1.35)
    dispatcher = Win32ActuatorDispatcher(dry_run=True)

    rng = np.random.default_rng(42)
    dummy_screen = rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    sample_dets = [
        {"box": [145, 110, 175, 195], "conf": 0.78, "center": (160, 152), "is_upper_part": False},
        {"box": [152, 96, 168, 114], "conf": 0.42, "center": (160, 105), "is_upper_part": True},
        {"box": [210, 140, 225, 175], "conf": 0.65, "center": (217, 157), "is_upper_part": False},
    ]

    t_cap_list, t_inf_list, t_post_list, t_ctrl_list, t_act_list = [], [], [], [], []

    for _ in range(n_iters):
        t0 = time.perf_counter()
        roi_tensor = np.ascontiguousarray(dummy_screen[380:700, 800:1120].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        t_cap_list.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = _run_synthetic_forward_pass(sess_320, roi_tensor)
        t_inf_list.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        recalled = foveated_dual_threshold_filter(sample_dets, crosshair_pt=(160.0, 160.0))
        anchor = selector.select(recalled, crosshair_pt=(160.0, 160.0))
        t_post_list.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        if anchor is not None:
            ux, uy = ctrl.step(anchor[0] + 15.0, anchor[1] - 8.0, (160.0, 160.0), dt=0.001)
        else:
            ux, uy = 0.0, 0.0
        t_ctrl_list.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        ix, iy = dispatcher.dispatch_relative(ux, uy)
        count_scale = max(1e-4, ctrl.last_count_scale)
        ctrl.last_sent_dx = ix / count_scale
        ctrl.last_sent_dy = iy / count_scale
        t_act_list.append((time.perf_counter() - t0) * 1000.0)

    paper_table1 = [
        ("Stage 1: Surface Ingestion (tau_cap)", 2.12, 0.38, t_cap_list),
        ("Stage 2: Neural Inference (tau_inf)",  8.24, 0.76, t_inf_list),
        ("Stage 3: Post-Processing (tau_post)",  0.58, 0.12, t_post_list),
        ("Stage 4: Control Synthesis (tau_ctrl)", 0.28, 0.05, t_ctrl_list),
        ("Stage 5: Actuator Dispatch (tau_act)", 1.18, 0.21, t_act_list),
    ]

    print(f"\n=== Table 1 Profiler: 5-Stage Latency Breakdown (N={n_iters}, Backend: {backend_name}) ===")
    print("| Pipeline Stage | Paper Full-System Telemetry (ms) | Live Host Micro-Benchmark (ms) |")
    print("|---|:---:|:---:|")
    summary = {}
    for name, ref_m, ref_s, lst in paper_table1:
        lm, ls = float(np.mean(lst)), float(np.std(lst))
        summary[name] = {"paper_mean_ms": ref_m, "live_mean_ms": round(lm, 4)}
        print(f"| {name} | {ref_m:.2f} ± {ref_s:.2f} | {lm:.4f} ± {ls:.4f} |")

    return summary


if __name__ == "__main__":
    profile_end_to_end_stages(n_iters=100)
    print()
    profile_multiscale_roi_latency()
