# -*- coding: utf-8 -*-
"""
ssvs_core.py — Low-Latency Screen-Space Visual Servoing (SSVS) Core Library
===========================================================================
Reference Implementation for:
"Low-Latency Screen-Space Visual Servoing via Asynchronous Perception-Control
Decoupling and Gain-Scheduled Discrete Tracking"
Submitted to: Journal of Real-Time Image Processing (Springer)
"""

import math
import cmath
import time
import ctypes
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Deque
import numpy as np


# ======================================================================
# 1. Speed-Adaptive 1-Euro Filter (Casiez et al., CHI 2012)
# ======================================================================
class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.020,
        d_cutoff: float = 1.0,
        v_clamp_px_s: float = 800.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.v_clamp_px_s = float(v_clamp_px_s)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / max(1e-4, dt))

    def filter(self, x: float, dt: Optional[float] = None, now: Optional[float] = None) -> Tuple[float, float]:
        if dt is None or not math.isfinite(dt) or dt <= 1e-5:
            if now is None:
                now = time.perf_counter()
            dt = 0.00625 if self.t_prev is None else max(1e-4, min(0.1, now - self.t_prev))
            self.t_prev = now
        else:
            dt = max(1e-4, min(0.1, float(dt)))
            if now is not None:
                self.t_prev = now

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x, 0.0

        raw_dx = (x - self.x_prev) / dt
        if self.v_clamp_px_s > 0.0:
            raw_dx = max(-self.v_clamp_px_s, min(self.v_clamp_px_s, raw_dx))

        alpha_d = self._alpha(self.d_cutoff, dt)
        filtered_dx = alpha_d * raw_dx + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = filtered_dx

        cutoff = self.min_cutoff + self.beta * abs(filtered_dx)
        alpha = self._alpha(cutoff, dt)
        filtered_x = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = filtered_x

        return filtered_x, filtered_dx


# ======================================================================
# 2. Five-Term Discrete-Time SSVS Controller
# ======================================================================
@dataclass
class SSVSConfig:
    p_gain_base: float = 0.36
    d_damping: float = 0.22
    ki_gain: float = 12.0
    smooth_ema: float = 0.70
    lead_delay_ms: float = 12.4
    sensitivity: float = 2.50
    sched_near_px: float = 15.0
    sched_far_px: float = 80.0
    dead_zone_px: float = 0.5
    u_max_px: float = 50.0
    synthetic_latency_injection_ms: float = 0.0
    blocking_wait_ms: float = 0.0
    dt_ref: float = 0.00625


def vector_actuator_saturation(ux: float, uy: float, max_mag: float, eps: float = 1e-6) -> Tuple[float, float]:
    mag = float(math.hypot(ux, uy))
    if mag > max_mag and mag > eps:
        scale = max_mag / mag
        return ux * scale, uy * scale
    return ux, uy


class SmoothSSVSController:
    def __init__(self, cfg: Optional[SSVSConfig] = None):
        self.cfg = cfg or SSVSConfig()
        self.filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.020, d_cutoff=1.0, v_clamp_px_s=800.0)
        self.filter_y = OneEuroFilter(min_cutoff=0.5, beta=0.008, d_cutoff=0.8, v_clamp_px_s=800.0)

        self.prev_err_x: Optional[float] = None
        self.prev_err_y: Optional[float] = None
        self.lead_smoothed_x: float = 0.0
        self.lead_smoothed_y: float = 0.0
        self.cruise_integral_x: float = 0.0
        self.last_sent_dx: float = 0.0
        self.last_sent_dy: float = 0.0
        self.last_count_scale: float = 1.0
        self.last_step_time: Optional[float] = None

    def reset(self) -> None:
        self.filter_x.reset()
        self.filter_y.reset()
        self.prev_err_x = None
        self.prev_err_y = None
        self.lead_smoothed_x = 0.0
        self.lead_smoothed_y = 0.0
        self.cruise_integral_x = 0.0
        self.last_sent_dx = 0.0
        self.last_sent_dy = 0.0
        self.last_step_time = None

    def step(
        self,
        raw_tx: float,
        raw_ty: float,
        crosshair_c: Tuple[float, float],
        dt: Optional[float] = None,
        target_switched: bool = False,
        enable_ego_comp: bool = True,
        enable_filter: bool = True,
        enable_damping: bool = True,
        enable_unloading_softland: bool = True,
        enable_ff: bool = True,
        enable_lead: bool = True,
        enable_cruise_i: bool = True,
    ) -> Tuple[float, float]:
        cfg = self.cfg
        now = time.perf_counter()
        dt_ref = max(1e-4, float(cfg.dt_ref))
        if dt is None or not math.isfinite(dt) or dt <= 1e-5:
            dt = dt_ref if self.last_step_time is None else max(1e-4, min(0.05, now - self.last_step_time))
        else:
            dt = max(1e-4, min(0.05, float(dt)))
        self.last_step_time = now

        if target_switched:
            self.filter_x.reset()
            self.filter_y.reset()
            self.cruise_integral_x = 0.0
            self.prev_err_x = None
            self.prev_err_y = None
            self.lead_smoothed_x = 0.0
            self.lead_smoothed_y = 0.0
            self.last_sent_dx = 0.0
            self.last_sent_dy = 0.0

        # Shift filter state anchor into current camera frame to decouple ego-motion velocity
        if enable_ego_comp:
            if self.filter_x.x_prev is not None:
                self.filter_x.x_prev -= self.last_sent_dx
            if self.filter_y.x_prev is not None:
                self.filter_y.x_prev -= self.last_sent_dy

        if enable_filter:
            filt_x, filt_vx = self.filter_x.filter(raw_tx, dt=dt, now=now)
            filt_y, filt_vy = self.filter_y.filter(raw_ty, dt=dt, now=now)
        else:
            filt_x, filt_y = raw_tx, raw_ty
            filt_vx, filt_vy = 0.0, 0.0

        vx_clean = 0.0 if abs(filt_vx) < 30.0 else filt_vx
        vy_clean = 0.0 if abs(filt_vy) < 45.0 else filt_vy

        cx, cy = crosshair_c
        raw_err_dist = float(math.hypot(filt_x - cx, filt_y - cy))
        prox_factor = max(0.20, min(1.0, raw_err_dist / 22.0))

        if raw_err_dist > 45.0:
            lead_snap_gate = 0.25
        elif raw_err_dist < 30.0:
            lead_snap_gate = 1.0
        else:
            lead_snap_gate = 0.25 + 0.75 * (45.0 - raw_err_dist) / 15.0

        if enable_lead and cfg.lead_delay_ms > 0.0:
            lead_sec = cfg.lead_delay_ms / 1000.0
            raw_lead_x = max(-35.0, min(35.0, vx_clean * lead_sec)) * prox_factor * lead_snap_gate
            raw_lead_y = max(-5.0, min(5.0, vy_clean * lead_sec * 0.15)) * prox_factor * lead_snap_gate

            ema_base = min(0.95, max(0.10, cfg.smooth_ema))
            ema_w = math.pow(ema_base, dt / dt_ref)
            self.lead_smoothed_x = ema_w * self.lead_smoothed_x + (1.0 - ema_w) * raw_lead_x
            self.lead_smoothed_y = ema_w * self.lead_smoothed_y + (1.0 - ema_w) * raw_lead_y

            pred_x = filt_x + self.lead_smoothed_x
            pred_y = filt_y + self.lead_smoothed_y
        else:
            self.lead_smoothed_x = 0.0
            self.lead_smoothed_y = 0.0
            pred_x, pred_y = filt_x, filt_y

        err_x = pred_x - cx
        err_y = pred_y - cy
        err_dist = float(math.hypot(err_x, err_y))

        if self.prev_err_x is None:
            self.prev_err_x = err_x
            self.prev_err_y = err_y

        if err_dist < cfg.dead_zone_px:
            # Execute leaky integral decay inside deadband before returning zero output
            if enable_cruise_i:
                self.cruise_integral_x *= math.pow(0.92, dt / 0.001)
            self.prev_err_x = err_x
            self.prev_err_y = err_y
            self.last_sent_dx = 0.0
            self.last_sent_dy = 0.0
            return 0.0, 0.0

        count_scale = 2.51 / max(0.1, cfg.sensitivity)
        self.last_count_scale = count_scale

        gain_near = max(0.10, cfg.p_gain_base * 0.90)
        gain_far = max(0.08, cfg.p_gain_base * 0.70)
        if err_dist >= cfg.sched_far_px:
            gain = gain_far
        elif err_dist <= cfg.sched_near_px:
            gain = gain_near
        else:
            t_sched = (err_dist - cfg.sched_near_px) / max(1e-4, cfg.sched_far_px - cfg.sched_near_px)
            gain = gain_near + t_sched * (gain_far - gain_near)

        p_scale = dt / dt_ref
        d_scale = dt_ref / dt
        p_pull_x = (gain * p_scale * err_x) * count_scale
        p_pull_y = (gain * p_scale * err_y) * count_scale

        d_err_x = (err_x - self.prev_err_x) * count_scale
        d_err_y = (err_y - self.prev_err_y) * count_scale
        self.prev_err_x = err_x
        self.prev_err_y = err_y

        if enable_cruise_i and err_dist < 60.0:
            self.cruise_integral_x += (err_x * count_scale) * dt
            self.cruise_integral_x = max(-15.0, min(15.0, self.cruise_integral_x))
            if abs(err_x) < 1.5:
                self.cruise_integral_x *= math.pow(0.92, dt / 0.001)
        else:
            self.cruise_integral_x = 0.0
        i_term_x = cfg.ki_gain * self.cruise_integral_x if enable_cruise_i else 0.0

        if enable_damping:
            if enable_unloading_softland:
                is_chasing_x = (err_x * d_err_x < 0) and (abs(err_x) > 3.0)
                is_chasing_y = (err_y * d_err_y < 0) and (abs(err_y) > 3.0)
                braking_x = max(0.0, min(1.0, -(err_x * d_err_x) / (abs(err_x) * max(1.0, abs(d_err_x))))) if abs(err_x) > 1e-4 else 0.0
                braking_y = max(0.0, min(1.0, -(err_y * d_err_y) / (abs(err_y) * max(1.0, abs(d_err_y))))) if abs(err_y) > 1e-4 else 0.0

                d_factor_x = (0.35 + braking_x * 0.50) * (0.40 if is_chasing_x else 1.0)
                d_factor_y = (0.55 + braking_y * 0.45) * (0.50 if is_chasing_y else 1.0)
            else:
                d_factor_x, d_factor_y = 0.45, 0.45

            d_term_x = (cfg.d_damping * d_scale * d_factor_x) * d_err_x
            d_term_y = (cfg.d_damping * d_scale * d_factor_y) * d_err_y

            if enable_unloading_softland:
                if err_x * d_term_x < 0 and abs(d_term_x) > abs(p_pull_x) * 0.85:
                    d_term_x = -math.copysign(abs(p_pull_x) * 0.85, err_x)
                if err_y * d_term_y < 0 and abs(d_term_y) > abs(p_pull_y) * 0.85:
                    d_term_y = -math.copysign(abs(p_pull_y) * 0.85, err_y)
        else:
            d_term_x, d_term_y = 0.0, 0.0

        if enable_ff and cfg.lead_delay_ms > 0.0:
            kf = 0.30 * prox_factor * lead_snap_gate
            ff_x = max(-10.0, min(10.0, kf * (vx_clean * count_scale) * dt))
            ff_y = 0.0
        else:
            ff_x, ff_y = 0.0, 0.0

        raw_step_x = p_pull_x + i_term_x + d_term_x + ff_x
        raw_step_y = p_pull_y + d_term_y + ff_y

        step_x = max(0.0, raw_step_x) if err_x > 0 else min(0.0, raw_step_x)
        step_y = max(0.0, raw_step_y) if err_y > 0 else min(0.0, raw_step_y)

        eff_err_x = err_x * count_scale
        v_step_x = (vx_clean * count_scale) * dt
        if eff_err_x > 1e-4:
            step_x = max(0.0, min(step_x, eff_err_x + max(0.0, v_step_x) * 1.15))
        elif eff_err_x < -1e-4:
            step_x = min(0.0, max(step_x, eff_err_x + min(0.0, v_step_x) * 1.15))

        max_step_dist = cfg.u_max_px * count_scale
        step_x, step_y = vector_actuator_saturation(step_x, step_y, max_step_dist)

        self.last_sent_dx = step_x / max(1e-4, count_scale)
        self.last_sent_dy = step_y / max(1e-4, count_scale)
        return step_x, step_y


# ======================================================================
# 3. Asynchronous 1000 Hz Servoing Control Worker (AsyncControlWorker)
# ======================================================================
class AsyncControlWorker:
    def __init__(
        self,
        controller: Optional[SmoothSSVSController] = None,
        cfg: Optional[SSVSConfig] = None,
        crosshair_c: Tuple[float, float] = (160.0, 160.0),
        dispatch_fn: Optional[callable] = None,
        synthetic_latency_injection_ms: float = 0.0,
        blocking_wait_ms: float = 0.0,
    ):
        self.cfg = cfg or SSVSConfig()
        self.controller = controller or SmoothSSVSController(self.cfg)
        self.crosshair_c = crosshair_c
        self.dispatch_fn = dispatch_fn
        self.synthetic_latency_injection_ms = float(synthetic_latency_injection_ms)
        self.blocking_wait_ms = float(blocking_wait_ms)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._timer_period_active = False

        self._obs_target_x: Optional[float] = None
        self._obs_target_y: Optional[float] = None
        self._obs_timestamp: float = 0.0
        self._has_new_observation: bool = False

        self._view_x: float = 0.0
        self._view_y: float = 0.0
        self._obs_view_x: float = 0.0
        self._obs_view_y: float = 0.0

        self.res_x: float = 0.0
        self.res_y: float = 0.0

        self.last_error: Optional[Exception] = None
        self._latency_queue: Deque[Tuple[float, float, float, float, float]] = deque()

    def reset(self) -> None:
        with self._lock:
            self.controller.reset()
            self._obs_target_x = None
            self._obs_target_y = None
            self._obs_timestamp = 0.0
            self._has_new_observation = False
            self._view_x = 0.0
            self._view_y = 0.0
            self._obs_view_x = 0.0
            self._obs_view_y = 0.0
            self.res_x = 0.0
            self.res_y = 0.0
            self.last_error = None
            self._latency_queue.clear()

    def update_observation(self, target_x: float, target_y: float, timestamp: Optional[float] = None) -> None:
        now = time.perf_counter() if timestamp is None else timestamp
        with self._lock:
            if self.synthetic_latency_injection_ms > 0.0:
                self._latency_queue.append((now, target_x, target_y, self._view_x, self._view_y))
            else:
                self._obs_target_x = target_x
                self._obs_target_y = target_y
                self._obs_timestamp = now
                self._has_new_observation = True
                self._obs_view_x = self._view_x
                self._obs_view_y = self._view_y

    def _flush_latency_queue(self, now: float) -> None:
        delay_sec = self.synthetic_latency_injection_ms / 1000.0
        while self._latency_queue and (now - self._latency_queue[0][0]) >= delay_sec:
            _, tx, ty, vx_obs, vy_obs = self._latency_queue.popleft()
            self._obs_target_x = tx
            self._obs_target_y = ty
            self._obs_timestamp = now
            self._has_new_observation = True
            self._obs_view_x = vx_obs
            self._obs_view_y = vy_obs

    def tick(self, dt: float = 0.001) -> Tuple[int, int]:
        now = time.perf_counter()
        with self._lock:
            if self.synthetic_latency_injection_ms > 0.0:
                self._flush_latency_queue(now)

            if self._obs_target_x is None or self._obs_target_y is None:
                return 0, 0

            virt_tx = self._obs_target_x - (self._view_x - self._obs_view_x)
            virt_ty = self._obs_target_y - (self._view_y - self._obs_view_y)

            step_x_cnt, step_y_cnt = self.controller.step(
                virt_tx, virt_ty, self.crosshair_c, dt=dt
            )

            raw_x = step_x_cnt + self.res_x
            raw_y = step_y_cnt + self.res_y

            cmd_x = math.trunc(raw_x) if abs(raw_x) >= 1.0 else 0
            cmd_y = math.trunc(raw_y) if abs(raw_y) >= 1.0 else 0

            self.res_x = raw_x - cmd_x
            self.res_y = raw_y - cmd_y

            count_scale = max(1e-4, self.controller.last_count_scale)
            disp_px_x = cmd_x / count_scale
            disp_px_y = cmd_y / count_scale

            self.controller.last_sent_dx = disp_px_x
            self.controller.last_sent_dy = disp_px_y

            self._view_x += disp_px_x
            self._view_y += disp_px_y

        if self.dispatch_fn and (cmd_x != 0 or cmd_y != 0):
            self.dispatch_fn(cmd_x, cmd_y)

        if self.blocking_wait_ms > 0.0:
            time.sleep(self.blocking_wait_ms / 1000.0)

        return cmd_x, cmd_y

    def start(self) -> None:
        if self._running:
            return
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            self._timer_period_active = True
        except Exception:
            self._timer_period_active = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AsyncSSVSControlWorker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._thread = None
        if self._timer_period_active:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
            self._timer_period_active = False

    def _loop(self) -> None:
        dt_target = 0.001
        last_t = time.perf_counter()
        while self._running:
            t0 = time.perf_counter()
            dt_actual = max(1e-4, min(0.01, t0 - last_t))
            last_t = t0

            try:
                self.tick(dt=dt_actual)
            except Exception as exc:
                self.last_error = exc
                self._running = False
                break

            elapsed = time.perf_counter() - t0
            sleep_time = dt_target - elapsed
            if sleep_time > 0.0002:
                time.sleep(sleep_time)


# ======================================================================
# 4. Closed-Loop Z-Domain Pole & Stability Analyzer (Section 4.3)
# ======================================================================
def analyze_closed_loop_poles(kp_eff: float = 0.085, kd_eff: float = 0.835) -> Dict[str, object]:
    b = -(1.0 - kp_eff - kd_eff)
    c = -kd_eff
    discriminant = b * b - 4.0 * c

    if discriminant >= 0.0:
        sqrt_d = math.sqrt(discriminant)
        z1 = (-b + sqrt_d) / 2.0
        z2 = (-b - sqrt_d) / 2.0
        rho = max(abs(z1), abs(z2))
        tau_ms = -1.0 / math.log(abs(z1)) if 0 < abs(z1) < 1 else float("inf")
        z1_repr, z2_repr = round(z1, 4), round(z2, 4)
    else:
        sqrt_d = cmath.sqrt(discriminant)
        z1 = (-b + sqrt_d) / 2.0
        z2 = (-b - sqrt_d) / 2.0
        rho = max(abs(z1), abs(z2))
        tau_ms = -1.0 / math.log(rho) if 0 < rho < 1 else float("inf")
        z1_repr = f"{z1.real:.4f}{z1.imag:+.4f}j"
        z2_repr = f"{z2.real:.4f}{z2.imag:+.4f}j"

    return {
        "kp_eff": kp_eff,
        "kd_eff": kd_eff,
        "discriminant": round(discriminant, 4),
        "z1": z1_repr,
        "z2": z2_repr,
        "spectral_radius": round(rho, 4),
        "tau_dominant_ms": round(tau_ms, 2),
        "is_overdamped_real": discriminant > 0.0 and rho < 1.0,
    }


if __name__ == "__main__":
    print("=== SSVS Characteristic Pole Verification (Section 4.3) ===")
    res_bench = analyze_closed_loop_poles(kp_eff=0.052, kd_eff=0.483)
    print(f"[Benchmark Profile     Kp=0.36, Kd=0.22]: {res_bench}")
    res_std = analyze_closed_loop_poles(kp_eff=0.063, kd_eff=0.700)
    print(f"[Standard Profile      Kp=0.44, Kd=0.32]: {res_std}")
    res_high = analyze_closed_loop_poles(kp_eff=0.085, kd_eff=0.835)
    print(f"[High-Response Profile Kp=0.59, Kd=0.38]: {res_high}")

    print("\n=== Deadband Integral Decay & Zero-dt Guard Verification ===")
    ctrl = SmoothSSVSController()
    ctrl.cruise_integral_x = 5.0
    ctrl.step(160.1, 160.0, (160.0, 160.0), dt=0.0)
    assert ctrl.cruise_integral_x < 5.0, "Deadband leaky decay failed!"
    print(f"[Verified] Deadband integral decayed 5.0 -> {ctrl.cruise_integral_x:.2f} with dt=0.0 guard.")
