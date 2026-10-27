"""
SSVS Synthetic Trajectory Benchmark & Ablation Harness
======================================================
Strictly reproduces and quantitatively validates the 5-term Gain-Scheduled
Screen-Space Visual Servoing (SSVS) control law across synthetic kinematic
benchmarks (Step, Ramp, Harmonic, and Jitter).

Targeting Springer JRTIP empirical ablation matrix (M0 ~ M9).
Produces Table 4 data (and Section 6.2 delay sweep) and publication-grade figures.
"""

import sys
import math
import time
import json
import os
from collections import deque
from typing import Dict, List, Tuple
import numpy as np

from ssvs_core import OneEuroFilter

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ======================================================================
# 1. Linear Constant-Velocity Kalman Filter (For M9 Baseline Comparison)
# ======================================================================
class LinearCVKalmanFilter:
    """
    Standard 2-state Constant-Velocity Kalman Filter [err, v_err]^T.
    Lacks the nonlinear 800 px/s velocity clamp of 1-Euro, demonstrating
    innovation velocity overshoot under 120 px step discontinuities.
    Unrolled 2x2 scalar arithmetic avoids per-step array allocation overhead.
    """
    def __init__(self, q_acc: float = 3600.0, r_std: float = 1.80):
        self.q_acc = float(q_acc)
        self.r_var = float(r_std) ** 2
        self.x0 = 0.0
        self.x1 = 0.0
        self.P00 = 10.0
        self.P01 = 0.0
        self.P10 = 0.0
        self.P11 = 10.0
        self.initialized = False

    def reset(self):
        self.x0 = 0.0
        self.x1 = 0.0
        self.P00 = 10.0
        self.P01 = 0.0
        self.P10 = 0.0
        self.P11 = 10.0
        self.initialized = False

    def filter(self, z: float, dt: float) -> Tuple[float, float]:
        if dt is None or not math.isfinite(dt) or dt <= 1e-5:
            dt = 0.001
        else:
            dt = max(1e-4, min(0.1, float(dt)))
        if not self.initialized:
            self.x0 = float(z)
            self.x1 = 0.0
            self.initialized = True
            return self.x0, 0.0

        # Predict step (unrolled 2x2 state & covariance propagation)
        x0 = self.x0 + dt * self.x1
        x1 = self.x1
        dt2 = dt * dt
        dt3 = 0.5 * dt2 * dt
        dt4 = 0.25 * dt2 * dt2
        q = self.q_acc

        P00 = self.P00 + dt * (self.P10 + self.P01) + dt2 * self.P11 + dt4 * q
        P01 = self.P01 + dt * self.P11 + dt3 * q
        P10 = self.P10 + dt * self.P11 + dt3 * q
        P11 = self.P11 + dt2 * q

        # Measurement update step
        y = z - x0
        s = P00 + self.r_var
        K0 = P00 / s
        K1 = P10 / s
        self.x0 = x0 + K0 * y
        self.x1 = x1 + K1 * y

        self.P00 = (1.0 - K0) * P00
        self.P01 = (1.0 - K0) * P01
        self.P10 = P10 - K1 * P00
        self.P11 = P11 - K1 * P01

        return self.x0, self.x1


# ======================================================================
# 2. Configurable SSVS Controller with Ablation Flags (M0 ~ M9)
# ======================================================================
class AblationController:
    def __init__(
        self,
        config_id: str = "M7",
        p_base: float = 0.36,
        d_base: float = 0.22,
        ki_base: float = 12.0,
        sens: float = 2.50,
        lead_delay_ms: float = 12.4,
        u_max_px: float = 50.0,
    ):
        self.config_id = config_id
        self.p_base = float(p_base)
        self.d_base = float(d_base)
        self.ki_base = float(ki_base)
        self.sens = max(0.1, float(sens))
        self.count_scale = 2.51 / self.sens  # ~1.004 px/count
        self.lead_delay_ms = float(lead_delay_ms)
        self.u_max_px = float(u_max_px)

        self.filter = OneEuroFilter(min_cutoff=0.8, beta=0.020, d_cutoff=1.0)
        self.kf = LinearCVKalmanFilter(q_acc=3600.0, r_std=1.80)
        self.prev_err = None
        self.cruise_integral = 0.0
        self.lead_smoothed = 0.0
        self.last_sent_dx = 0.0
        self.res_x = 0.0  # Sub-count residual accumulator

    def reset(self):
        self.filter.reset()
        self.kf.reset()
        self.prev_err = None
        self.cruise_integral = 0.0
        self.lead_smoothed = 0.0
        self.last_sent_dx = 0.0
        self.res_x = 0.0

    def step(self, raw_err: float, dt: float) -> Tuple[float, float]:
        """
        Calculates actuator displacement count based on config M0 ~ M9.
        raw_err: apparent error in pixels (target - viewport)
        dt: sampling step in seconds (0.001s for 1000Hz)
        Returns: (sent_px, filtered_target_err)
        """
        if dt is None or not math.isfinite(dt) or dt <= 1e-5:
            dt = 0.001
        else:
            dt = max(1e-4, min(0.1, float(dt)))

        cfg = self.config_id

        # --- A. Inverse Ego-Motion Reprojection (M3+) ---
        # Shift filter state anchor by camera movement to isolate true target velocity
        if cfg in ("M3", "M4", "M5", "M6", "M7", "M9"):
            if self.filter.x_prev is not None:
                self.filter.x_prev -= self.last_sent_dx
            if self.kf.initialized:
                self.kf.x0 -= self.last_sent_dx

        # --- B. Perception Filtering ---
        # M0/M8: Direct un-filtered observation | M9: Linear CV-KF | M1-M7: Adaptive 1-Euro Filter
        # (Under 12-step transport delay + zero-order observation hold, cascading 1-Euro or first-order
        #  LPF into classical unconstrained PID erodes phase margin; M8 therefore uses direct observation
        #  paired with stability-tuned integral feedback.)
        if cfg in ("M0", "M8"):
            filt_err = raw_err
            filt_v = 0.0
        elif cfg == "M9":
            filt_err, filt_v = self.kf.filter(raw_err, dt)
        else:
            filt_err, filt_v = self.filter.filter(raw_err, dt)

        # Velocity deadband
        v_clean = 0.0 if abs(filt_v) < 30.0 else filt_v

        # --- C. Lead Horizon Extrapolation (M6, M7, M9) ---
        err_dist = abs(filt_err)
        prox_factor = max(0.20, min(1.0, err_dist / 22.0))
        if err_dist > 45.0:
            lead_snap_gate = 0.25
        elif err_dist < 30.0:
            lead_snap_gate = 1.0
        else:
            lead_snap_gate = 0.25 + 0.75 * (45.0 - err_dist) / 15.0

        if cfg in ("M6", "M7", "M9") and self.lead_delay_ms > 0.0:
            lead_sec = self.lead_delay_ms / 1000.0
            raw_lead = max(-35.0, min(35.0, v_clean * lead_sec)) * prox_factor * lead_snap_gate
            ema_base = 0.70
            ema_w = math.pow(ema_base, dt / 0.00625)
            self.lead_smoothed = ema_w * self.lead_smoothed + (1.0 - ema_w) * raw_lead
            pred_err = filt_err + self.lead_smoothed
        else:
            self.lead_smoothed = 0.0
            pred_err = filt_err

        # --- C2. Near-Field Dead Zone (delta = 0.5 px, Eq. 11) ---
        # Suppresses sub-pixel observation noise micro-stepping; synchronizes error history
        # inside the dead zone to prevent derivative kick upon dead-zone exit.
        if abs(pred_err) < 0.5:
            if cfg == "M7":
                self.cruise_integral *= math.pow(0.92, dt / 0.001)
            if self.prev_err is not None:
                self.prev_err = pred_err
            return 0.0, filt_err

        # --- D. Proportional Branch (u_P) ---
        if cfg == "M8":
            gain_p = self.p_base
        elif cfg == "M0":
            gain_p = self.p_base * 0.85
        else:
            # Piecewise Linear Gain Scheduling (Eq. 12)
            near_thresh, far_thresh = 15.0, 80.0
            gain_near = max(0.10, self.p_base * 0.90)
            gain_far = max(0.08, self.p_base * 0.70)
            if err_dist >= far_thresh:
                gain_p = gain_far
            elif err_dist <= near_thresh:
                gain_p = gain_near
            else:
                t = (err_dist - near_thresh) / (far_thresh - near_thresh)
                gain_p = gain_near + t * (gain_far - gain_near)

        # Rate normalization (DT_REF = 6.25 ms, Section 4.1):
        # Scales proportional gain by dt / DT_REF and derivative single-step difference by DT_REF / dt
        # to maintain invariant closed-loop damping across control sampling rates.
        p_scale = dt / 0.00625
        d_scale = 0.00625 / dt
        u_p = (gain_p * p_scale * pred_err) * self.count_scale

        # --- E. Derivative Damping Branch (u_D) ---
        # In discrete displacement servomechanisms, u_D directly eats Δe (without dividing by Δt)
        if self.prev_err is None:
            self.prev_err = pred_err
        d_err = (pred_err - self.prev_err) * self.count_scale
        self.prev_err = pred_err

        u_d = 0.0
        if cfg == "M8":
            # Classical discrete PID derivative: unnormalized single-step difference without
            # multi-rate DT_REF / dt scaling, illustrating classical frame-coupled discretization.
            u_d = (self.d_base * 0.30) * d_err
        elif cfg in ("M2", "M3", "M9"):
            # Fixed-damping classical derivative (rate-normalized via d_scale = DT_REF / dt)
            u_d = (self.d_base * 0.45 * d_scale) * d_err
        elif cfg in ("M4", "M5", "M6", "M7"):
            # State-dependent chasing unloading & dynamic brake (rate-normalized)
            is_chasing = (pred_err * d_err < 0) and (abs(pred_err) > 3.0)
            braking = max(0.0, min(1.0, -(pred_err * d_err) / (abs(pred_err) * max(1.0, abs(d_err))))) if abs(pred_err) > 1e-4 else 0.0
            d_factor = 0.35 + braking * 0.50
            if is_chasing:
                d_factor *= 0.40  # Release 60% damping during forward chase

            u_d = (self.d_base * d_scale * d_factor) * d_err

            # Soft-landing saturation clamp: |u_D| <= 0.85 * |u_P|
            if pred_err * u_d < 0 and abs(u_d) > abs(u_p) * 0.85:
                u_d = -math.copysign(abs(u_p) * 0.85, pred_err)

        # --- F. Velocity Feedforward (u_ff) ---
        u_ff = 0.0
        if cfg in ("M5", "M6", "M7", "M9"):
            kf = 0.30 * prox_factor * lead_snap_gate
            u_ff = max(-10.0, min(10.0, kf * (v_clean * self.count_scale) * dt))

        # --- G. Integral Branch (u_I) ---
        u_i = 0.0
        if cfg == "M8":
            # Classical unconstrained continuous integrator (stability-domain tuned to 0.80 s^-1
            # with [-50, 50] saturation to stabilize delayed random-walk drift, contrasting with
            # M7's conditional leaky cruise integrator in Eqs. 13-14).
            self.cruise_integral += (pred_err * self.count_scale) * dt
            self.cruise_integral = max(-50.0, min(50.0, self.cruise_integral))
            u_i = 0.80 * self.cruise_integral
        elif cfg == "M7":
            # Conditional Leaky Cruise Integrator
            if abs(pred_err) < 60.0:
                self.cruise_integral += (pred_err * self.count_scale) * dt
                self.cruise_integral = max(-15.0, min(15.0, self.cruise_integral))
                if abs(pred_err) < 1.5:
                    self.cruise_integral *= math.pow(0.92, dt / 0.001)  # Leaky discharge
            else:
                self.cruise_integral = 0.0
            u_i = self.ki_base * self.cruise_integral

        # Total continuous displacement command
        raw_cmd = u_p + u_d + u_ff + u_i

        # Directional brake constraint (M4+)
        if cfg in ("M4", "M5", "M6", "M7"):
            if pred_err > 0:
                cmd = max(0.0, raw_cmd)
            else:
                cmd = min(0.0, raw_cmd)

            # Smart Ramp Barrier: cannot step beyond remaining error
            eff_err = pred_err * self.count_scale
            v_step = (v_clean * self.count_scale) * dt
            if eff_err > 1e-4:
                max_fwd = eff_err + max(0.0, v_step) * 1.15
                cmd = max(0.0, min(cmd, max_fwd))
            elif eff_err < -1e-4:
                min_fwd = eff_err + min(0.0, v_step) * 1.15
                cmd = min(0.0, max(cmd, min_fwd))
        else:
            cmd = raw_cmd

        # Nonlinear Actuator Saturation Operator S_{u_max} (Slew Rate Limiter, Eq. u_k = clip(\tilde{u}_k, -u_max, u_max))
        u_max_px = getattr(self, "u_max_px", 50.0)
        u_max_counts = u_max_px * self.count_scale
        cmd = max(-u_max_counts, min(u_max_counts, cmd))

        # Sub-count residual accumulator & integer truncation
        raw_x = cmd + self.res_x
        out_counts = math.trunc(raw_x) if abs(raw_x) >= 1.0 else 0
        self.res_x = raw_x - out_counts

        sent_px = out_counts / self.count_scale
        self.last_sent_dx = sent_px

        return sent_px, filt_err


# ======================================================================
# 3. Dynamic Environment & Synthetic Trajectory Simulator
# ======================================================================
class SimulationHarness:
    def __init__(
        self,
        dt_ctrl: float = 0.001,           # 1000 Hz actuation
        dt_perc: float = 0.00625,         # 160 Hz perception (~6.25ms)
        transport_delay_ms: float = 12.4, # Measured pipeline delay
        noise_std: float = 1.80,          # Observation centroid jitter
        synthetic_delay_ms: float = 0.0,  # Synthetic transport latency injection \tau_inj (ms)
    ):
        self.dt_ctrl = dt_ctrl
        self.dt_perc = dt_perc
        total_delay = transport_delay_ms + synthetic_delay_ms
        self.delay_steps = int(round((total_delay / 1000.0) / dt_ctrl))
        self.noise_std = noise_std
        self.synthetic_delay_ms = synthetic_delay_ms

    def run_trajectory(
        self,
        controller: AblationController,
        trajectory_type: str = "step",
        duration: float = 2.0,
        seed: int = 42,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes a single closed-loop trajectory simulation.
        trajectory_type: 'step', 'ramp', 'harmonic', 'jitter'
        """
        np.random.seed(seed)
        controller.reset()

        num_steps = int(round(duration / self.dt_ctrl))
        times = np.arange(num_steps, dtype=np.float64) * self.dt_ctrl

        # Target motion ground-truth p_target(t)
        target_pos = np.zeros(num_steps)
        if trajectory_type == "step":
            step_idx = int(0.5 / self.dt_ctrl)
            target_pos[step_idx:] = 120.0
        elif trajectory_type == "ramp":
            step_idx = int(0.5 / self.dt_ctrl)
            target_pos[step_idx:] = 300.0 * (times[step_idx:] - 0.5)
        elif trajectory_type == "harmonic":
            target_pos = 80.0 * np.sin(2.0 * np.pi * 1.2 * times)
        elif trajectory_type == "jitter":
            target_pos[:] = 0.0

        delay_buf = deque([0.0] * max(1, self.delay_steps))

        viewport_pos = 0.0
        apparent_errors = np.zeros(num_steps)
        viewport_positions = np.zeros(num_steps)
        filtered_errors = np.zeros(num_steps)

        last_perc_time = -1.0
        obs_apparent_err = 0.0
        obs_viewport_pos = 0.0

        for k in range(num_steps):
            t = times[k]

            # Physical true instantaneous error
            true_err = target_pos[k] - viewport_pos

            # Transport delay buffer
            delay_buf.append(true_err)
            delayed_err = delay_buf.popleft()

            # Perception update at rate dt_perc
            if last_perc_time < 0 or (t - last_perc_time) >= (self.dt_perc - 1e-6):
                noise = np.random.normal(0.0, self.noise_std)
                obs_apparent_err = delayed_err + noise
                obs_viewport_pos = viewport_pos
                last_perc_time = t

            # Multi-rate Inter-Frame Virtual Ego-Motion Extrapolation (M3+)
            cfg = controller.config_id
            if cfg in ("M3", "M4", "M5", "M6", "M7", "M9"):
                virt_err = obs_apparent_err - (viewport_pos - obs_viewport_pos)
                sent_dx, f_err = controller.step(virt_err, self.dt_ctrl)
            else:
                sent_dx, f_err = controller.step(obs_apparent_err, self.dt_ctrl)

            # Actuator displaces viewport
            viewport_pos += sent_dx

            viewport_positions[k] = viewport_pos
            apparent_errors[k] = target_pos[k] - viewport_pos
            filtered_errors[k] = f_err

        # Compute trajectory-specific quantitative metrics
        metrics = {}
        if trajectory_type == "step":
            step_idx = int(0.5 / self.dt_ctrl)
            t_after = times[step_idx:] - 0.5
            p_after = viewport_positions[step_idx:]

            # Rise time t_90%: time to reach 90% of 120 px (108 px)
            idx_90 = np.where(p_after >= 108.0)[0]
            if len(idx_90) > 0:
                rise_time_ms = float(t_after[idx_90[0]] * 1000.0)
                reached_90 = True
            else:
                rise_time_ms = float("nan")
                reached_90 = False

            # Percentage overshoot %OS
            max_p = np.max(p_after)
            overshoot_pct = max(0.0, (max_p - 120.0) / 120.0) * 100.0
            metrics["t_90"] = rise_time_ms
            metrics["overshoot"] = overshoot_pct
            metrics["reached_90"] = reached_90

        elif trajectory_type == "ramp":
            # Steady-state error e_ss over t in [1.5s, 2.5s]
            ramp_start = int(1.5 / self.dt_ctrl)
            e_ss = np.mean(np.abs(apparent_errors[ramp_start:]))
            metrics["e_ss"] = e_ss

        elif trajectory_type == "harmonic":
            # Harmonic RMS error after 1.0s
            eval_start = int(1.0 / self.dt_ctrl)
            rms_e = np.sqrt(np.mean(apparent_errors[eval_start:] ** 2))
            metrics["rms_e"] = rms_e

        elif trajectory_type == "jitter":
            # Standard deviation of filtered/tracked state under observation noise
            eval_start = int(0.5 / self.dt_ctrl)
            metrics["sigma_jitter"] = np.std(filtered_errors[eval_start:])

        return metrics, times, target_pos, viewport_positions


# ======================================================================
# 4. Monte Carlo Evaluation Campaign (M0 ~ M9, N = 100)
# ======================================================================
def run_ablation_campaign(num_runs: int = 100) -> Dict[str, Dict[str, float]]:
    configs = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
    descriptions = {
        "M0": "Raw Proportional Baseline (No Filter / D / I / FF)",
        "M1": "M0 + Adaptive 1-Euro Low-Pass Filter",
        "M2": "M1 + Fixed-Damping Classical Derivative (u_D)",
        "M3": "M2 + Inverse Ego-Motion Reprojection",
        "M4": "M3 + Chasing Unloading & Soft-Landing Clamping",
        "M5": "M4 + Target Velocity Feedforward (u_ff)",
        "M6": "M5 + Horizon Lead Delay Gating (u_lead)",
        "M7": "Proposed Full Framework (M6 + Conditional Leaky u_I)",
        "M8": "Classical Discrete PID (Unconstrained Integrator)",
        "M9": "Constant-Velocity Kalman Filter (CV-KF + PD)",
    }

    harness = SimulationHarness(dt_ctrl=0.001, dt_perc=0.00625, transport_delay_ms=12.4, noise_std=1.80)

    results = {}
    print(f"=== Starting SSVS Monte Carlo Ablation Campaign (N = {num_runs} runs per config) ===")

    for cfg in configs:
        t0 = time.time()
        t90_list = []
        reached_90_count = 0
        os_list = []
        ess_list = []
        rms_list = []
        jitter_list = []

        ctrl = AblationController(config_id=cfg)

        for run in range(num_runs):
            seed = 1000 + run
            m_step, _, _, _ = harness.run_trajectory(ctrl, "step", duration=2.0, seed=seed)
            m_ramp, _, _, _ = harness.run_trajectory(ctrl, "ramp", duration=2.5, seed=seed)
            m_harm, _, _, _ = harness.run_trajectory(ctrl, "harmonic", duration=3.0, seed=seed)
            m_jit, _, _, _ = harness.run_trajectory(ctrl, "jitter", duration=2.0, seed=seed)

            if m_step["reached_90"] and not math.isnan(m_step["t_90"]):
                t90_list.append(m_step["t_90"])
                reached_90_count += 1
            os_list.append(m_step["overshoot"])
            ess_list.append(m_ramp["e_ss"])
            rms_list.append(m_harm["rms_e"])
            jitter_list.append(m_jit["sigma_jitter"])

        elapsed = time.time() - t0
        censored_t90_count = num_runs - reached_90_count
        reach_90_rate = float(reached_90_count) / float(num_runs) if num_runs > 0 else 0.0
        res = {
            "name": descriptions[cfg],
            "t_90": float(np.mean(t90_list)) if len(t90_list) > 0 else float("nan"),
            "t_90_std": float(np.std(t90_list)) if len(t90_list) > 0 else float("nan"),
            "reach_90_rate": reach_90_rate,
            "censored_t90_count": int(censored_t90_count),
            "overshoot": float(np.mean(os_list)),
            "overshoot_std": float(np.std(os_list)),
            "e_ss": float(np.mean(ess_list)),
            "e_ss_std": float(np.std(ess_list)),
            "rms_e": float(np.mean(rms_list)),
            "rms_e_std": float(np.std(rms_list)),
            "sigma_jitter": float(np.mean(jitter_list)),
            "sigma_jitter_std": float(np.std(jitter_list)),
        }
        results[cfg] = res
        os_tag = f"{res['overshoot']:.1f}% ± {res['overshoot_std']:.1f}%" if res['overshoot'] >= 0.05 else "0.0% ± 0.0%"
        print(f"[{cfg}] t_90={res['t_90']:.1f}±{res['t_90_std']:.1f}ms (reach={reached_90_count}/{num_runs}), %OS={os_tag}, e_ss={res['e_ss']:.2f}±{res['e_ss_std']:.2f}px, RMS_e={res['rms_e']:.2f}±{res['rms_e_std']:.2f}px, jitter={res['sigma_jitter']:.2f}±{res['sigma_jitter_std']:.2f}px ({elapsed:.2f}s)")

    return results


def print_markdown_table(results: Dict[str, Dict[str, float]]):
    print("\n### Generated Table 4 (Ablation Matrix for Paper, Mean ± Std over N=100 runs):\n")
    header = "| ID | Architecture Configuration | Step Rise Time $t_{90\\%}$ (ms) | Step Overshoot $\\%OS$ (%) | Ramp Error $e_{ss}$ (px) | Harmonic Error $\\text{RMS}_e$ (px) | Static Jitter $\\sigma_{\\text{jitter}}$ (px) |"
    sep = "|:---:|---|:---:|:---:|:---:|:---:|:---:|"
    print(header)
    print(sep)
    for cfg, m in results.items():
        bold = "**" if cfg == "M7" else ""
        if m['overshoot'] < 0.05:
            os_str = "**0.0% ± 0.0%**" if cfg == "M7" else "0.0% ± 0.0%"
        else:
            os_str = f"**{m['overshoot']:.1f}% ± {m['overshoot_std']:.1f}%**" if cfg == "M7" else f"{m['overshoot']:.1f}% ± {m['overshoot_std']:.1f}%"
        row = f"| **{cfg}** | {bold}{m['name']}{bold} | {m['t_90']:.1f} ± {m['t_90_std']:.1f} | {os_str} | {m['e_ss']:.2f} ± {m['e_ss_std']:.2f} | {m['rms_e']:.2f} ± {m['rms_e_std']:.2f} | {m['sigma_jitter']:.2f} ± {m['sigma_jitter_std']:.2f} |"
        print(row)


def run_delay_margin_sweep(max_injected_ms: int = 16) -> Dict[int, Dict[str, float]]:
    """
    Evaluates closed-loop empirical delay degradation characteristics (Section 6.2)
    by injecting synthetic transport latency \tau_inj in {0, 2, 4, ..., 16} ms.
    """
    print("\n### Synthetic Transport Delay Injection & Degradation Sweep (tau_inj in [0, 16] ms):\n")
    results = {}
    delays = list(range(0, max_injected_ms + 1, 2))
    header = "| Injected $\\tau_{\\text{inj}}$ (ms) | Nominal Total $\\tau_{\\text{total}}$ (ms) | Discrete Buffer Steps | Step Rise Time $t_{90\\%}$ (ms) | Step Overshoot $\\%OS$ (%) | Harmonic Tracking $\\text{RMS}_e$ (px) |"
    sep = "|:---:|:---:|:---:|:---:|:---:|:---:|"
    print(header)
    print(sep)
    for d in delays:
        h = SimulationHarness(dt_ctrl=0.001, dt_perc=0.00625, transport_delay_ms=12.4, noise_std=1.80, synthetic_delay_ms=float(d))
        ctrl = AblationController("M7")
        s, _, _, _ = h.run_trajectory(ctrl, "step", duration=2.0, seed=42)
        hm, _, _, _ = h.run_trajectory(ctrl, "harmonic", duration=3.0, seed=42)
        valid_t90 = s["reached_90"] and not math.isnan(s["t_90"])
        results[d] = {
            "total_delay_ms": round(12.4 + d, 1),
            "delay_steps": h.delay_steps,
            "t_90": round(s["t_90"], 1) if valid_t90 else float("nan"),
            "reached_90": bool(s["reached_90"]),
            "overshoot": round(s["overshoot"], 1),
            "rms_e": round(hm["rms_e"], 2),
        }
        t90_str = f"{s['t_90']:.1f}" if valid_t90 else "NaN (censored)"
        print(f"| +{d} ms | {12.4 + d:.1f} ms | {h.delay_steps} ({h.delay_steps:.1f} ms) | {t90_str} | {s['overshoot']:.1f}% | {hm['rms_e']:.2f} |")
    return results


def plot_trajectory_figures(harness: SimulationHarness, save_dir: str):
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 2: Step Response Tracking (M0 vs M8 vs M7)
    plt.figure(figsize=(9, 4.8), dpi=300)
    ctrl_m0 = AblationController("M0")
    ctrl_m8 = AblationController("M8")
    ctrl_m7 = AblationController("M7")

    _, times, tgt, pos_m0 = harness.run_trajectory(ctrl_m0, "step", duration=1.2, seed=42)
    _, _, _, pos_m8 = harness.run_trajectory(ctrl_m8, "step", duration=1.2, seed=42)
    _, _, _, pos_m7 = harness.run_trajectory(ctrl_m7, "step", duration=1.2, seed=42)

    t_ms = (times - 0.5) * 1000.0  # Align step at t = 0 ms
    mask = (t_ms >= -20) & (t_ms <= 300)

    plt.plot(t_ms[mask], tgt[mask], 'k--', linewidth=1.5, label='Target Ground Truth (120 px Step)')
    plt.plot(t_ms[mask], pos_m0[mask], color='#94a3b8', linestyle=':', linewidth=1.8, label='M0: Raw Proportional Baseline')
    plt.plot(t_ms[mask], pos_m8[mask], color='#ef4444', linestyle='-.', linewidth=2.0, label='M8: Classical Discrete PID (Integral Windup Overshoot)')
    plt.plot(t_ms[mask], pos_m7[mask], color='#10b981', linewidth=2.5, label='M7: Proposed SSVS (Monotonic Soft-Landing)')

    plt.axhline(120.0, color='gray', linestyle='--', alpha=0.4)
    plt.axvline(0.0, color='gray', linestyle='--', alpha=0.4)
    plt.title("Closed-Loop Step Response Comparison ($120\\text{ px}$ Displacement, seed=42)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Elapsed Time from Step Onset (ms)", fontsize=11)
    plt.ylabel("Viewport Displacement (px)", fontsize=11)
    plt.xlim(-10, 250)
    plt.ylim(-5, 180)
    plt.legend(loc='lower right', frameon=True, fontsize=9.5)
    plt.tight_layout()

    fig2_path = os.path.join(save_dir, "fig2_step_response.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"[Plot] Saved Figure 2 to {fig2_path}")

    # Figure 3: Dynamic Ramp and Harmonic Tracking (M4 vs M7)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

    # Left: Dynamic Ramp Tracking
    ctrl_m4 = AblationController("M4")
    _, times_r, tgt_r, pos_m4_r = harness.run_trajectory(ctrl_m4, "ramp", duration=2.0, seed=42)
    _, _, _, pos_m7_r = harness.run_trajectory(ctrl_m7, "ramp", duration=2.0, seed=42)
    t_r_ms = times_r * 1000.0
    mask_r = (t_r_ms >= 400) & (t_r_ms <= 1500)

    err_m4_r = tgt_r - pos_m4_r
    err_m7_r = tgt_r - pos_m7_r

    ax1.plot(t_r_ms[mask_r], err_m4_r[mask_r], color='#f59e0b', linestyle='--', linewidth=1.8, label='M4: PD (Type-1 Finite Lag)')
    ax1.plot(t_r_ms[mask_r], err_m7_r[mask_r], color='#10b981', linewidth=2.2, label='M7: Proposed (Cruise Integral)')
    ax1.axhline(0.0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title("(a) Dynamic Ramp Tracking Error ($v_0 = 300\\text{ px/s}$)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Simulation Time (ms)", fontsize=10.5)
    ax1.set_ylabel("Tracking Error $e(t)$ (px)", fontsize=10.5)
    ax1.set_ylim(-3, 15)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)

    # Right: Harmonic Sine Tracking
    _, times_h, tgt_h, pos_m4_h = harness.run_trajectory(ctrl_m4, "harmonic", duration=2.5, seed=42)
    _, _, _, pos_m7_h = harness.run_trajectory(ctrl_m7, "harmonic", duration=2.5, seed=42)
    t_h_ms = times_h * 1000.0
    mask_h = (t_h_ms >= 1000) & (t_h_ms <= 2200)

    ax2.plot(t_h_ms[mask_h], tgt_h[mask_h], 'k:', linewidth=1.5, label='Target Ground Truth ($f = 1.2\\text{ Hz}$)')
    ax2.plot(t_h_ms[mask_h], pos_m4_h[mask_h], color='#f59e0b', linestyle='--', linewidth=1.8, label='M4: PD (Phase Lag)')
    ax2.plot(t_h_ms[mask_h], pos_m7_h[mask_h], color='#10b981', linewidth=2.2, label='M7: Proposed (Lead-Ahead)')
    ax2.set_title("(b) Harmonic Trajectory Tracking ($A = 80\\text{ px}, f = 1.2\\text{ Hz}$)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Simulation Time (ms)", fontsize=10.5)
    ax2.set_ylabel("Viewport Position (px)", fontsize=10.5)
    ax2.legend(loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    fig3_path = os.path.join(save_dir, "fig3_ramp_harmonic.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"[Plot] Saved Figure 3 to {fig3_path}")


if __name__ == "__main__":
    harness = SimulationHarness(dt_ctrl=0.001, dt_perc=0.00625, transport_delay_ms=12.4, noise_std=1.80)
    benchmark_results = run_ablation_campaign(num_runs=100)
    print_markdown_table(benchmark_results)

    # Run synthetic transport delay injection & delay margin sweep (Section 6.2)
    delay_sweep_results = run_delay_margin_sweep(max_injected_ms=16)

    # Save quantitative results JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Quantitative ablation benchmarks saved to {out_path}")

    # Generate figures
    fig_dir = os.path.join(script_dir, "figures")
    plot_trajectory_figures(harness, fig_dir)
