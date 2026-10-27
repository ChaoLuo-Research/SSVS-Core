# -*- coding: utf-8 -*-
"""
generate_extra_figures_and_m9.py
================================================================================
Generates publication-grade figures (Fig. 2, Fig. 3, and Fig. 6 in manuscript.tex;
stored as fig4_latency_roi_pareto.png, fig5_m1_aliasing_and_zpoles.png, and
fig6_pareto_and_sensitivity.png) by directly invoking the unified SimulationHarness
and AblationController from bench_trajectory.py.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from bench_trajectory import SimulationHarness, AblationController
from ssvs_core import analyze_closed_loop_poles


def plot_fig4_latency_and_roi_pareto(save_dirs):
    """
    Generates fig4_latency_roi_pareto.png:
    (a) Parametric log-normal latency model (N = 10,000 synthetic samples parameterized
        by the calibrated Table 1 hardware telemetry means and standard deviations).
    (b) Calibrated RTX 3080 DirectML multi-scale ROI latency & throughput trade-off.
    """
    rng = np.random.default_rng(2026)
    n_samples = 10000

    def _sample_stage(mean_val, std_val):
        sigma2 = math.log(1.0 + (std_val / mean_val) ** 2)
        mu = math.log(mean_val) - 0.5 * sigma2
        return rng.lognormal(mean=mu, sigma=math.sqrt(sigma2), size=n_samples)

    s_cap = _sample_stage(2.12, 0.38)
    s_inf = _sample_stage(8.24, 0.76)
    s_post = _sample_stage(0.58, 0.12)
    s_ctrl = _sample_stage(0.28, 0.05)
    s_act = _sample_stage(1.18, 0.21)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

    data = [s_cap, s_inf, s_post, s_ctrl, s_act]
    labels = [
        "Stage 1\nCapture\n(2.12 ms)",
        "Stage 2\nDirectML\n(8.24 ms)",
        "Stage 3\nPost-Proc\n(0.58 ms)",
        "Stage 4\nControl\n(0.28 ms)",
        "Stage 5\nActuator\n(1.18 ms)",
    ]
    bp = ax1.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    colors = ["#94a3b8", "#3b82f6", "#10b981", "#f59e0b", "#64748b"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax1.set_title("(a) Parametric Log-Normal Stage Latency Model ($N = 10,000$ Synthetic Samples)", fontsize=10.2, fontweight="bold")
    ax1.set_ylabel("Modeled Stage Latency (ms)", fontsize=10.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    res = [256, 320, 416, 512, 640]
    inf_ms = [5.82, 8.24, 11.45, 15.10, 21.30]
    fps_ceil = [1000.0 / m for m in inf_ms]

    ax2_r = ax2.twinx()
    l1 = ax2.plot(res, inf_ms, "-o", color="#2563eb", linewidth=2.2, markersize=7, label=r"Calibrated DirectML $\tau_{\mathrm{inf}}$ (ms)")
    l2 = ax2_r.plot(res, fps_ceil, "--s", color="#10b981", linewidth=2.0, markersize=6.5, label="Throughput Ceiling (FPS)")
    ax2.scatter([320], [8.24], s=160, facecolors="none", edgecolors="#dc2626", linewidths=2.5, zorder=6, label="Selected Default (320x320)")

    ax2.set_title("(b) Calibrated Multi-Scale ROI Resolution vs. Latency & Throughput", fontsize=10.2, fontweight="bold")
    ax2.set_xlabel("Isotropic ROI Tensor Resolution $S \\times S$ (px)", fontsize=10.5)
    ax2.set_ylabel("Mean Inference Latency $\\tau_{\\mathrm{inf}}$ (ms)", color="#2563eb", fontsize=10.5)
    ax2_r.set_ylabel("Perceptual Throughput Ceiling (FPS)", color="#10b981", fontsize=10.5)
    ax2.set_xticks(res)
    ax2.grid(True, linestyle="--", alpha=0.4)

    lines = l1 + l2
    lbls = [l.get_label() for l in lines]
    ax2.legend(lines, lbls, loc="upper center", frameon=True, fontsize=8.8)

    plt.tight_layout()
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, "fig4_latency_roi_pareto.png"))
    plt.close()


def plot_fig5_aliasing_and_zpoles(save_dirs):
    """
    Generates fig5_m1_aliasing_and_zpoles.png using live closed-loop step trajectories
    from SimulationHarness (M1 vs M3) and exact Z-domain characteristic roots.
    """
    harness = SimulationHarness(dt_ctrl=0.001, dt_perc=0.00625, transport_delay_ms=12.4, noise_std=1.80)
    ctrl_m1 = AblationController("M1")
    ctrl_m3 = AblationController("M3")

    _, times, tgt, pos_m1 = harness.run_trajectory(ctrl_m1, "step", duration=1.2, seed=42)
    _, _, _, pos_m3 = harness.run_trajectory(ctrl_m3, "step", duration=1.2, seed=42)

    t_ms = (times - 0.5) * 1000.0
    mask = (t_ms >= -10) & (t_ms <= 350)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    ax1.plot(t_ms[mask], tgt[mask], "k--", linewidth=1.4, label="Target Step (120 px)")
    ax1.plot(t_ms[mask], pos_m1[mask], color="#dc2626", linestyle="-.", linewidth=1.9, label="M1: 1-Euro Without Ego-Comp (Aliasing Ringing)")
    ax1.plot(t_ms[mask], pos_m3[mask], color="#10b981", linewidth=2.3, label="M3: 1-Euro + Inverse Ego-Motion Reprojection")
    ax1.set_title("(a) Simulated M1 Ego-Motion Aliasing vs. M3 Decoupling (seed=42)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Elapsed Time from Step Onset (ms)", fontsize=10.5)
    ax1.set_ylabel("Viewport Displacement (px)", fontsize=10.5)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1.2, alpha=0.6, label="Schur-Cohn Unit Circle ($|z|=1$)")
    ax2.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color="gray", linewidth=0.8, alpha=0.5)

    p_bench = analyze_closed_loop_poles(0.052, 0.483)
    p_std = analyze_closed_loop_poles(0.063, 0.700)
    p_high = analyze_closed_loop_poles(0.085, 0.835)

    ax2.plot([p_bench["z1"], p_bench["z2"]], [0, 0], "s", color="#2563eb", markersize=8.5,
             label=f"Benchmark Poles ($z_1={p_bench['z1']:.4f}, z_2={p_bench['z2']:.4f}$)")
    ax2.plot([p_std["z1"], p_std["z2"]], [0, 0], "^", color="#f59e0b", markersize=8.5,
             label=f"Standard Poles ($z_1={p_std['z1']:.4f}, z_2={p_std['z2']:.4f}$)")
    ax2.plot([p_high["z1"], p_high["z2"]], [0, 0], "o", color="#10b981", markersize=9.0,
             label=f"High-Response Poles ($z_1={p_high['z1']:.4f}, z_2={p_high['z2']:.4f}$)")

    ax2.set_title("(b) Nominal Linearized $\\mathcal{Z}$-Domain Characteristic Pole Locus", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Real Axis $\\mathrm{Re}(z)$", fontsize=10.5)
    ax2.set_ylabel("Imaginary Axis $\\mathrm{Im}(z)$", fontsize=10.5)
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(loc="lower left", frameon=True, fontsize=8.2)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(os.path.join(d, "fig5_m1_aliasing_and_zpoles.png"))
    plt.close()


def plot_fig6_pareto_and_contour(save_dirs, n_sweep_runs: int = 25):
    """
    Generates fig6_pareto_and_sensitivity.png using unified trajectory durations
    (step=2.0s, harmonic=3.0s) and direct array indexing for the d_cutoff=1.0 default point.
    """
    harness = SimulationHarness(dt_ctrl=0.001, dt_perc=0.00625, transport_delay_ms=12.4, noise_std=1.80)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    # Live sweep of 1-Euro derivative cutoff frequency d_cutoff in M7 using unified 2.0s/3.0s durations
    d_cutoffs = [0.4, 0.7, 1.0, 1.5, 2.5, 4.0]
    harm_rms_list, step_os_list = [], []
    for dc in d_cutoffs:
        ctrl = AblationController("M7")
        ctrl.filter.d_cutoff = dc
        os_runs = [harness.run_trajectory(ctrl, "step", duration=2.0, seed=1000 + r)[0]["overshoot"] for r in range(n_sweep_runs)]
        rms_runs = [harness.run_trajectory(ctrl, "harmonic", duration=3.0, seed=1000 + r)[0]["rms_e"] for r in range(n_sweep_runs)]
        step_os_list.append(float(np.mean(os_runs)))
        harm_rms_list.append(float(np.mean(rms_runs)))

    # Compute M3 and M8 reference points under the exact same n_sweep_runs protocol
    ctrl_m3 = AblationController("M3")
    m3_os = float(np.mean([harness.run_trajectory(ctrl_m3, "step", duration=2.0, seed=1000 + r)[0]["overshoot"] for r in range(n_sweep_runs)]))
    m3_rms = float(np.mean([harness.run_trajectory(ctrl_m3, "harmonic", duration=3.0, seed=1000 + r)[0]["rms_e"] for r in range(n_sweep_runs)]))

    ctrl_m8 = AblationController("M8")
    m8_os = float(np.mean([harness.run_trajectory(ctrl_m8, "step", duration=2.0, seed=1000 + r)[0]["overshoot"] for r in range(n_sweep_runs)]))
    m8_rms = float(np.mean([harness.run_trajectory(ctrl_m8, "harmonic", duration=3.0, seed=1000 + r)[0]["rms_e"] for r in range(n_sweep_runs)]))

    ctrl_m9 = AblationController("M9")
    m9_os = float(np.mean([harness.run_trajectory(ctrl_m9, "step", duration=2.0, seed=1000 + r)[0]["overshoot"] for r in range(n_sweep_runs)]))
    m9_rms = float(np.mean([harness.run_trajectory(ctrl_m9, "harmonic", duration=3.0, seed=1000 + r)[0]["rms_e"] for r in range(n_sweep_runs)]))

    idx_default = d_cutoffs.index(1.0)
    def_rms, def_os = harm_rms_list[idx_default], step_os_list[idx_default]

    ax1.plot(harm_rms_list, step_os_list, "-o", color="#2563eb", linewidth=2.0, markersize=6.5,
             label="M7 Cutoff Sweep ($d_{\\mathrm{cutoff}} \\in [0.4, 4.0]$ Hz)")
    ax1.plot([m8_rms], [m8_os], "^", color="#dc2626", markersize=9,
             label=f"M8: Classical PID ({m8_os:.1f}% OS, {m8_rms:.2f} px RMS)")
    ax1.plot([m3_rms], [m3_os], "s", color="#64748b", markersize=7.5,
             label=f"M3: PD + Ego-Comp ({m3_os:.1f}% OS, {m3_rms:.2f} px RMS)")
    ax1.scatter([def_rms], [def_os], s=150, facecolors="none", edgecolors="#10b981", linewidths=2.5, zorder=6,
                label=f"M7 Default $d_{{\\mathrm{{cutoff}}}}=1.0$ Hz ({def_os:.1f}% OS, {def_rms:.2f} px)")

    # Text note for off-scale M9 (without misleading arrow)
    ax1.text(
        0.04, 0.69,
        f"Off-Scale Baseline Note (N={n_sweep_runs} sweep):\n"
        f"M9 (CV-KF + PD): {m9_os:.1f}% OS, RMS = {m9_rms:.2f} px\n"
        f"(Table 4 N=100 benchmark: 23.9% OS, 60.71 px; off x-axis)",
        transform=ax1.transAxes,
        fontsize=8.0, fontweight="bold", color="#b45309",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffbeb", ec="#d97706", lw=1.0),
    )
    ax1.set_title("(a) Multi-Objective Trade-off: Step Overshoot vs. Harmonic Lag", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Harmonic Tracking Error $\\mathrm{RMS}_e$ (px)", fontsize=10.5)
    ax1.set_ylabel("Mean Step Percentage Overshoot $\\%OS$ (%)", fontsize=10.5)
    ax1.set_xlim(3.0, 24.0)
    ax1.set_ylim(-2.0, 75.0)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.1)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Live 2D grid simulation over (Kp_base, Kd_base)
    kp_vals = np.linspace(0.24, 0.64, 7)
    kd_vals = np.linspace(0.12, 0.42, 7)
    os_grid = np.zeros((len(kd_vals), len(kp_vals)))

    for i, kd in enumerate(kd_vals):
        for j, kp in enumerate(kp_vals):
            ctrl = AblationController("M7", p_base=float(kp), d_base=float(kd))
            runs = [harness.run_trajectory(ctrl, "step", duration=2.0, seed=1000 + r)[0]["overshoot"] for r in range(10)]
            os_grid[i, j] = float(np.mean(runs))

    cp = ax2.contourf(kp_vals, kd_vals, os_grid, levels=12, cmap="YlOrRd")
    plt.colorbar(cp, ax=ax2, label="Simulated Mean Step Overshoot $\\%OS$ (%)")
    ax2.plot([0.36], [0.22], "*", color="#10b981", markersize=13, markeredgecolor="black", label="Benchmark Calibrated (0.36, 0.22)")
    ax2.plot([0.44], [0.32], "o", color="#2563eb", markersize=8, markeredgecolor="white", label="Standard Profile (0.44, 0.32)")
    ax2.plot([0.59], [0.38], "^", color="#f59e0b", markersize=8, markeredgecolor="white", label="High-Response Profile (0.59, 0.38)")

    ax2.set_title("(b) Simulated Mean Step Overshoot Contour Over $(K_{p,\\mathrm{base}}, K_{d,\\mathrm{base}})$", fontsize=10.2, fontweight="bold")
    ax2.set_xlabel("Base Proportional Gain $K_{p,\\mathrm{base}}$", fontsize=10.5)
    ax2.set_ylabel("Base Derivative Damping $K_{d,\\mathrm{base}}$", fontsize=10.5)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.3)

    plt.tight_layout()
    for d in save_dirs:
        plt.savefig(os.path.join(d, "fig6_pareto_and_sensitivity.png"))
    plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = [os.path.join(script_dir, "figures")]

    print("=== Generating Figure 4, Figure 5, and Figure 6 ===")
    plot_fig4_latency_and_roi_pareto(dirs)
    print("[Plot] Saved fig4_latency_roi_pareto.png")
    plot_fig5_aliasing_and_zpoles(dirs)
    print("[Plot] Saved fig5_m1_aliasing_and_zpoles.png")
    plot_fig6_pareto_and_contour(dirs)
    print("[Plot] Saved fig6_pareto_and_sensitivity.png")
