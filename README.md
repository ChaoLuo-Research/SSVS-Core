# SSVS-Core: Screen-Space Visual Servoing Benchmark & Replication Package

Official open-source implementation and Monte-Carlo benchmark suite for **low-latency Screen-Space Visual Servoing (SSVS)** in assistive human-machine interaction and real-time closed-loop visual tracking research.

This replication package contains the five-term discrete-time control law, multi-rate inverse ego-motion compensated state estimator, lightweight perception post-processing algorithms, and reproducible simulation harnesses supporting the manuscript:
> **"Low-Latency Screen-Space Visual Servoing via Asynchronous Perception-Control Decoupling and Gain-Scheduled Discrete Tracking"**  
> Chao Luo (*Journal of Real-Time Image Processing*, Springer Nature)

---

> [!NOTE]
> **Reproducibility & Hardware Independence**  
> To enable deterministic reproduction under a fixed software stack, random seed, and discrete simulation protocol without requiring proprietary 3D rendering engines or physical HID actuation hardware:
> * `bench_trajectory.py` and `generate_extra_figures_and_m9.py` implement the discrete-time kinematic plant ($\Delta t_{\text{ctrl}} = 1.0\text{ ms}$, a 12-step ($12.0\text{ ms}$) discrete delay buffer representing the nominal $\tau_{\text{total}} = 12.4\text{ ms}$ transport latency, and calibrated $\sigma = 1.80\text{ px}$ detector centroid standard deviation), reproducing Table 4 (M0–M9 ablation, $N=100$), the Section 6.2 Empirical Delay Degradation Sweep, and empirical trajectory figures in approximately 90 seconds.
> * `ssvs_core.py` provides the standalone Five-Term Discrete SSVS Controller (`SmoothSSVSController`), Speed-Adaptive 1-Euro Filter (`OneEuroFilter`), Asynchronous 1000 Hz Control Worker (`AsyncControlWorker`), 2D Vector Actuator Saturation Operator ($\mathcal{S}_{u_{\max}}$), and nominal $\mathcal{Z}$-domain pole analyzer.
> * `perception_pipeline_benchmark.py` provides the reference implementation of Foveated Dual-Threshold Recall, Sub-Part Bipartite Geometric Clustering, Depth-Scale Hysteresis Target Locking (Section 3.4), and a dry-run safe wrapper of the Win32 `SendInput` relative dispatcher (preventing unintended cursor displacement on reviewer workstations).

---

## 1. Repository Structure

```text
SSVS-Core/
├── ssvs_core.py                      # Core library (OneEuroFilter, SmoothSSVSController, AsyncControlWorker, Pole Analyzer)
├── bench_trajectory.py               # Primary Monte-Carlo benchmark suite (M0-M9 ablation, Delay Degradation Sweep, Figs. 4-5)
├── perception_pipeline_benchmark.py  # Section 3.4 reference implementation (Foveated Recall, Bipartite Clustering, Selector)
├── generate_extra_figures_and_m9.py  # Simulation sweep & publication plot generator (Manuscript Figs. 2, 3, and 6)
├── generate_fig1_architecture.py     # Publication architectural schematic generator (Manuscript Fig. 1)
├── benchmark_results.json            # Frozen Monte-Carlo quantitative metrics (N = 100 runs per model, M0-M9)
├── requirements.txt                  # Python scientific computing dependencies
├── LICENSE                           # Apache License 2.0
├── README.md                         # Replication guide and benchmark documentation
└── figures/                          # 300 DPI publication-grade figures
    ├── fig1_system_architecture.png      # Manuscript Fig. 1: Asynchronous decoupled perception-control architecture
    ├── fig4_latency_roi_pareto.png       # Manuscript Fig. 2: Stage latency breakdown & dynamic ROI Pareto curve
    ├── fig5_m1_aliasing_and_zpoles.png   # Manuscript Fig. 3: Ego-motion aliasing mechanism & Z-domain pole placement
    ├── fig2_step_response.png            # Manuscript Fig. 4: 120 px step response comparison
    ├── fig3_ramp_harmonic.png            # Manuscript Fig. 5: 300 px/s ramp & 1.2 Hz harmonic tracking comparison
    └── fig6_pareto_and_sensitivity.png   # Manuscript Fig. 6: Bandwidth trade-off & Kp-Kd mean overshoot contour
```

---

## 2. Environment Setup

* **Python**: 3.9 or higher (verified on Python 3.10–3.14)
* **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

---

## 3. Reproducing Paper Results (Table 4 & Figures 1–6)

Run the standalone verification and Monte-Carlo benchmark scripts from the repository root:

```bash
# 1. Verify perception post-processing & dry-run actuator algorithms (Section 3.4)
python perception_pipeline_benchmark.py

# 2. Verify nominal delay-free linearized Z-domain closed-loop poles (Section 4.3)
python ssvs_core.py

# 3. Regenerate Manuscript Fig. 1 (system architecture schematic)
python generate_fig1_architecture.py

# 4. Run M0-M9 Monte-Carlo campaign (N = 100) & regenerate Manuscript Figs. 4 and 5 (Table 4)
python bench_trajectory.py

# 5. Run latency/parameter sweeps & regenerate Manuscript Figs. 2, 3, and 6
python generate_extra_figures_and_m9.py
```

### Expected Quantitative Outputs

1. **Table 4 Ablation & Baseline Comparison ($N = 100$ Monte-Carlo trials per model, 1000/1000 step trials reaching the 90% threshold with 0 censored runs)**:

| Model ID | Configuration | $t_{90\%}$ (ms) | $\%OS$ (%) | $e_{ss}$ (px) | $\text{RMS}_e$ (px) | $\sigma_{\text{jitter}}$ (px) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **M0** | Raw Proportional Baseline | 36.3 ± 0.5 | 26.5 ± 1.0% | 5.81 ± 0.15 | 8.46 ± 0.10 | 2.15 ± 0.14 |
| **M1** | M0 + 1-Euro Adaptive Filter | 74.7 ± 8.0 | 88.2 ± 9.3% | 13.56 ± 6.26 | 30.28 ± 7.00 | 4.23 ± 1.94 |
| **M2** | M1 + Fixed-Damping Derivative | 65.2 ± 8.1 | 56.3 ± 5.9% | 5.47 ± 0.16 | 7.96 ± 0.12 | 0.92 ± 0.25 |
| **M3** | M2 + Inverse Ego-Motion Reprojection | 112.1 ± 12.5 | 2.1 ± 0.6% | 9.43 ± 0.16 | 19.98 ± 0.22 | **0.50 ± 0.10** |
| **M4** | M3 + Chasing Unloading & Soft-Landing | 110.8 ± 12.1 | 2.9 ± 0.7% | 8.64 ± 0.16 | 19.36 ± 0.23 | **0.50 ± 0.11** |
| **M5** | M4 + Target Velocity Feedforward | 107.7 ± 12.7 | 3.3 ± 0.6% | 7.51 ± 0.18 | 18.83 ± 0.23 | **0.52 ± 0.10** |
| **M6** | M5 + Horizon Lead Delay Gating | 104.9 ± 13.1 | 4.6 ± 0.9% | 5.65 ± 0.18 | 18.05 ± 0.28 | 0.59 ± 0.09 |
| **M7** | **Proposed Full Framework** | **81.2 ± 17.2** | **2.4 ± 2.9%** | **3.58 ± 0.21** | **14.58 ± 0.31** | **0.64 ± 0.09** |
| **M8** | Classical Discrete PID Baseline | 30.8 ± 0.4 | 68.4 ± 1.6% | 1.92 ± 0.19 | 4.54 ± 0.19 | 2.52 ± 0.21 |
| **M9** | Constant-Velocity Kalman Filter (CV-KF + PD) | 172.2 ± 1.1 | 23.9 ± 0.5% | 5.08 ± 0.17 | 60.71 ± 0.15 | **0.52 ± 0.14** |

2. **Empirical Delay Degradation Sweep ($\tau_{\text{inj}} \in [0, 16]\text{ ms}$, nominal $\tau_{\text{total}} \in [12.4, 28.4]\text{ ms}$, `seed=42`, 3.0 s harmonic window, Section 6.2)**:

| Injected $\tau_{\text{inj}}$ (ms) | Nominal Total $\tau_{\text{total}}$ (ms) | Discrete Buffer Steps | Step Rise Time $t_{90\%}$ (ms) | Step Overshoot $\%OS$ (%) | Harmonic Tracking $\text{RMS}_e$ (px) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| +0 ms | 12.4 ms | 12 (12.0 ms) | 51.0 | 8.7% | 14.97 |
| +2 ms | 14.4 ms | 14 (14.0 ms) | 51.0 | 13.7% | 15.09 |
| +4 ms | 16.4 ms | 16 (16.0 ms) | 51.0 | 20.4% | 15.31 |
| +6 ms | 18.4 ms | 18 (18.0 ms) | 53.0 | 25.3% | 15.21 |
| +8 ms | 20.4 ms | 20 (20.0 ms) | 55.0 | 39.4% | 14.98 |
| +10 ms | 22.4 ms | 22 (22.0 ms) | 77.0 | 19.5% | 14.80 |
| +12 ms | 24.4 ms | 24 (24.0 ms) | 62.0 | 37.8% | 15.39 |
| +14 ms | 26.4 ms | 26 (26.0 ms) | 63.0 | 59.4% | 16.76 |
| +16 ms | 28.4 ms | 28 (28.0 ms) | 65.0 | 62.7% | 18.33 |

---

## 4. Citation

If you find this benchmark suite or control formulation useful in your research, please cite:

```bibtex
@article{luo2026ssvs,
  author  = {Luo, Chao},
  title   = {Low-Latency Screen-Space Visual Servoing via Asynchronous Perception-Control Decoupling and Gain-Scheduled Discrete Tracking},
  journal = {Journal of Real-Time Image Processing},
  year    = {2026},
  url     = {https://github.com/ChaoLuo-Research/SSVS-Core}
}
```

---

## 5. License and Ethics

This project is licensed under the [Apache License 2.0](LICENSE). All benchmark evaluations are conducted strictly on synthetic mathematical trajectories and isolated offline simulation testbeds.
