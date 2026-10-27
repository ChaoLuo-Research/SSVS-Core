"""
Publication-Quality System Architecture Diagram Generator (Figure 1)
Manuscript: Low-Latency Screen-Space Visual Servoing via Asynchronous
            Perception-Control Decoupling and Gain-Scheduled Discrete Tracking
Output: figures/fig1_system_architecture.png
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def create_system_architecture_diagram(output_path: str) -> None:
    fig = plt.figure(figsize=(15, 9.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 95)
    ax.axis("off")

    fig.patch.set_facecolor("#ffffff")

    # Publication color palette
    c_blue_bg = "#f0f9ff"
    c_blue_border = "#0284c7"
    c_blue_box = "#ffffff"
    c_blue_title = "#0369a1"

    c_green_bg = "#f0fdf4"
    c_green_border = "#16a34a"
    c_green_box = "#ffffff"
    c_green_title = "#15803d"

    c_purple_bg = "#faf5ff"
    c_purple_border = "#7e22ce"
    c_purple_title = "#6b21a8"

    c_dark_text = "#0f172a"
    c_muted_text = "#334155"
    c_arrow = "#1e293b"
    c_feedback = "#b91c1c"

    # =========================================================================
    # 1. Main Title Banner
    # =========================================================================
    title_box = FancyBboxPatch(
        (6, 86.5), 138, 6.5,
        boxstyle="round,pad=0.5,rounding_size=1.2",
        fc="#0f172a", ec="none", lw=0, zorder=3
    )
    ax.add_patch(title_box)
    ax.text(
        75, 89.7,
        "Asynchronous Decoupled Perception-Control Architecture (SSVS)",
        color="#ffffff", fontsize=13.5, fontweight="bold",
        ha="center", va="center", zorder=4
    )

    # =========================================================================
    # 2. TOP SWIMLANE: Perception Pipeline Thread (~160 FPS, Asynchronous)
    # =========================================================================
    top_lane = FancyBboxPatch(
        (6, 52), 138, 32,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        fc=c_blue_bg, ec=c_blue_border, lw=2.0, linestyle="--", zorder=1
    )
    ax.add_patch(top_lane)

    top_tag = FancyBboxPatch(
        (9, 77.5), 68, 5.0,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        fc=c_blue_border, ec="none", zorder=3
    )
    ax.add_patch(top_tag)
    ax.text(
        43, 80.0,
        "PERCEPTION PIPELINE THREAD (Event-Driven ~160 FPS, GPU DirectML)",
        color="#ffffff", fontsize=9.8, fontweight="bold",
        ha="center", va="center", zorder=4
    )

    # Block 1: Surface Ingestion
    b1 = FancyBboxPatch(
        (10, 55.5), 36, 19.5,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        fc=c_blue_box, ec="#7dd3fc", lw=1.5, zorder=3
    )
    ax.add_patch(b1)
    ax.text(28, 71.5, "1. Surface Ingestion", color=c_blue_title, fontsize=11, fontweight="bold", ha="center", zorder=4)
    ax.text(28, 67.2, "[DXGI Desktop Duplication]", color=c_dark_text, fontsize=9.2, fontweight="bold", ha="center", zorder=4)
    ax.text(
        28, 61.2,
        "• Direct3D 11 VRAM surface lock\n"
        "• GPU-to-GPU staging transfer\n"
        "• Zero CPU copy overhead\n"
        "• Mean Latency: 2.12 ms (±0.38)",
        color=c_muted_text, fontsize=8.2, ha="center", va="center", linespacing=1.3, zorder=4
    )

    # Block 2: Neural Feature Inference
    b2 = FancyBboxPatch(
        (54, 55.5), 42, 19.5,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        fc=c_blue_box, ec="#7dd3fc", lw=1.5, zorder=3
    )
    ax.add_patch(b2)
    ax.text(75, 71.5, "2. Neural Feature Inference", color=c_blue_title, fontsize=11, fontweight="bold", ha="center", zorder=4)
    ax.text(75, 67.2, "[DirectML Tensor Acceleration]", color=c_dark_text, fontsize=9.2, fontweight="bold", ha="center", zorder=4)
    ax.text(
        75, 61.2,
        "• FP16 half-precision execution graph\n"
        "• Multi-scale dynamic ROI (320×320)\n"
        "• GPU kernel compute isolation\n"
        "• Mean Latency: 8.24 ms (±0.76)",
        color=c_muted_text, fontsize=8.2, ha="center", va="center", linespacing=1.3, zorder=4
    )

    # Block 3: Post-Processing & Centroid Parsing
    b3 = FancyBboxPatch(
        (104, 55.5), 36, 19.5,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        fc=c_blue_box, ec="#7dd3fc", lw=1.5, zorder=3
    )
    ax.add_patch(b3)
    ax.text(122, 71.5, "3. Post-Processing & Parse", color=c_blue_title, fontsize=11, fontweight="bold", ha="center", zorder=4)
    ax.text(122, 67.2, "[Foveated Recall & Clustering]", color=c_dark_text, fontsize=9.2, fontweight="bold", ha="center", zorder=4)
    ax.text(
        122, 61.2,
        "• Foveated dual-threshold recall\n"
        "• Sub-part bipartite clustering\n"
        "• Hysteresis target lock: p_raw\n"
        "• Mean Latency: 0.58 ms (±0.12)",
        color=c_muted_text, fontsize=8.2, ha="center", va="center", linespacing=1.3, zorder=4
    )

    # Forward arrows in Perception lane
    arr1 = FancyArrowPatch(
        (46.5, 65.2), (53.5, 65.2),
        arrowstyle="simple,tail_width=2.0,head_width=6,head_length=6",
        fc=c_arrow, ec=c_arrow, zorder=3
    )
    ax.add_patch(arr1)

    arr2 = FancyArrowPatch(
        (96.5, 65.2), (103.5, 65.2),
        arrowstyle="simple,tail_width=2.0,head_width=6,head_length=6",
        fc=c_arrow, ec=c_arrow, zorder=3
    )
    ax.add_patch(arr2)

    # =========================================================================
    # 3. INTER-THREAD ASYNCHRONOUS BRIDGE (Thread-Safe Synchronized Context Slot)
    # =========================================================================
    bridge_box = FancyBboxPatch(
        (96, 41.5), 46, 7.5,
        boxstyle="round,pad=0.5,rounding_size=0.9",
        fc=c_purple_bg, ec=c_purple_border, lw=2.0, zorder=3
    )
    ax.add_patch(bridge_box)
    ax.text(
        119, 46.2,
        "Thread-Safe Synchronized Context Slot",
        color=c_purple_title, fontsize=9.8, fontweight="bold", ha="center", zorder=4
    )
    ax.text(
        119, 43.3,
        "Mutex-guarded latest-sample slot: (Timestamp t_n, Centroid p_raw)",
        color=c_muted_text, fontsize=8.0, ha="center", zorder=4
    )

    # Arrow down from Block 3 to Context Slot
    arr_store = FancyArrowPatch(
        (120, 55.5), (120, 49.3),
        arrowstyle="simple,tail_width=2.2,head_width=6.5,head_length=6.5",
        fc=c_purple_border, ec=c_purple_border, zorder=3
    )
    ax.add_patch(arr_store)
    ax.text(
        123.0, 52.5, "Slot Update\n(Fresh Frame)",
        color=c_purple_title, fontsize=7.8, fontweight="bold", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec="none", alpha=0.92), zorder=4
    )

    # Orthogonal bus from Context Slot to Control Thread Block 4
    ax.plot([96, 28], [45.2, 45.2], color=c_purple_border, lw=2.2, linestyle="-", zorder=2)
    arr_load = FancyArrowPatch(
        (28, 45.2), (28, 30.8),
        arrowstyle="simple,tail_width=2.2,head_width=6.5,head_length=6.5",
        fc=c_purple_border, ec=c_purple_border, zorder=2
    )
    ax.add_patch(arr_load)

    bus_tag = FancyBboxPatch(
        (35, 43.0), 56, 4.4,
        boxstyle="round,pad=0.3,rounding_size=0.6",
        fc="#f3e8ff", ec=c_purple_border, lw=1.4, zorder=4
    )
    ax.add_patch(bus_tag)
    ax.text(
        63, 45.2,
        "Read Latest Observation (1000 Hz Isochronous Polling)",
        color=c_purple_title, fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=5
    )

    # =========================================================================
    # 4. BOTTOM SWIMLANE: Servoing Control Thread (Nominal 1000 Hz)
    # =========================================================================
    bot_lane = FancyBboxPatch(
        (6, 9), 138, 30.5,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        fc=c_green_bg, ec=c_green_border, lw=2.0, zorder=1
    )
    ax.add_patch(bot_lane)

    bot_tag = FancyBboxPatch(
        (50, 33.5), 78, 4.8,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        fc=c_green_border, ec="none", zorder=3
    )
    ax.add_patch(bot_tag)
    ax.text(
        89.0, 35.9,
        "SERVOING CONTROL THREAD (Nominal 1000 Hz, WinMM Timer, Δt = 1.0 ms)",
        color="#ffffff", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4
    )

    # Block 4: State Estimation & Filter
    b4 = FancyBboxPatch(
        (10, 11), 36, 19.5,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        fc=c_green_box, ec="#86efac", lw=1.5, zorder=3
    )
    ax.add_patch(b4)
    ax.text(28, 27.2, "4. State Filter & Decouple", color=c_green_title, fontsize=11, fontweight="bold", ha="center", zorder=4)
    ax.text(28, 23.0, "[Adaptive 1-Euro Filter]", color=c_dark_text, fontsize=9.2, fontweight="bold", ha="center", zorder=4)
    ax.text(
        28, 17.0,
        "• Speed-adaptive cutoff frequency\n"
        "• Internal velocity derivative: v(k)\n"
        "• Continuous inverse ego-motion shift\n"
        "• Static Jitter: 0.64 px (70.2% drop)",
        color=c_muted_text, fontsize=8.2, ha="center", va="center", linespacing=1.3, zorder=4
    )

    # Block 5: 5-Term Control Law
    b5 = FancyBboxPatch(
        (53, 11), 44, 19.5,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        fc=c_green_box, ec="#86efac", lw=1.5, zorder=3
    )
    ax.add_patch(b5)
    ax.text(75, 27.2, "5. Five-Term Discrete Law", color=c_green_title, fontsize=11, fontweight="bold", ha="center", zorder=4)
    ax.text(75, 23.0, "[Gain-Scheduled Rate-Normalized]", color=c_dark_text, fontsize=9.2, fontweight="bold", ha="center", zorder=4)
    ax.text(
        75, 17.0,
        "• Piecewise linear gain-scheduled Kp (0.90/0.70)\n"
        "• Dynamic chasing-unloading Kd & soft clamp\n"
        "• Conditional leaky cruise integrator Ki (Anti-Windup)\n"
        "• Feedforward u_ff + horizon lead gating u_lead\n"
        "• Nominal PD poles: z1 = 0.9547, z2 = -0.8747",
        color=c_muted_text, fontsize=7.9, ha="center", va="center", linespacing=1.25, zorder=4
    )

    # Block 6: Actuation Dispatch
    b6 = FancyBboxPatch(
        (104, 11), 36, 19.5,
        boxstyle="round,pad=0.5,rounding_size=1.0",
        fc=c_green_box, ec="#86efac", lw=1.5, zorder=3
    )
    ax.add_patch(b6)
    ax.text(122, 27.2, "6. Actuator Dispatch", color=c_green_title, fontsize=11, fontweight="bold", ha="center", zorder=4)
    ax.text(122, 23.0, "[Sub-Count Residual & Clamp]", color=c_dark_text, fontsize=9.2, fontweight="bold", ha="center", zorder=4)
    ax.text(
        122, 17.0,
        "• 2D vector saturation (u_max = 120)\n"
        "• Sub-count residual accumulator: Δu\n"
        "• OS input dispatch (1.18 ± 0.24 ms)\n"
        "• Actuator scale: K_scale ≈ 1.004 counts/px",
        color=c_muted_text, fontsize=8.1, ha="center", va="center", linespacing=1.3, zorder=4
    )

    # Forward arrows in Control lane
    arr4 = FancyArrowPatch(
        (46.5, 20.7), (52.5, 20.7),
        arrowstyle="simple,tail_width=2.0,head_width=6,head_length=6",
        fc=c_arrow, ec=c_arrow, zorder=3
    )
    ax.add_patch(arr4)

    arr5 = FancyArrowPatch(
        (97.5, 20.7), (103.5, 20.7),
        arrowstyle="simple,tail_width=2.0,head_width=6,head_length=6",
        fc=c_arrow, ec=c_arrow, zorder=3
    )
    ax.add_patch(arr5)

    # Output Physical Displacement arrow
    ax.set_xlim(0, 152)
    arr_out = FancyArrowPatch(
        (140.5, 20.7), (150.5, 20.7),
        arrowstyle="simple,tail_width=2.4,head_width=7,head_length=7",
        fc="#059669", ec="#059669", zorder=3
    )
    ax.add_patch(arr_out)
    ax.text(
        147.8, 25.0, "Physical\nRotation",
        color="#059669", fontsize=8.4, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="none", alpha=0.95), zorder=4
    )

    # =========================================================================
    # 5. FEEDBACK LOOP: Inverse Ego-Motion Reprojection (Routing along bottom)
    # =========================================================================
    ax.plot([122, 122], [11.0, 4.0], color=c_feedback, lw=2.2, linestyle="-", zorder=2)
    ax.plot([122, 28], [4.0, 4.0], color=c_feedback, lw=2.2, linestyle="-", zorder=2)
    arr_feedback = FancyArrowPatch(
        (28, 4.0), (28, 10.8),
        arrowstyle="simple,tail_width=2.2,head_width=6.5,head_length=6.5",
        fc=c_feedback, ec=c_feedback, zorder=2
    )
    ax.add_patch(arr_feedback)

    fb_badge = FancyBboxPatch(
        (36, 1.8), 78, 4.4,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        fc="#fef2f2", ec=c_feedback, lw=1.5, zorder=4
    )
    ax.add_patch(fb_badge)
    ax.text(
        75, 4.0,
        r"Inverse Ego-Motion Reprojection Feedback: $\Delta p_{\mathrm{ego}}(k) = -K_{\mathrm{scale}}^{-1} \cdot \Delta u(k)$",
        color=c_feedback, fontsize=9.2, fontweight="bold", ha="center", va="center", zorder=5
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(script_dir, "figures", "fig1_system_architecture.png")
    create_system_architecture_diagram(out_file)
