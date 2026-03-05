"""
PMSM Motor Control: Clarke & Park Transformations
==================================================
Converts three-phase (abc) currents → αβ (Clarke) → dq (Park)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────
# 1. SIMULATION PARAMETERS
# ─────────────────────────────────────────
fs      = 10_000          # sample rate (Hz)
f_elec  = 50              # electrical frequency (Hz)  — one pole-pair → 3000 RPM
t_end   = 2 / f_elec      # simulate 2 full electrical cycles
t       = np.arange(0, t_end, 1/fs)

I_peak  = 10.0            # peak phase current (A)
omega_e = 2 * np.pi * f_elec   # electrical angular velocity (rad/s)
theta_e = omega_e * t          # rotor electrical angle (rad)

# ─────────────────────────────────────────
# 2. THREE-PHASE CURRENTS  (balanced, 120° apart)
# ─────────────────────────────────────────
i_a =  I_peak * np.cos(theta_e)
i_b =  I_peak * np.cos(theta_e - 2*np.pi/3)
i_c =  I_peak * np.cos(theta_e + 2*np.pi/3)

# Verify: balanced set sums to zero at every instant
assert np.allclose(i_a + i_b + i_c, 0, atol=1e-10), "Currents not balanced!"

# ─────────────────────────────────────────
# 3. CLARKE TRANSFORM  (abc → αβ)
#    Power-invariant form (factor = √(2/3))
# ─────────────────────────────────────────
def clarke_transform(ia, ib, ic):
    """
    Maps three-phase currents onto two orthogonal stationary axes (α, β).
    Uses the power-invariant (amplitude-preserving for balanced sets) form.
    """
    k = np.sqrt(2/3)
    i_alpha = k * (ia - 0.5*ib - 0.5*ic)
    i_beta  = k * (np.sqrt(3)/2 * ib - np.sqrt(3)/2 * ic)
    # i_zero  = k * (ia + ib + ic) / np.sqrt(2)   # zero-seq (= 0 for balanced)
    return i_alpha, i_beta

i_alpha, i_beta = clarke_transform(i_a, i_b, i_c)

# ─────────────────────────────────────────
# 4. PARK TRANSFORM  (αβ → dq)
#    Rotates the stationary αβ frame by θ_e
#    so that d-axis aligns with rotor flux.
# ─────────────────────────────────────────
def park_transform(i_alpha, i_beta, theta):
    """
    Projects the stationary αβ vector onto axes that rotate with the rotor.
    d-axis → aligned with rotor permanent-magnet flux
    q-axis → 90° ahead of d-axis (torque-producing direction)
    """
    i_d =  i_alpha * np.cos(theta) + i_beta * np.sin(theta)
    i_q = -i_alpha * np.sin(theta) + i_beta * np.cos(theta)
    return i_d, i_q

i_d, i_q = park_transform(i_alpha, i_beta, theta_e)

# ─────────────────────────────────────────
# 5. INVERSE TRANSFORMS (verification)
# ─────────────────────────────────────────
def inverse_park(id_, iq_, theta):
    alpha = id_ * np.cos(theta) - iq_ * np.sin(theta)
    beta  = id_ * np.sin(theta) + iq_ * np.cos(theta)
    return alpha, beta

def inverse_clarke(i_alpha, i_beta):
    ia =  np.sqrt(2/3) * i_alpha
    ib =  np.sqrt(2/3) * (-i_alpha/2 + np.sqrt(3)/2 * i_beta)
    ic =  np.sqrt(2/3) * (-i_alpha/2 - np.sqrt(3)/2 * i_beta)
    return ia, ib, ic

alpha_r, beta_r = inverse_park(i_d, i_q, theta_e)
ia_r, ib_r, ic_r = inverse_clarke(alpha_r, beta_r)
assert np.allclose(ia_r, i_a, atol=1e-9), "Inverse transform mismatch!"

# ─────────────────────────────────────────
# 6. PLOTTING
# ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccee",
    "xtick.color":      "#888899",
    "ytick.color":      "#888899",
    "text.color":       "#ccccee",
    "grid.color":       "#2a2a44",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "monospace",
})

t_ms = t * 1000   # display in milliseconds

fig = plt.figure(figsize=(16, 14))
fig.suptitle("PMSM Motor Control — Clarke & Park Transformations",
             fontsize=15, fontweight="bold", color="#e0d0ff", y=0.98)

gs = gridspec.GridSpec(3, 2, hspace=0.52, wspace=0.32,
                       left=0.07, right=0.96, top=0.93, bottom=0.06)

COLOR_A = "#ff6b6b"
COLOR_B = "#4ecdc4"
COLOR_C = "#ffe66d"
COLOR_ALPHA = "#a29bfe"
COLOR_BETA  = "#fd79a8"
COLOR_D     = "#00cec9"
COLOR_Q     = "#fdcb6e"

# ── Panel 1 : Three-phase currents ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t_ms, i_a, color=COLOR_A, lw=1.8, label=r"$i_a$")
ax1.plot(t_ms, i_b, color=COLOR_B, lw=1.8, label=r"$i_b$")
ax1.plot(t_ms, i_c, color=COLOR_C, lw=1.8, label=r"$i_c$")
ax1.axhline(0, color="#555577", lw=0.8)
ax1.set_title("[1] Three-Phase Currents (abc frame)", fontsize=11, color="#e0d0ff")
ax1.set_ylabel("Current (A)")
ax1.set_xlabel("Time (ms)")
ax1.legend(loc="upper right", framealpha=0.3, ncol=3)
ax1.grid(True)
ax1.set_ylim(-13, 13)

# ── Panel 2 : Clarke αβ waveforms ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t_ms, i_alpha, color=COLOR_ALPHA, lw=1.8, label=r"$i_\alpha$")
ax2.plot(t_ms, i_beta,  color=COLOR_BETA,  lw=1.8, label=r"$i_\beta$", linestyle="--")
ax2.axhline(0, color="#555577", lw=0.8)
ax2.set_title("[2] Clarke Transform  (αβ stationary)", fontsize=11, color="#e0d0ff")
ax2.set_ylabel("Current (A)")
ax2.set_xlabel("Time (ms)")
ax2.legend(loc="upper right", framealpha=0.3)
ax2.grid(True)
ax2.set_ylim(-13, 13)

# ── Panel 3 : Park dq waveforms ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t_ms, i_d, color=COLOR_D, lw=1.8, label=r"$i_d$  (flux)")
ax3.plot(t_ms, i_q, color=COLOR_Q, lw=1.8, label=r"$i_q$  (torque)", linestyle="--")
ax3.axhline(0, color="#555577", lw=0.8)
ax3.set_title("[3] Park Transform  (dq rotating)", fontsize=11, color="#e0d0ff")
ax3.set_ylabel("Current (A)")
ax3.set_xlabel("Time (ms)")
ax3.legend(loc="upper right", framealpha=0.3)
ax3.grid(True)
ax3.set_ylim(-13, 13)

# ── Panel 4 : αβ locus (should be a circle) ─────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0], aspect="equal")
sc = ax4.scatter(i_alpha, i_beta, c=t, cmap="plasma", s=4, alpha=0.8)
plt.colorbar(sc, ax=ax4, label="Time →", fraction=0.046, pad=0.04)
ax4.set_title("αβ Current Vector Locus\n(circle = balanced supply)", fontsize=10, color="#e0d0ff")
ax4.set_xlabel(r"$i_\alpha$ (A)")
ax4.set_ylabel(r"$i_\beta$ (A)")
ax4.axhline(0, color="#555577", lw=0.7)
ax4.axvline(0, color="#555577", lw=0.7)
ax4.grid(True)

# ── Panel 5 : dq locus (should be a point / very small circle) ──────────────
ax5 = fig.add_subplot(gs[2, 1], aspect="equal")
sc2 = ax5.scatter(i_d, i_q, c=t, cmap="plasma", s=4, alpha=0.8)
plt.colorbar(sc2, ax=ax5, label="Time →", fraction=0.046, pad=0.04)
# Mark the mean (DC) operating point
ax5.plot(np.mean(i_d), np.mean(i_q), "w*", ms=14, label=f"DC point\n({np.mean(i_d):.2f}, {np.mean(i_q):.2f})")
ax5.set_title("dq Current Vector Locus\n(single point = pure DC in rotating frame)", fontsize=10, color="#e0d0ff")
ax5.set_xlabel(r"$i_d$ (A)")
ax5.set_ylabel(r"$i_q$ (A)")
ax5.axhline(0, color="#555577", lw=0.7)
ax5.axvline(0, color="#555577", lw=0.7)
ax5.grid(True)
ax5.legend(fontsize=8, framealpha=0.3)

plt.savefig(r"C:\Users\PRANAV\OneDrive\Desktop\pmsm_transforms.png", dpi=150, bbox_inches="tight")
print("Plot saved ✓")

# ─────────────────────────────────────────
# 7. PRINT STEADY-STATE VALUES
# ─────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Peak phase current   : {I_peak:.2f} A")
print(f"  Mean i_d (flux)      : {np.mean(i_d):.4f} A  (≈ 0 for pure torque control)")
print(f"  Mean i_q (torque)    : {np.mean(i_q):.4f} A")
print(f"  Peak i_alpha         : {np.max(i_alpha):.4f} A")
print(f"  Peak i_beta          : {np.max(i_beta):.4f} A")
print(f"{'='*50}")