"""
APPS1 vs APPS2 – Side-by-side comparison
Highlights samples where the normalized pedal travel difference exceeds 10%.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

# ── Configuration ─────────────────────────────────────────────────────────────
ADC_RESOLUTION = 4096
V_REF          = 3.3
MOVING_AVG_WIN = 10
DIFF_THRESHOLD = 10.0          # percent

SENSORS = {
    "APPS1": dict(file="apps1_app.txt", v_min=0.00, v_max=1.68, color="#4fc3f7"),
    "APPS2": dict(file="apps2_app.txt", v_min=1.70, v_max=3.30, color="#a5d6a7"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_adc(filepath: str) -> np.ndarray:
    raw, skipped = [], 0
    with open(filepath) as fh:
        for lineno, line in enumerate(fh, 1):
            s = line.strip()
            if not s:
                skipped += 1; continue
            try:
                raw.append(int(s))
            except ValueError:
                skipped += 1
                print(f"  [SKIP] {os.path.basename(filepath)} line {lineno}: '{s}'")
    print(f"  {os.path.basename(filepath)}: {len(raw):,} samples, {skipped} skipped")
    return np.array(raw, dtype=np.float64)

def normalize(adc: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    adc_min = round((v_min / V_REF) * (ADC_RESOLUTION - 1))
    adc_max = round((v_max / V_REF) * (ADC_RESOLUTION - 1))
    return np.clip((adc - adc_min) / (adc_max - adc_min) * 100.0, 0.0, 100.0)

def moving_average(data: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(data, np.ones(w) / w, mode="full")[:len(data)]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data = {}
    for name, cfg in SENSORS.items():
        for base in (script_dir, os.getcwd()):
            fp = os.path.join(base, cfg["file"])
            if os.path.isfile(fp):
                adc    = load_adc(fp)
                norm   = normalize(adc, cfg["v_min"], cfg["v_max"])
                smooth = moving_average(norm, MOVING_AVG_WIN)
                data[name] = dict(adc=adc, norm=norm, smooth=smooth, **cfg)
                break
        else:
            print(f"[ERROR] '{cfg['file']}' not found."); sys.exit(1)

    # Align lengths (use the shorter one)
    n = min(len(data["APPS1"]["smooth"]), len(data["APPS2"]["smooth"]))
    s1 = data["APPS1"]["smooth"][:n]
    s2 = data["APPS2"]["smooth"][:n]
    idx = np.arange(n)

    diff      = s1 - s2                          # signed difference
    abs_diff  = np.abs(diff)
    fault_mask = abs_diff > DIFF_THRESHOLD

    fault_indices = np.where(fault_mask)[0]
    print(f"\n{'═'*50}")
    print(f"  APPS1 vs APPS2 Discrepancy Report")
    print(f"{'═'*50}")
    print(f"  Total samples    : {n:,}")
    print(f"  Threshold        : ±{DIFF_THRESHOLD:.0f} %")
    print(f"  Fault samples    : {fault_mask.sum():,}  "
          f"({fault_mask.sum()/n*100:.2f} % of run)")
    if fault_mask.sum():
        print(f"  Max deviation    : {abs_diff.max():.2f} %  @ sample {abs_diff.argmax():,}")
    print(f"{'═'*50}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10), facecolor="#0f1117")
    fig.suptitle("APPS1 vs APPS2 – Normalised Pedal Travel Comparison",
                 fontsize=14, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(3, 1, hspace=0.5,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    panel_bg = "#1a1d27"
    fault_color = "#ff1744"

    # ── Panel 1: both smoothed signals ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(panel_bg)
    ax1.plot(idx, s1, color=data["APPS1"]["color"], lw=1.0, label="APPS1 smoothed")
    ax1.plot(idx, s2, color=data["APPS2"]["color"], lw=1.0, label="APPS2 smoothed")

    # shade fault regions
    ax1.fill_between(idx, s1, s2,
                     where=fault_mask, color=fault_color, alpha=0.35,
                     label=f"|diff| > {DIFF_THRESHOLD:.0f} %")
    ax1.set_ylim(-5, 105)
    ax1.set_xlim(0, n - 1)
    ax1.set_ylabel("Pedal Travel (%)", color="#aaaaaa", fontsize=9)
    ax1.set_title("Smoothed Normalised Signals", color="white",
                  fontsize=10, fontweight="bold")
    ax1.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor("#333344")
    ax1.legend(fontsize=8, facecolor="#22253a", labelcolor="white",
               edgecolor="#444466", loc="upper right")

    # ── Panel 2: signed difference ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(panel_bg)
    ax2.plot(idx, diff, color="#ffe082", lw=0.8, label="APPS1 − APPS2")
    ax2.axhline( DIFF_THRESHOLD, color=fault_color, lw=1.1, ls="--",
                 label=f"+{DIFF_THRESHOLD:.0f} % threshold")
    ax2.axhline(-DIFF_THRESHOLD, color=fault_color, lw=1.1, ls="--",
                 label=f"−{DIFF_THRESHOLD:.0f} % threshold")
    ax2.axhline(0, color="#555566", lw=0.7, ls="-")
    ax2.fill_between(idx, diff, 0,
                     where=diff >  DIFF_THRESHOLD,
                     color=fault_color, alpha=0.30)
    ax2.fill_between(idx, diff, 0,
                     where=diff < -DIFF_THRESHOLD,
                     color=fault_color, alpha=0.30)
    ax2.set_xlim(0, n - 1)
    ax2.set_ylabel("Difference (%)", color="#aaaaaa", fontsize=9)
    ax2.set_title("Signed Difference (APPS1 − APPS2)", color="white",
                  fontsize=10, fontweight="bold")
    ax2.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333344")
    ax2.legend(fontsize=8, facecolor="#22253a", labelcolor="white",
               edgecolor="#444466", loc="upper right")

    # ── Panel 3: fault indicator bar ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(panel_bg)
    ax3.fill_between(idx, 0, fault_mask.astype(float),
                     color=fault_color, alpha=0.85, step="mid",
                     label="Fault (|diff| > 10 %)")
    ax3.set_xlim(0, n - 1)
    ax3.set_ylim(-0.05, 1.2)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["OK", "FAULT"], color="#aaaaaa", fontsize=8)
    ax3.set_xlabel("Sample Index", color="#aaaaaa", fontsize=9)
    ax3.set_title("Fault Flag", color="white", fontsize=10, fontweight="bold")
    ax3.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax3.spines.values(): sp.set_edgecolor("#333344")
    ax3.legend(fontsize=8, facecolor="#22253a", labelcolor="white",
               edgecolor="#444466", loc="upper right")

    out = os.path.join(script_dir, "apps_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()