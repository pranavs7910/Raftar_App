"""
APPS2 Accelerator Pedal Position Sensor - Data Analysis Script
Analyzes raw 12-bit ADC values from apps2_app.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# ── Configuration ────────────────────────────────────────────────────────────
ADC_RESOLUTION  = 4096          # 12-bit ADC  (2^12)
V_REF           = 3.3           # Microcontroller reference voltage (V)
V_MIN           = 1.7           # APPS2 valid range: lower bound (V)
V_MAX           = 3.3           # APPS2 valid range: upper bound  (V)
MOVING_AVG_WIN  = 10            # Moving-average window size (samples)
FILE_NAME       = "apps2_app.txt"

# Derived ADC limits from voltage range
ADC_MIN = round((V_MIN / V_REF) * (ADC_RESOLUTION - 1))   # → ~2111
ADC_MAX = round((V_MAX / V_REF) * (ADC_RESOLUTION - 1))   # → 4095

# ── 1. Data Loading ───────────────────────────────────────────────────────────
def load_adc_values(filepath: str) -> np.ndarray:
    """Read file, skip non-numeric / empty lines, return int array."""
    raw = []
    skipped = 0
    with open(filepath, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                skipped += 1
                continue
            try:
                val = int(stripped)
                if not (0 <= val <= 4095):
                    print(f"  [WARN] Line {lineno}: value {val} out of 12-bit range – kept as-is")
                raw.append(val)
            except ValueError:
                skipped += 1
                print(f"  [SKIP] Line {lineno}: non-numeric content '{stripped}'")

    print(f"  Loaded  : {len(raw):,} samples")
    print(f"  Skipped : {skipped} lines")
    return np.array(raw, dtype=np.float64)

# ── 2. Normalisation ──────────────────────────────────────────────────────────
def normalize_to_pedal_travel(adc: np.ndarray) -> np.ndarray:
    """Map ADC values to 0–100 % pedal travel; clip outside calibrated range."""
    pedal_pct = (adc - ADC_MIN) / (ADC_MAX - ADC_MIN) * 100.0
    return np.clip(pedal_pct, 0.0, 100.0)

# ── 3. Moving-Average Filter ──────────────────────────────────────────────────
def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average – output length equals input length."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="full")[:len(data)]

# ── 4. Visualisation ──────────────────────────────────────────────────────────
def plot_results(adc_raw: np.ndarray,
                 pedal_raw: np.ndarray,
                 pedal_smooth: np.ndarray) -> None:

    time_axis = np.arange(len(adc_raw))

    fig = plt.figure(figsize=(14, 8), facecolor="#0f1117")
    fig.suptitle("APPS2 – Accelerator Pedal Position Sensor Analysis",
                 fontsize=15, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(2, 1, hspace=0.45, left=0.08, right=0.97,
                           top=0.92, bottom=0.08)

    # ── Top: Raw ADC ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#1a1d27")
    ax1.plot(time_axis, adc_raw, color="#4fc3f7", linewidth=0.6,
             alpha=0.85, label="Raw ADC")
    ax1.axhline(ADC_MAX, color="#ef5350", linewidth=1.2, linestyle="--",
                label=f"ADC_MAX ({ADC_MAX} ≈ {V_MAX} V)")
    ax1.axhline(ADC_MIN, color="#66bb6a", linewidth=1.2, linestyle="--",
                label=f"ADC_MIN ({ADC_MIN} ≈ {V_MIN} V)")
    ax1.set_xlim(0, len(adc_raw) - 1)
    ax1.set_ylim(-50, ADC_RESOLUTION + 50)
    ax1.set_xlabel("Sample Index", color="#aaaaaa", fontsize=9)
    ax1.set_ylabel("ADC Count (0 – 4095)", color="#aaaaaa", fontsize=9)
    ax1.set_title("Raw 12-bit ADC Values", color="white", fontsize=11,
                  fontweight="bold")
    ax1.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333344")
    ax1.legend(fontsize=8, facecolor="#22253a", labelcolor="white",
               edgecolor="#444466", loc="upper right")

    # ── Bottom: Pedal Travel % ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#1a1d27")
    ax2.plot(time_axis, pedal_raw, color="#ffe082", linewidth=0.5,
             alpha=0.5, label="Normalised (raw)")
    ax2.plot(time_axis, pedal_smooth, color="#ff7043", linewidth=1.4,
             label=f"Smoothed (MA window={MOVING_AVG_WIN})")
    ax2.fill_between(time_axis, 0, pedal_smooth,
                     color="#ff7043", alpha=0.12)
    ax2.set_xlim(0, len(adc_raw) - 1)
    ax2.set_ylim(-5, 105)
    ax2.set_xlabel("Sample Index", color="#aaaaaa", fontsize=9)
    ax2.set_ylabel("Pedal Travel (%)", color="#aaaaaa", fontsize=9)
    ax2.set_title("Normalised & Smoothed Pedal Travel Percentage",
                  color="white", fontsize=11, fontweight="bold")
    ax2.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333344")
    ax2.legend(fontsize=8, facecolor="#22253a", labelcolor="white",
               edgecolor="#444466", loc="upper right")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "apps2_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {out_path}")
    plt.show()

# ── 5. Statistical Analysis ───────────────────────────────────────────────────
def print_statistics(pedal_smooth: np.ndarray) -> None:
    print("\n" + "═" * 46)
    print("  PEDAL TRAVEL STATISTICS  (smoothed signal)")
    print("═" * 46)
    print(f"  Maximum  : {pedal_smooth.max():7.2f} %")
    print(f"  Minimum  : {pedal_smooth.min():7.2f} %")
    print(f"  Average  : {pedal_smooth.mean():7.2f} %")
    print(f"  Std Dev  : {pedal_smooth.std():7.2f} %")
    print("═" * 46)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, FILE_NAME),
        os.path.join(os.getcwd(),  FILE_NAME),
    ]
    filepath = next((p for p in candidates if os.path.isfile(p)), None)

    if filepath is None:
        print(f"[ERROR] '{FILE_NAME}' not found.\n"
              f"  Looked in:\n" +
              "\n".join(f"    {p}" for p in candidates))
        sys.exit(1)

    print(f"\n{'═'*46}")
    print(f"  APPS2 Sensor Log Analysis")
    print(f"{'═'*46}")
    print(f"  File     : {filepath}")
    print(f"  ADC_MIN  : {ADC_MIN}  ({V_MIN} V)")
    print(f"  ADC_MAX  : {ADC_MAX}  ({V_MAX} V)")
    print(f"  MA Win   : {MOVING_AVG_WIN} samples\n")

    adc_raw      = load_adc_values(filepath)
    pedal_raw    = normalize_to_pedal_travel(adc_raw)
    pedal_smooth = moving_average(pedal_raw, MOVING_AVG_WIN)

    print_statistics(pedal_smooth)
    plot_results(adc_raw, pedal_raw, pedal_smooth)


if __name__ == "__main__":
    main()