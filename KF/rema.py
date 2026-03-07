# BMS Current Filter — LEM HO-250-S Hall-Effect Sensor
# EMA formula: y[n] = (1 - alpha) * y[n-1] + alpha * x[n]
#
# Alpha trade-off:
#   low  (0.1–0.3) → heavy smoothing, high phase lag   (SOC estimation)
#   mid  (0.4–0.6) → balanced                          (general monitoring)
#   high (0.7–0.9) → light smoothing, low phase lag    (transient detection)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


def load_current_data(csv_path: str | Path, column: str = "raw_current") -> pd.Series:
    """Read 'raw_current' column from CSV. Raises if file or column is missing."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Got: {list(df.columns)}")

    return df[column].reset_index(drop=True)


def ema_filter(signal: pd.Series | np.ndarray, alpha: float) -> np.ndarray:
    """
    First-order IIR EMA filter.
      y[0] = x[0]  — warm-start avoids a startup transient
      y[n] = (1 - alpha) * y[n-1] + alpha * x[n]
    alpha must be in (0, 1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    x = np.asarray(signal, dtype=float)
    y = np.empty(len(x), dtype=float)
    y[0] = x[0]  # initialise with first sample, not zero

    one_minus_alpha = 1.0 - alpha
    for i in range(1, len(x)):
        y[i] = one_minus_alpha * y[i - 1] + alpha * x[i]

    return y


def plot_filter_comparison(
    raw: pd.Series | np.ndarray,
    filtered: np.ndarray,
    alpha: float,
    sample_rate_hz: float = 1.0,
    save_path: str | Path | None = None,
) -> None:
    """Two-panel plot: raw vs filtered (top) and residual/noise (bottom)."""
    raw = np.asarray(raw, dtype=float)
    time = np.arange(len(raw)) / sample_rate_hz
    x_label = "Time (s)" if sample_rate_hz != 1.0 else "Sample index"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Raw vs filtered overlay
    axes[0].plot(time, raw, color="#58a6ff", linewidth=0.8, alpha=0.55, label="Raw current")
    axes[0].plot(time, filtered, color="#f78166", linewidth=1.8, label=f"EMA filtered (α={alpha})")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("LEM HO-250-S — EMA Low-Pass Filter: Raw vs. Filtered")
    axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", loc="upper right")
    axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    axes[0].grid(True, color="#21262d", linewidth=0.6)

    # Residual — isolated noise component
    axes[1].plot(time, raw - filtered, color="#3fb950", linewidth=0.7, alpha=0.8, label="Residual (raw − filtered)")
    axes[1].axhline(0, color="#6e7681", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Residual (A)")
    axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    axes[1].grid(True, color="#21262d", linewidth=0.6)

    lag_note = ("Heavy smoothing, higher phase lag" if alpha < 0.3
                else "Light smoothing, lower phase lag" if alpha > 0.7
                else "Balanced smoothing / phase lag")
    fig.text(0.5, 0.01, f"α = {alpha:.2f}  |  {lag_note}",
             ha="center", fontsize=9, color="#8b949e")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {Path(save_path).resolve()}")
    else:
        plt.show()

    plt.close(fig)


def main(
    csv_path: str | Path = "current_data.csv",
    alpha: float = 0.25,        # smoothing factor — tune per application
    sample_rate_hz: float = 100.0,
) -> None:
    csv_path = Path(csv_path)

    raw_current = load_current_data(csv_path, column="INV_DC_Bus_Current")
    print(f"Loaded {len(raw_current)} samples.")

    filtered_current = ema_filter(raw_current, alpha=alpha)
    print(f"EMA filter applied (α={alpha}).")

    # Save raw + filtered to a new CSV
    out_csv = csv_path.with_stem(csv_path.stem + "_filtered")
    pd.DataFrame({"INV_DC_Bus_Current": raw_current.values,
                  "filtered_current": filtered_current}).to_csv(out_csv, index=False)
    print(f"Filtered data saved → {out_csv.resolve()}")

    plot_filter_comparison(
        raw=raw_current,
        filtered=filtered_current,
        alpha=alpha,
        sample_rate_hz=sample_rate_hz,
        save_path=csv_path.with_stem(csv_path.stem + "_plot").with_suffix(".png"),
    )


if __name__ == "__main__":
    main(
        csv_path=r"C:\Users\PRANAV\OneDrive\Desktop\RAFT\current_data.csv",
        alpha=0.25,
        sample_rate_hz=100.0,
    )