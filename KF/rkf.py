# Kalman Filter — Constant Voltage Estimation
# State model:   x_{k+1} = x_k          (A = 1, voltage is constant)
# Measurement:   z_k = x_k + v_k        (H = 1, direct noisy observation)
#
# Filter equations (scalar form):
#   Predict:  x_k|k-1 = x_{k-1}         P_k|k-1 = P_{k-1} + Q
#   Update:   K_k     = P_k|k-1 / (P_k|k-1 + R)
#             x_k     = x_k|k-1 + K_k * (z_k - x_k|k-1)
#             P_k     = (1 - K_k) * P_k|k-1
#
# Q: process noise covariance  — how much the true state can drift
# R: measurement noise covariance — sensor noise variance
# As K → 0 the filter trusts its own estimate; as K → 1 it trusts the measurement.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


def load_voltage_data(csv_path: str | Path, column: str) -> np.ndarray:
    """Load voltage column from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Got: {list(df.columns)}")
    return df[column].to_numpy(dtype=float)


def kalman_filter(measurements: np.ndarray, Q: float, R: float,
                  x0: float = None, P0: float = 1.0):
    """
    Scalar Kalman Filter for a constant-state model (A=1, H=1).

    Parameters
    ----------
    measurements : array of noisy voltage readings z_k
    Q  : process noise covariance (set near 0 for truly constant voltage)
    R  : measurement noise covariance (estimate from sensor spec / variance)
    x0 : initial state estimate (defaults to first measurement)
    P0 : initial error covariance

    Returns
    -------
    x_est : filtered state estimates
    K_log : Kalman gain at each step
    P_log : error covariance at each step
    """
    n = len(measurements)
    x_est = np.empty(n)
    K_log = np.empty(n)
    P_log = np.empty(n)

    # Initialisation
    x = measurements[0] if x0 is None else x0  # warm-start on first sample
    P = P0

    for k in range(n):
        # ── Predict ──────────────────────────────────────────────
        # x_k|k-1 = A * x_{k-1} = x_{k-1}  (A=1, no change)
        # P_k|k-1 = P_{k-1} + Q
        P = P + Q

        # ── Update ───────────────────────────────────────────────
        K = P / (P + R)                          # Kalman gain
        x = x + K * (measurements[k] - x)       # state update
        P = (1 - K) * P                          # covariance update

        x_est[k] = x
        K_log[k] = K
        P_log[k] = P

    return x_est, K_log, P_log


def plot_results(measurements: np.ndarray, x_est: np.ndarray,
                 K_log: np.ndarray, P_log: np.ndarray,
                 save_path: str | Path | None = None) -> None:
    """Four-panel plot: measured vs filtered, Kalman gain, error covariance, estimation error."""
    n = len(measurements)
    steps = np.arange(n)
    error = measurements - x_est

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(True, color="#21262d", linewidth=0.6)

    # 1 — Measured vs filtered voltage
    axes[0].plot(steps, measurements, color="#58a6ff", linewidth=0.9,
                 alpha=0.55, label="Measured voltage z_k")
    axes[0].plot(steps, x_est, color="#f78166", linewidth=1.8,
                 label="Kalman estimate x̂_k")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Measured vs Kalman-Filtered Voltage")
    axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # 2 — Kalman gain (converges toward 0 as filter gains confidence)
    axes[1].plot(steps, K_log, color="#d2a8ff", linewidth=1.5)
    axes[1].set_ylabel("Kalman Gain K_k")
    axes[1].set_title("Kalman Gain over Time  (converges → 0 as uncertainty drops)")

    # 3 — Error covariance P (reflects remaining estimation uncertainty)
    axes[2].plot(steps, P_log, color="#ffa657", linewidth=1.5)
    axes[2].set_ylabel("Covariance P_k")
    axes[2].set_title("Error Covariance over Time  (converges → Q·R / (Q+R) steady state)")

    # 4 — Estimation error (innovation): z_k - x̂_k
    axes[3].plot(steps, error, color="#3fb950", linewidth=0.8, alpha=0.85)
    axes[3].axhline(0, color="#6e7681", linewidth=0.9, linestyle="--")
    axes[3].set_ylabel("Error (V)")
    axes[3].set_xlabel("Time step k")
    axes[3].set_title("Estimation Error  z_k − x̂_k ")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {Path(save_path).resolve()}")
    else:
        plt.show()

    plt.close(fig)


def main(
    csv_path: str | Path = r"C:\Users\PRANAV\OneDrive\Desktop\RAFT\voltage_data.csv",
    column: str = "Measured_Voltage",
    Q: float = 1e-5,   # process noise — near 0 because voltage is ideally constant
    R: float = 0.1,    # measurement noise — tune to sensor noise variance
    P0: float = 1.0,   # initial error covariance — high = start uncertain
) -> None:
    measurements = load_voltage_data(csv_path, column=column)
    print(f"Loaded {len(measurements)} samples.")

    x_est, K_log, P_log = kalman_filter(measurements, Q=Q, R=R, P0=P0)

    # Print convergence summary
    print(f"\nKalman Gain:   initial={K_log[0]:.4f}  →  final={K_log[-1]:.6f}")
    print(f"Covariance:    initial={P_log[0]:.4f}  →  final={P_log[-1]:.6f}")
    print(f"Estimate:      final x̂ = {x_est[-1]:.4f} V")

    out_csv = Path(csv_path).with_stem(Path(csv_path).stem + "_kalman")
    pd.DataFrame({
        "measured": measurements,
        "kalman_estimate": x_est,
        "kalman_gain": K_log,
        "error_covariance": P_log,
    }).to_csv(out_csv, index=False)
    print(f"Results saved → {out_csv.resolve()}")

    plot_results(
        measurements, x_est, K_log, P_log,
        save_path=Path(csv_path).with_stem(Path(csv_path).stem + "_kalman_plot").with_suffix(".png"),
    )


if __name__ == "__main__":
    main()