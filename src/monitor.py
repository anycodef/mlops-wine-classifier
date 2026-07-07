"""Monitoring stage: simulate post-deployment accuracy decay (data drift).

Generates a synthetic daily-accuracy series over a horizon and compares it to a
business retraining threshold, reporting the first day the model would fall
below it. In production this series would come from live traffic metrics; here
it is deterministic (seeded) so the report is reproducible.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import DECAY_PNG, ensure_dirs, load_params  # noqa: E402


def simulate(threshold: float, horizon: int, seed: int):
    rng = np.random.default_rng(seed)
    days = np.arange(horizon + 1)
    # Linear downward drift from ~0.96 plus small daily noise.
    trend = 0.96 - 0.008 * days
    accuracy = np.clip(trend + rng.normal(0, 0.012, size=days.shape), 0, 1)

    below = np.where(accuracy < threshold)[0]
    cross_day = int(below[0]) if below.size else None
    return days, accuracy, cross_day


def main() -> None:
    ensure_dirs()
    p = load_params()["monitor"]
    days, accuracy, cross_day = simulate(
        p["retrain_threshold"], p["horizon_days"], p["random_state"]
    )

    plt.figure(figsize=(8, 4))
    plt.plot(days, accuracy, marker="o", label="daily accuracy")
    plt.axhline(p["retrain_threshold"], color="red", linestyle="--",
                label="retrain threshold")
    plt.title("Model decay simulation")
    plt.xlabel("Days since deployment")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DECAY_PNG, dpi=120)
    plt.close()

    print(f"[monitor] decay chart -> {DECAY_PNG}")
    if cross_day is not None:
        print(f"[monitor] accuracy drops below "
              f"{p['retrain_threshold']} on day {cross_day} -> trigger retraining")
    else:
        print("[monitor] model stayed above threshold over the horizon")


if __name__ == "__main__":
    main()
