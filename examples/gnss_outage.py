"""Fuse IMU + GNSS, survive a 20 s GNSS outage, and compare with IMU-only dead reckoning.

    python examples/gnss_outage.py --plot   # writes docs/demo.png
"""
import argparse

import numpy as np

from ekf import run, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--plot", action="store_true")
args = ap.parse_args()

data = simulate(seed=0)
fused, dr = run(data, use_gnss=True), run(data, use_gnss=False)
t, truth = data["t"], data["truth"]
err = lambda r: np.linalg.norm(r["est"][:, :2] - truth[:, :2], axis=1)
o = (t >= data["outage"][0]) & (t < data["outage"][1])

print(f"position RMSE  fused: {np.sqrt(np.mean(err(fused)**2)):6.2f} m   IMU only: {np.sqrt(np.mean(err(dr)**2)):8.2f} m")
print(f"max error during outage (fused): {err(fused)[o].max():.2f} m")
print(f"estimated gyro bias: {fused['est'][-1, 5]:.4f} rad/s (true {data['spec'].gyro_bias})")
print(f"mean NIS: {fused['nis'].mean():.2f} (expected ~2 for 2-D GNSS fixes)")

if args.plot:
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(truth[:, 0], truth[:, 1], "k", lw=1, label="truth")
    a.plot(fused["est"][:, 0], fused["est"][:, 1], label="EKF (IMU + GNSS)")
    a.plot(dr["est"][:, 0], dr["est"][:, 1], alpha=0.6, label="IMU only")
    g = ~np.isnan(data["gnss"][:, 0])
    a.plot(data["gnss"][g, 0], data["gnss"][g, 1], ".", ms=3, label="GNSS fixes")
    a.set_aspect("equal"); a.legend(fontsize=8); a.set_title("trajectory (20 s GNSS outage)")
    a.set_xlim(truth[:, 0].min() - 50, truth[:, 0].max() + 50); a.set_ylim(truth[:, 1].min() - 50, truth[:, 1].max() + 50)
    b.plot(t, err(fused), label="position error")
    b.plot(t, 3 * np.hypot(fused["sigma"][:, 0], fused["sigma"][:, 1]), "--", label="3σ bound")
    b.axvspan(*data["outage"], color="0.9", label="outage")
    b.set_xlabel("time (s)"); b.set_ylabel("m"); b.legend(fontsize=8); b.set_title("error vs filter uncertainty")
    fig.tight_layout(); fig.savefig("docs/demo.png", dpi=100)
    print("wrote docs/demo.png")
