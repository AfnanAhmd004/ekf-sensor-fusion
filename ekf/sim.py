"""Simulate a ground vehicle with an IMU (100 Hz) and a GNSS receiver (1 Hz), including an outage."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorSpec:
    dt: float = 0.01
    accel_noise: float = 0.05     # m/s^2
    gyro_noise: float = 0.002     # rad/s
    gyro_bias: float = 0.01       # rad/s, constant, unknown to the filter
    gnss_noise: float = 1.5       # m
    gnss_period: float = 1.0      # s


def simulate(duration: float = 120.0, outage=(60.0, 80.0), spec: SensorSpec = SensorSpec(), seed: int = 0):
    """Returns a dict of time series: truth state, IMU readings and GNSS fixes (NaN when unavailable)."""
    rng = np.random.default_rng(seed)
    n = int(duration / spec.dt)
    t = np.arange(n) * spec.dt
    yaw_rate = 0.15 * np.sin(2 * np.pi * t / 40)          # gentle S-curves
    speed = 8 + 2 * np.sin(2 * np.pi * t / 30)
    accel_fwd = np.gradient(speed, spec.dt)
    yaw = np.cumsum(yaw_rate) * spec.dt
    vx, vy = speed * np.cos(yaw), speed * np.sin(yaw)
    x, y = np.cumsum(vx) * spec.dt, np.cumsum(vy) * spec.dt
    # body-frame specific force: forward accel and centripetal accel
    ax_b = accel_fwd + rng.normal(0, spec.accel_noise, n)
    ay_b = speed * yaw_rate + rng.normal(0, spec.accel_noise, n)
    gyro = yaw_rate + spec.gyro_bias + rng.normal(0, spec.gyro_noise, n)
    gnss = np.full((n, 2), np.nan)
    every = int(spec.gnss_period / spec.dt)
    idx = np.arange(0, n, every)
    idx = idx[(t[idx] < outage[0]) | (t[idx] >= outage[1])]
    gnss[idx] = np.c_[x[idx], y[idx]] + rng.normal(0, spec.gnss_noise, (len(idx), 2))
    truth = np.c_[x, y, vx, vy, yaw]
    return {"t": t, "truth": truth, "imu": np.c_[ax_b, ay_b, gyro], "gnss": gnss, "spec": spec, "outage": outage}
