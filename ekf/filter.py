"""Loosely coupled GNSS/INS extended Kalman filter in 2D.

State x = [px, py, vx, vy, yaw, gyro_bias].
Prediction uses IMU specific force (body frame) and yaw rate; GNSS position
fixes correct the state. The gyro bias is estimated online.
"""
from __future__ import annotations

import numpy as np


def f(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    px, py, vx, vy, yaw, bg = x
    ax_b, ay_b, w = u
    c, s = np.cos(yaw), np.sin(yaw)
    ax, ay = c * ax_b - s * ay_b, s * ax_b + c * ay_b  # rotate to the navigation frame
    return np.array([px + vx * dt, py + vy * dt, vx + ax * dt, vy + ay * dt, yaw + (w - bg) * dt, bg])


def F_jac(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    yaw = x[4]
    ax_b, ay_b, _ = u
    c, s = np.cos(yaw), np.sin(yaw)
    J = np.eye(6)
    J[0, 2] = J[1, 3] = dt
    J[2, 4] = (-s * ax_b - c * ay_b) * dt
    J[3, 4] = (c * ax_b - s * ay_b) * dt
    J[4, 5] = -dt
    return J


H = np.zeros((2, 6))
H[0, 0] = H[1, 1] = 1.0


class GnssInsEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, accel_noise: float, gyro_noise: float, gnss_noise: float,
                 bias_walk: float = 1e-5):
        self.x, self.P = x0.astype(float), P0.astype(float)
        self.qa, self.qg, self.qb = accel_noise**2, gyro_noise**2, bias_walk**2
        self.R = np.eye(2) * gnss_noise**2

    def predict(self, u: np.ndarray, dt: float) -> None:
        F = F_jac(self.x, u, dt)
        G = np.zeros((6, 4))  # noise input: body accel (2), gyro, bias walk
        yaw = self.x[4]
        c, s = np.cos(yaw), np.sin(yaw)
        G[2:4, 0:2] = np.array([[c, -s], [s, c]]) * dt
        G[4, 2] = dt
        G[5, 3] = dt
        Qc = np.diag([self.qa, self.qa, self.qg, self.qb])
        self.x = f(self.x, u, dt)
        self.P = F @ self.P @ F.T + G @ Qc @ G.T

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form keeps P symmetric positive definite
        return y, S


def run(data: dict, use_gnss: bool = True) -> dict:
    spec, truth = data["spec"], data["truth"]
    x0 = np.r_[truth[0], 0.0]
    P0 = np.diag([4, 4, 1, 1, 0.01, 0.02**2])
    ekf = GnssInsEKF(x0, P0, spec.accel_noise, spec.gyro_noise, spec.gnss_noise)
    n = len(data["t"])
    est, sig, nis = np.zeros((n, 6)), np.zeros((n, 6)), []
    for k in range(n):
        if k > 0:
            ekf.predict(data["imu"][k - 1], spec.dt)
        if use_gnss and not np.isnan(data["gnss"][k, 0]):
            y, S = ekf.update(data["gnss"][k])
            nis.append(float(y @ np.linalg.solve(S, y)))
        est[k], sig[k] = ekf.x, np.sqrt(np.diag(ekf.P))
    return {"est": est, "sigma": sig, "nis": np.array(nis)}
