import numpy as np

from ekf import F_jac, f, run, simulate


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    x, u, dt = rng.normal(size=6), rng.normal(size=3), 0.01
    J = F_jac(x, u, dt)
    num = np.zeros((6, 6))
    for i in range(6):
        e = np.zeros(6); e[i] = 1e-6
        num[:, i] = (f(x + e, u, dt) - f(x - e, u, dt)) / 2e-6
    assert np.allclose(J, num, atol=1e-6)


def test_fusion_beats_dead_reckoning():
    data = simulate(duration=120, outage=(60, 80), seed=1)
    fused, dr = run(data), run(data, use_gnss=False)
    e = lambda r: np.sqrt(np.mean(np.sum((r["est"][:, :2] - data["truth"][:, :2]) ** 2, 1)))
    assert e(fused) < 3.0 and e(fused) < e(dr) / 5


def test_uncertainty_grows_during_outage_and_shrinks_after():
    data = simulate(duration=60, outage=(30, 40), seed=2)
    sig = run(data)["sigma"][:, 0]
    t = data["t"]
    assert sig[np.searchsorted(t, 39.9)] > 3 * sig[np.searchsorted(t, 29.9)]
    assert sig[np.searchsorted(t, 45)] < sig[np.searchsorted(t, 39.9)]


def test_filter_is_consistent():
    nis = run(simulate(duration=120, seed=3))["nis"]
    assert 1.4 < nis.mean() < 2.8  # chi-square with 2 dof has mean 2


def test_gyro_bias_is_estimated():
    data = simulate(duration=120, seed=4)
    assert abs(run(data)["est"][-1, 5] - data["spec"].gyro_bias) < 0.004
