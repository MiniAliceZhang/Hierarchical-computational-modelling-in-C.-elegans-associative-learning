# -*- coding: utf-8 -*-
"""Shared kinematic and autocorrelation helpers for temporal-control analyses."""
import numpy as np


def xy_kinematics(x, y, dt=0.05, window=5, smooth=0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if smooth and smooth > 1:
        k = np.ones(int(smooth), dtype=float) / float(smooth)
        x = np.convolve(x, k, mode="same")
        y = np.convolve(y, k, mode="same")
    T = len(x)
    vx = np.full(T, np.nan)
    vy = np.full(T, np.nan)
    for t in range(window, T - window):
        vx[t] = (x[t + window] - x[t - window]) / (2 * window * dt)
        vy[t] = (y[t + window] - y[t - window]) / (2 * window * dt)
    speed = np.hypot(vx, vy)
    omega = np.full(T, np.nan)
    valid = np.arange(window, T - window)
    theta = np.unwrap(np.arctan2(vy[valid], vx[valid]))
    omega[valid[1:]] = np.diff(theta) / dt
    return speed, omega


def acf(x, max_lag=20):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < max_lag + 2:
        return np.full(max_lag + 1, np.nan)
    x = x - x.mean()
    den = np.sum(x * x)
    if den <= 0:
        return np.full(max_lag + 1, np.nan)
    return np.array([np.sum(x[:-k] * x[k:]) / den if k else 1.0 for k in range(max_lag + 1)])


def omega_metrics(x, y, max_lag=20, smooth=0):
    _, om = xy_kinematics(x, y, smooth=smooth)
    om = np.abs(om)
    return {
        "mean": float(np.nanmean(om)),
        "std": float(np.nanstd(om)),
        "acf": acf(om, max_lag=max_lag),
    }


def speed_profile_from_cache(seqs, bins=50):
    ts, vs = [], []
    for s in seqs:
        n = len(s["speed"])
        ts.append(np.linspace(0, 1, n))
        vs.append(s["speed"])
    t = np.concatenate(ts)
    v = np.concatenate(vs)
    edges = np.linspace(0, 1, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = []
    for i in range(bins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        means.append(float(np.nanmean(v[m])) if m.any() else np.nan)
    means = np.array(means)
    ok = ~np.isnan(means)
    means = np.interp(centers, centers[ok], means[ok])
    return centers, means
