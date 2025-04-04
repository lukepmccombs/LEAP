import matplotlib.pyplot as plt
import numpy as np
import scipy
import warnings


def complex_log(v):
    return np.log(v + 0j)


def log_sum_exp(log_vals, axis=-1):
    max_v = np.max(log_vals, axis=axis)
    sum_v = np.sum(np.exp(log_vals - max_v), axis=axis)
    return complex_log(sum_v) + max_v


def log_add_exp(lv1, lv2):
    if np.real(lv1) < np.real(lv2):
        lv1, lv2 = lv2, lv1
    return complex_log(np.exp(lv2-lv1) + 1) + lv1


def shrink_point(point, eps=1e-15, axis=-1):
    # Contracts space along the line [0 0 ... 0] to [1 1 ... 1] about [1/N 1/N ... 1/N]
    # Won't move points in the [0, 1] box outside, while moving points near the border inward
    return point + eps/point.shape[axis] - np.mean(point * eps, axis=axis, keepdims=True)


_has_warned_estimate_p = False


def estimate_p(point, max_iter=100, reg_eps=1e-15, convergence_eps=1e-20):
    # Estimates p such that the given point lies on the Lp unit geodesic
    eq_eps = np.finfo(point.dtype).eps
    point = shrink_point(point, reg_eps)
    if not (np.all(point >= 0) and np.all(point <= 1) and np.any(point > 0) and np.any(point < 1)):
        # For the expected case, where the centermost point is chosen, this should not occur
        global _has_warned_estimate_p
        if not _has_warned_estimate_p:
            _has_warned_estimate_p = True
            warnings.warn(
                "Point in estimate_p does not lie within the [0, 1] box. p will default to 1")
        return 1

    log_point = complex_log(point)
    loglog_point = complex_log(np.real(log_point))
    neg_rot = np.pi * 1j

    l_p = 0
    p = 1
    for _ in range(max_iter):
        p_log_point = np.real(log_point) * p + np.imag(log_point)

        l_dp = log_sum_exp(p_log_point)
        if np.abs(np.real(l_dp)) <= eq_eps:
            break

        l_dp += complex_log(np.real(l_dp))
        l_dp -= log_sum_exp(p_log_point + loglog_point)

        l_p = log_add_exp(l_p, l_dp + neg_rot)
        last_p, p = p, np.real(np.exp(l_p))

        if np.abs(p-last_p) <= convergence_eps:
            break

    return p


def min_axis_dist(points, obj, exclude_indices, eps=1e-8):
    N = points.shape[-1]
    eps_coeff = eps / (1 + (N-1) * eps ** 2)

    def axis_dist(idx, point):
        if idx in exclude_indices:
            return np.inf
        norm = point[obj] + (np.sum(point) - point[obj]) * eps_coeff
        base = (point[obj] - norm) ** 2
        update = (point - eps * norm) ** 2
        return base + np.sum(update) - update[obj]

    return min(enumerate(points), key=lambda iv: axis_dist(*iv))[0]


def hyperplane_normalize(points, eps=1e-8, seed=None):
    assert np.ndim(points) == 2
    N = points.shape[-1]
    if N > points.shape[0]:
        return None

    rng = np.random.default_rng(seed)
    exclude_indices = set()
    hyperplane_points = np.zeros((N, N))
    # Use a random permutation for invariance
    for obj in rng.permutation(N):
        obj_idx = min_axis_dist(points, obj, exclude_indices)
        exclude_indices.add(obj_idx)
        hyperplane_points[obj] = points[obj_idx]

    norm = np.linalg.solve(hyperplane_points, np.ones(N))
    return points * norm


x = np.linspace(1, 10, 100)
points = np.stack((x * 2, (10 - x)), -1)


fig, (ax1, ax2) = plt.subplots(2)
ax1.scatter(*points.T)
ax2.scatter(*hyperplane_normalize(points).T)
plt.show()
