from __future__ import division

import numpy as np
from numpy.linalg import lstsq, norm
from .bounds import step_size_to_bounds, in_bounds, check_bounds


def find_intersection(x, tr_bounds, l, u):
    l_centered = l - x
    u_centered = u - x

    l_total = np.maximum(l_centered, -tr_bounds)
    u_total = np.minimum(u_centered, tr_bounds)

    l_bound = np.equal(l_total, l_centered)
    u_bound = np.equal(u_total, u_centered)

    l_tr = np.equal(l_total, -tr_bounds)
    u_tr = np.equal(u_total, tr_bounds)

    return l_total, u_total, l_bound, u_bound, l_tr, u_tr


def dogleg_step(x, cauchy_step, newton_step, tr_bounds, l, u):
    """Find dogleg step in rectangular constraints."""
    l_total, u_total, l_bound, u_bound, l_tr, u_tr = find_intersection(
        x, tr_bounds, l, u
    )

    bound_hits = np.zeros_like(x, dtype=int)

    if in_bounds(newton_step, l_total, u_total):
        return newton_step, bound_hits, False

    if not in_bounds(cauchy_step, l_total, u_total):
        beta, _ = step_size_to_bounds(
            np.zeros_like(cauchy_step), cauchy_step, l_total, u_total)
        cauchy_step = beta * cauchy_step

    step_diff = newton_step - cauchy_step
    alpha, hits = step_size_to_bounds(cauchy_step, step_diff,
                                      l_total, u_total)
    bound_hits[(hits < 0) & l_bound] = -1
    bound_hits[(hits > 0) & u_bound] = 1
    box_hit = np.any((hits < 0) & l_tr | (hits > 0) & u_tr)

    return cauchy_step + alpha * step_diff, bound_hits, box_hit


def constrained_cauchy_step(x, cauchy_step, tr_bounds, l, u):
    """Find constrained Cauchy step in case when Newton step is not
    available."""
    l_total, u_total, l_bound, u_bound, l_tr, u_tr = find_intersection(
        x, tr_bounds, l, u
    )
    bound_hits = np.zeros_like(x, dtype=int)
    if in_bounds(cauchy_step, l_total, u_total):
        return cauchy_step, bound_hits, False

    beta, hits = step_size_to_bounds(
        np.zeros_like(cauchy_step), cauchy_step, l_total, u_total)

    bound_hits[(hits < 0) & l_bound] = -1
    bound_hits[(hits > 0) & u_bound] = 1
    box_hit = np.any((hits < 0) & l_tr | (hits > 0) & u_tr)

    return beta * cauchy_step, bound_hits, box_hit


def dogbox(fun, jac, x0, bounds=(None, None), ftol=1e-5, xtol=1e-5, gtol=1e-3,
           max_nfev=1000, scaling=1.0):
    """Minimize the sum of squares with bounds on independent variables
    by rectangular trust-region dogleg algorithm.

    Let f(x) maps from R^n to R^m, the function finds a local minimum of
    ``||f(x)||**2 = sum(f_i(x)**2, i = 1, ..., m) s. t. l <= x <= u``.

    Parameters
    ----------
    fun : callable
        Returns a 1d-array of residuals of size m.
    jac : callable
        Returns an m-by-n array containing partial derivatives of f with
        respect to x, known as Jacobian matrix.
    x0 : array, shape (n,)
        Initial guess on the independent variables.
    bounds : tuple of array, optional
        Lower and upper bounds on independent variables. None means that
        there is no lower/upper bound on any of the variables.
    ftol : float, optional
        Tolerance for termination by the change of the objective value.
    xtol : float. optional
        Tolerance for termination by the change of the independent variables.
    gtol : float, optional
        Tolerance for termination by the norm of gradient with respect
        to variables which isn't on the boundary in the final solution.
    max_nfev : int, optional
        Max number of function evaluations before the termination.
    scaling : array-like or 'auto', optional
        Determines scaling of the variables. A bigger value for some variable
        means that this variable can change stronger during iterations,
        compared to other variables. A scalar value won't affect the algorithm
        (except maybe fixing/introducing numerical problems). If 'auto', then
        scaling is inversely proportional to the norm of Jacobian columns.

    Returns
    -------
    x : array, shape (n,)
        Found solution.
    obj_value : float
        Objective function value at the solution.
    nfev : int
        The number of function evaluations done.
    """
    l, u, feasible = check_bounds(x0, bounds)
    if not feasible:
        raise ValueError("`x0` is infeasible.")

    nfev = 1
    f = fun(x0)
    J = jac(x0)

    if scaling == 'auto':
        J_norm = np.linalg.norm(J, axis=0)
        J_norm[J_norm == 0] = 1
        scale = 1 / J_norm
    else:
        scale = np.asarray(scaling)

    Delta = np.linalg.norm(x0 / scale, ord=np.inf)
    if Delta == 0:
        Delta = 1.0

    on_bound = np.zeros_like(x0, dtype=int)
    on_bound[np.equal(x0, l)] = -1
    on_bound[np.equal(x0, u)] = 1

    x = x0.copy()
    step = np.empty_like(x0)
    obj_val = np.dot(f, f)
    while nfev < max_nfev:
        g = J.T.dot(f)

        if scaling == 'auto':
            J_norm = np.linalg.norm(J, axis=0)
            with np.errstate(divide='ignore'):
                scale = np.minimum(scale, 1 / J_norm)

        active_set = on_bound * g < 0
        free_set = ~active_set
        if np.all(active_set):
            break

        J_free = J[:, free_set]
        g_free = g[free_set]
        x_free = x[free_set]
        l_free = l[free_set]
        u_free = u[free_set]

        if norm(g_free, ord=np.inf) <= gtol:
            break

        newton_step = lstsq(J_free, -f)[0]
        Jg = J_free.dot(g_free)
        cauchy_step = -np.dot(g_free, g_free) / np.dot(Jg, Jg) * g_free

        actual_change = 1.0
        while nfev < max_nfev and actual_change > 0:
            tr_bounds = Delta * scale
            if tr_bounds.ndim == 1:
                tr_bounds = tr_bounds[free_set]

            step_free, on_bound_free, box_hit = dogleg_step(
                x_free, cauchy_step, newton_step, tr_bounds, l_free, u_free)

            Js = J_free.dot(step_free)
            predicted_change = np.dot(Js, Js) + 2 * np.dot(Js, f)

            # In (nearly) rank deficient case Newton step can be
            # inappropriate, in this case use (constrained) Cauchy step.
            if predicted_change >= 0:
                step_free, on_bound_free, box_hit = constrained_cauchy_step(
                    x_free, cauchy_step, tr_bounds, l_free, u_free)
                predicted_change = np.dot(Js, Js) + 2 * np.dot(Js, f)

            step.fill(0.0)
            step[free_set] = step_free
            x_new = x + step

            f_new = fun(x_new)
            nfev += 1

            obj_val_new = np.dot(f_new, f_new)
            actual_change = obj_val_new - obj_val

            ratio = actual_change / predicted_change

            if ratio < 0.25:
                Delta = 0.25 * min(Delta, norm(step / scale, ord=np.inf))
            elif ratio > 0.75 and box_hit:
                Delta *= 2.0

        x = x_new
        f = f_new
        J = jac(x)
        obj_val = obj_val_new

        if abs(actual_change) < ftol * obj_val or norm(step) < xtol:
            break

        on_bound[free_set] = on_bound_free

    m = on_bound == -1
    x[m] = l[m]

    m = on_bound == 1
    x[m] = u[m]

    return x, obj_val, nfev
