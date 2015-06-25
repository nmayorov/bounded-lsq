"""Trust Region Reflective algorithm for least-squares optimization."""


from __future__ import division

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd

from .bounds import (step_size_to_bounds, make_strictly_feasible,
                     check_bounds, CL_scaling)
from .trust_region import get_intersection, solve_lsq_trust_region
from .helpers import EPS, check_tolerance, prepare_OptimizeResult


def minimize_quadratic(a, b, l, u):
    """Minimize a 1-d quadratic function subject to bounds.

    The free term is omitted, that is we consider y = a * t**2 + b * t.

    Returns
    -------
    t : float
        The minimum point.
    y : float
        The minimum value.
    """
    t = np.array([l, u])
    if a != 0:
        extremum = -0.5 * b / a
        if l <= extremum <= u:
            t = np.hstack((t, extremum))
    y = a * t**2 + b * t
    i = np.argmin(y)
    return t[i], y[i]


def build_1d_quadratic_function(J, diag, g, s, s0=None):
    """Compute coefficients of a 1-d quadratic function for the line search
    from the multidimensional quadratic function.

    The function is given as follows:
    ``f(t) = 0.5 * (s0 + t*s).T * (J.T*J + diag) * (s0 + t*s) +
             g.T * (s0 + t*s)``.

    Parameters
    ----------
    J : array, shape (m, n)
        Jacobian matrix.
    diag : array, shape (n,)
        Addition diagonal term in a quadratic part.
    g : array, shape (n,)
        Gradient, defines a linear term.
    s : array, shape (n,)
        Direction of search.
    s0 : None or array, shape (n,), optional
        Initial point. If None assumed to be 0.

    Returns
    ------
    a : float
        Coefficient for t**2.
    b : float
        Coefficient for t.

    Notes
    -----
    The free term "c" is not returned as it is not usually required.
    """
    v = J.dot(s)
    a = 0.5 * (np.dot(v, v) + np.dot(s * diag, s))
    b = np.dot(g, s)
    if s0 is not None:
        u = J.dot(s0)
        b += np.dot(u, v) + np.dot(s0 * diag, s)

    return a, b


def evaluate_quadratic_function(J, diag, g, steps):
    """Compute values of a quadratic function arising in least-squares.

    The function is s.T * (J.T * J + diag) * s + g.T * s.

    Parameters
    ----------
    J : array, shape (m, n)
        Jacobian matrix.
    diag : array, shape (n,)
        Additional diagonal term.
    f : array, shape (m,)
        Vector of residuals.
    steps : array, shape (k, m)
        Array containing k steps as rows.

    Returns
    -------
    values : array, shape (k,)
        Array containing k values of the function.
    """
    Js = J.dot(steps.T)
    return 0.5 * (np.sum(Js**2, axis=0) +
                  np.sum(diag * steps**2, axis=1)) + np.dot(steps, g)


def find_reflective_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, l, u, theta):
    """Find a single reflection step for Trust Region Reflective algorithm.

    Also corrects the initial step p/p_h. This function must be called only
    if x + p is not within the bounds.
    """
    p_stride, hits = step_size_to_bounds(x, p, l, u)

    # Compute the reflected direction.
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict the p step to exactly the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Reflected direction will cross first either feasible region or trust
    # region boundary.
    _, stride_to_tr = get_intersection(p_h, r_h, Delta)

    stride_to_bound, _ = step_size_to_bounds(x_on_bound, r, l, u)
    stride_to_bound *= theta  # Stay strictly interior.

    r_stride_u = min(stride_to_bound, stride_to_tr)

    # We want a reflected step be at the same theta distance from the bound,
    # so we introduce a lower bound on the allowed stride.
    # The formula below is correct as p_h and r_h has the same norm.

    with np.errstate(divide='ignore'):
        r_stride_l = (1 - theta) * p_stride / r_stride_u

    # Check if no reflection step is available.
    if r_stride_l < 0 or r_stride_l > r_stride_u:
        r_h = None
    else:
        a, b = build_1d_quadratic_function(J_h, diag_h, g_h, r_h, s0=p_h)
        r_stride, _ = minimize_quadratic(a, b, r_stride_l, r_stride_u)
        r_h = p_h + r_h * r_stride

    # Now we want to correct p_h to make it strictly interior.
    p_h *= theta

    # If no reflection step just return p_h for convenience.
    if r_h is None:
        return p_h, p_h
    else:
        return p_h, r_h


def find_gradient_step(x, J_h, diag_h, g_h, d, Delta, l, u, theta):
    """Find a minimizer of a quadratic model along scaled gradient."""
    stride_to_bound, _ = step_size_to_bounds(x, -g_h * d, l, u)
    stride_to_bound *= theta

    stride_to_tr = Delta / norm(g_h)
    g_stride = min(stride_to_bound, stride_to_tr)

    a, b = build_1d_quadratic_function(J_h, diag_h, g_h, -g_h)
    g_stride, _ = minimize_quadratic(a, b, 0.0, g_stride)

    return -g_stride * g_h


def trf(fun, jac, x0, bounds=(None, None), ftol=EPS**0.5, xtol=EPS**0.5,
        gtol=EPS**0.5, max_nfev=None, scaling=1.0):
    """Minimize the sum of squares with bounds on independent variables
    by Trust Region Reflective algorithm.

    Let f(x) maps from R^n to R^m, the function finds a local minimum of
    ``||f(x)||**2 = sum(f_i(x)**2, i = 1, ...,m) s. t. l <= x <= u``.

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
        Tolerance for termination by the norm of scaled gradient.
    max_nfev : None or int, optional
        Max number of function evaluations before the termination. If None,
        then it is assigned to 100 * n.
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
        Objective value at the solution.
    nfev : int
        The number of function evaluations performed.
    """
    l, u, feasible = check_bounds(x0, bounds)
    if not feasible:
        raise ValueError("`x0` is infeasible.")

    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol)

    x = make_strictly_feasible(x0, l, u, rstep=1e-10)

    f = fun(x)
    nfev = 1

    J = jac(x)
    njac = 1

    g = J.T.dot(f)
    m, n = J.shape

    if scaling == 'auto':
        J_norm = np.linalg.norm(J, axis=0)
        J_norm[J_norm == 0] = 1
        scale = 1 / J_norm
    else:
        scale = np.asarray(scaling)

    d_CL, jv = CL_scaling(x, g, l, u)
    Delta = norm(x0 / (scale * d_CL))
    if Delta == 0:
        Delta = 1.0

    J_extended = np.empty((m + n, n))
    f_extended = np.zeros((m + n))

    obj_value = np.dot(f, f)
    alpha = 0.0

    if max_nfev is None:
        max_nfev = 100 * n

    nit = 0
    termination_status = None
    while nfev < max_nfev:
        nit += 1
        if scaling == 'auto':
            J_norm = np.linalg.norm(J, axis=0)
            with np.errstate(divide='ignore'):
                scale = np.minimum(scale, 1 / J_norm)

        g = J.T.dot(f)

        d_CL, jv = CL_scaling(x, g, l, u)
        d = d_CL * scale
        g_h = d * g
        diag_h = g * jv * scale**2

        g_norm = norm(d * g_h, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if termination_status is not None:
            return prepare_OptimizeResult(x, f, J, l, u, obj_value, g_norm,
                                          nfev, njac, nit, termination_status)

        theta = max(0.995, 1 - g_norm)

        J_h = J * d

        J_extended[:m] = J_h
        J_extended[m:] = np.diag(diag_h**0.5)
        f_extended[:m] = f

        U, s, VT = svd(J_extended, full_matrices=False)
        V = VT.T
        uf = U.T.dot(f_extended)

        actual_reduction = -1
        while nfev < max_nfev and actual_reduction < 0:
            p_h, alpha, n_iter = solve_lsq_trust_region(
                n, m, uf, s, V, Delta, initial_alpha=alpha)
            p = d * p_h

            stride_to_bounds, _ = step_size_to_bounds(x, p, l, u)
            if stride_to_bounds >= 1:
                p_h *= min(theta * stride_to_bounds, 1)
                steps = np.atleast_2d(p_h)
            else:
                p_h, r_h = find_reflective_step(
                    x, J_h, diag_h, g_h, p, p_h, d, Delta, l, u, theta)
                g_step = find_gradient_step(
                    x, J_h, diag_h, g_h, d, Delta, l, u, theta)
                steps = np.array([p_h, r_h, g_step])

            qp_values = evaluate_quadratic_function(J_h, diag_h, g_h, steps)
            i = np.argmin(qp_values)
            step_h = steps[i]
            predicted_reduction = -qp_values[i]

            step = d * step_h
            x_new = make_strictly_feasible(x + step, l, u)

            nfev += 1
            f_new = fun(x_new)

            obj_value_new = np.dot(f_new, f_new)
            actual_reduction = obj_value - obj_value_new
            correction = np.dot(step_h * diag_h, step_h)

            if predicted_reduction > 0:
                ratio = (0.5 * (actual_reduction - correction) /
                         predicted_reduction)
            else:
                ratio = 0

            if ratio < 0.25:
                Delta_new = 0.25 * norm(step_h)
                alpha *= Delta / Delta_new
                Delta = Delta_new
            elif ratio > 0.75 and norm(step_h) > 0.95 * Delta:
                Delta *= 2.0
                alpha *= 0.5

            if abs(actual_reduction) < ftol * obj_value:
                termination_status = 2
                break

            if norm(step) < xtol * max(EPS**0.5, norm(x)):
                termination_status = 3
                break

        x = x_new
        f = f_new
        obj_value = obj_value_new

        J = jac(x)
        njac += 1

    return prepare_OptimizeResult(x, f, J, l, u, obj_value, g_norm,
                                  nfev, njac, nit, 0)
