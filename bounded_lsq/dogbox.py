from __future__ import division

import numpy as np
from numpy.linalg import lstsq, norm
from .bounds import step_size_to_bound, in_bounds, check_bounds


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
    """Find dogleg step in rectangular constraints.

    Returns
    -------
    step : ndarray, shape (n,)
        Computed dogleg step.
    bound_hits : ndarray of int, shape (n,)
        Each component shows whether a corresponding variable reaches the
        initial bound after the step is taken:
             0 - a variable doesn't reach the bound.
            -1 - `l` is reached.
             1 - `u` is reached.
    tr_hit : bool
        Whether the step reaches the boundary of the trust-region.
    """
    l_total, u_total, l_bound, u_bound, l_tr, u_tr = find_intersection(
        x, tr_bounds, l, u
    )

    bound_hits = np.zeros_like(x, dtype=int)

    if in_bounds(newton_step, l_total, u_total):
        return newton_step, bound_hits, False

    if not in_bounds(cauchy_step, l_total, u_total):
        alpha, _ = step_size_to_bound(
            np.zeros_like(cauchy_step), cauchy_step, l_total, u_total)
        cauchy_step = alpha * cauchy_step

    step_diff = newton_step - cauchy_step
    alpha, hits = step_size_to_bound(cauchy_step, step_diff,
                                      l_total, u_total)
    bound_hits[(hits < 0) & l_bound] = -1
    bound_hits[(hits > 0) & u_bound] = 1
    tr_hit = np.any((hits < 0) & l_tr | (hits > 0) & u_tr)

    return cauchy_step + alpha * step_diff, bound_hits, tr_hit


def constrained_cauchy_step(x, cauchy_step, tr_bounds, l, u):
    """Find constrained Cauchy step.

    Returns are the same as in dogleg_step function.
    """
    l_total, u_total, l_bound, u_bound, l_tr, u_tr = find_intersection(
        x, tr_bounds, l, u
    )
    bound_hits = np.zeros_like(x, dtype=int)
    if in_bounds(cauchy_step, l_total, u_total):
        return cauchy_step, bound_hits, False

    beta, hits = step_size_to_bound(
        np.zeros_like(cauchy_step), cauchy_step, l_total, u_total)

    bound_hits[(hits < 0) & l_bound] = -1
    bound_hits[(hits > 0) & u_bound] = 1
    tr_hit = np.any((hits < 0) & l_tr | (hits > 0) & u_tr)

    return beta * cauchy_step, bound_hits, tr_hit


def dogbox(fun, jac, x0, l, u, ftol, xtol, gtol, max_nfev, scaling):
    """Minimize the sum of squares with bounds on independent variables
    by rectangular trust-region dogleg algorithm [1]_.

    Let f(x) maps from R^n to R^m, the function finds a local minimum of
    F(x) = ||f(x)||**2 = sum(f_i(x)**2, i = 1, ...,m),
    subject to bound constraints l <= x <= u

    Parameters
    ----------
    fun : callable
        Returns a 1d-array of residuals of size m.
    jac : callable
        Returns an m-by-n array containing partial derivatives of f with
        respect to x, known as Jacobian matrix.
    x0 : array-like, shape (n,)
        Initial guess on the independent variables.
    bounds : tuple of array-like, optional
        Lower and upper bounds on independent variables. None means that
        there is no lower/upper bound on any of the variables.
    ftol : float, optional
        Tolerance for termination by the change of the objective value.
        Default is square root of machine epsilon. The optimization process
        is stopped when ``dF < ftol * F``, where dF is a change of the
        objective value in the last iteration.
    xtol : float. optional
        Tolerance for termination by the change of the independent variables.
        Default is square root of machine epsilon. The optimization process
        is stopped when ``Delta < xtol * max(EPS**0.5, norm(scaled_x))``,
        where Delta is a trust-region radius, scaled_x is a scaled value
        of x (see `scaling` below), EPS is machine epsilon.
    gtol : float, optional
        Tolerance for termination by the norm of gradient with respect
        to variables which isn't on the boundary in the final solution.
        Default is square root of machine epsilon. The optimization process
        is stopped when ``norm(g, ord=np.inf) < gtol``, where g is the
        gradient of objective function at the current iterate. If all
        variables reach optimum on the boundary, then g is effectively
        assigned to zero and the algorithm terminates.
    max_nfev : None or int, optional
        Maximum number of function evaluations before the termination.
        If None (default), it is assigned to 100 * n.
    scaling : array-like or 'auto', optional
        Determines scaling of the variables. Default is 1.0 which means no
        scaling. A bigger value for some variable means that this variable can
        change stronger during iterations, compared to other variables.
        A scalar value won't affect the algorithm (except maybe
        fixing/introducing numerical problems and changing termination
        criteria). If 'auto', then scaling is inversely proportional to the
        norm of Jacobian columns.

    Returns
    -------
    OptimizeResult with the following fields defined.
    x : ndarray, shape (n,)
        Found solution.
    obj_value : float
        Sum of squares at the solution.
    fun : ndarray, shape (m,)
        Vector of residuals at the solution.
    jac : ndarray, shape (m, n)
        Jacobian at the solution.
    optimality : float
        Firs-order optimality measure. Uniform norm of a gradient with respect
        to variables which aren't on the boundary. This quantity was compared
        with `gtol` during iterations.
    active_mask : ndarray of bool, shape (n,)
        Each component shows whether the corresponding constraint is active:
             0 - a constraint is not active.
            -1 - a lower bound is active.
             1 - an upper bound is active.
        Very accurate as the algorithm tracks active constraints during
        iterations.
    nfev : int
        Number of function evaluations done.
    njac : int
        Number of Jacobian evaluations done.
    nit : int
        Number of main iterations done.
    status : int
        Reason for algorithm termination:
            - 0 - maximum number of function evaluations reached.
            - 1 - `gtol` convergence test is satisfied.
            - 2 - `ftol` convergence test is satisfied.
            - 3 - `xtol` convergence test is satisfied.
    message : string
        Verbal description of the termination reason.
    success : int
        True if one of the convergence criteria is satisfied.

    Notes
    -----
    The algorithm operates in a trust-region framework, but considers
    rectangular trust regions as opposed to conventional elliptical.
    The intersection of the current trust region and initial bounds is again
    rectangular, so on each iteration a quadratic minimization problem subject
    to bounds is solved. Powell's dogleg method [2]_ is applied to solve these
    subproblems. The algorithm exhibits slow convergence when the rank of
    Jacobian is less than the number of variables.

    References
    ----------
    .. [1] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region Dogleg
           Approach for Unconstrained and Bound Constrained Nonlinear
           Optimization", WSEAS International Conference on Applied
           Mathematics, Corfu, Greece, 2004.
    .. [2] J. Nocedal and S. J. Wright, "Numerical optimization, 2nd edition",
           Chapter 4.
    """
    EPS = np.finfo(float).eps

    f = fun(x0)
    nfev = 1

    J = jac(x0, f)
    njev = 1

    if scaling == 'auto':
        J_norm = np.linalg.norm(J, axis=0)
        J_norm[J_norm == 0] = 1
        scale = 1 / J_norm
    else:
        scale = np.asarray(scaling)

    if scale.ndim == 0:
        scale = np.full_like(x0, scale)

    Delta = np.linalg.norm(x0 / scale, ord=np.inf)
    if Delta == 0:
        Delta = 1.0

    on_bound = np.zeros_like(x0, dtype=int)
    on_bound[np.equal(x0, l)] = -1
    on_bound[np.equal(x0, u)] = 1

    x = x0.copy()
    step = np.empty_like(x0)
    obj_value = np.dot(f, f)

    nit = 0
    termination_status = None
    while nfev < max_nfev:
        nit += 1
        g = J.T.dot(f)

        if scaling == 'auto':
            J_norm = np.linalg.norm(J, axis=0)
            with np.errstate(divide='ignore'):
                scale = np.minimum(scale, 1 / J_norm)

        active_set = on_bound * g < 0
        free_set = ~active_set

        J_free = J[:, free_set]
        g_free = g[free_set]
        x_free = x[free_set]
        l_free = l[free_set]
        u_free = u[free_set]
        scale_free = scale[free_set]

        if np.all(active_set):
            g_norm = 0.0
            termination_status = 1
        else:
            g_norm = norm(g_free, ord=np.inf)
            if g_norm < gtol:
                termination_status = 1

        if termination_status is not None:
            return (x, f, J, obj_value, g_norm,
                    nfev, njev, nit, termination_status, on_bound)

        # Compute (Gauss-)Newton and Cauchy steps
        newton_step = lstsq(J_free, -f)[0]
        Jg = J_free.dot(g_free)
        cauchy_step = -np.dot(g_free, g_free) / np.dot(Jg, Jg) * g_free

        actual_reduction = -1.0
        while actual_reduction < 0 and nfev < max_nfev:
            tr_bounds = Delta * scale_free

            step_free, on_bound_free, tr_hit = dogleg_step(
                x_free, cauchy_step, newton_step, tr_bounds, l_free, u_free)

            Js = J_free.dot(step_free)
            predicted_reduction = -np.dot(Js, Js) - 2 * np.dot(Js, f)

            # In (nearly) rank deficient case Newton (and thus dogleg) step
            # can be inadequate, in this case use (constrained) Cauchy step.
            if predicted_reduction <= 0:
                step_free, on_bound_free, tr_hit = constrained_cauchy_step(
                    x_free, cauchy_step, tr_bounds, l_free, u_free)
                predicted_reduction = -np.dot(Js, Js) - 2 * np.dot(Js, f)

            step.fill(0.0)
            step[free_set] = step_free
            x_new = x + step

            f_new = fun(x_new)
            nfev += 1

            # Usual trust-region step quality estimation.
            obj_value_new = np.dot(f_new, f_new)
            actual_reduction = obj_value - obj_value_new

            if predicted_reduction > 0:
                ratio = actual_reduction / predicted_reduction
            else:
                ratio = 0

            if ratio < 0.25:
                Delta = 0.25 * norm(step / scale, ord=np.inf)
            elif ratio > 0.75 and tr_hit:
                Delta *= 2.0

            if abs(actual_reduction) < ftol * obj_value:
                termination_status = 2
                break

            if Delta < xtol * max(EPS**0.5, norm(x / scale, ord=np.inf)):
                termination_status = 3
                break

        if actual_reduction > 0:
            on_bound[free_set] = on_bound_free

            x = x_new
            # Set variables exactly at the boundary.
            mask = on_bound == -1
            x[mask] = l[mask]
            mask = on_bound == 1
            x[mask] = u[mask]

            f = f_new
            obj_value = obj_value_new

            J = jac(x, f)
            njev += 1

    return x, f, J, obj_value, g_norm, nfev, njev, nit, 0, on_bound
