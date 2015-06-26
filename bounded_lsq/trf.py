"""Trust Region Reflective algorithm for least-squares optimization."""


from __future__ import division

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd

from .bounds import (step_size_to_bound, make_strictly_feasible,
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
    J : ndarray, shape (m, n)
        Jacobian matrix.
    diag : ndarray, shape (n,)
        Addition diagonal term in a quadratic part.
    g : ndarray, shape (n,)
        Gradient, defines a linear term.
    s : ndarray, shape (n,)
        Direction of search.
    s0 : None or ndarray with shape (n,), optional
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

    The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.

    Parameters
    ----------
    J : ndarray, shape (m, n)
        Jacobian matrix.
    diag : ndarray, shape (n,)
        Additional diagonal term.
    f : ndarray, shape (m,)
        Vector of residuals.
    steps : ndarray, shape (k, m)
        Array containing k steps as rows.

    Returns
    -------
    values : ndarray, shape (k,)
        Array containing k values of the function.
    """
    Js = J.dot(steps.T)
    return 0.5 * (np.sum(Js**2, axis=0) +
                  np.sum(diag * steps**2, axis=1)) + np.dot(steps, g)


def find_reflected_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, l, u, theta):
    """Find a single reflection step for Trust Region Reflective algorithm.

    Also corrects the initial step p/p_h. This function must be called only
    if x + p is not within the bounds.
    """
    # Use term "stride" for scalar step length.
    p_stride, hits = step_size_to_bound(x, p, l, u)

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
    _, to_tr = get_intersection(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, l, u)
    to_bound *= theta  # Stay strictly interior.

    r_stride_u = min(to_bound, to_tr)

    # We want a reflected step be at the same theta distance from the bound,
    # so we introduce a lower bound on the allowed stride.
    # The formula below is correct as p_h and r_h has the same norm.

    if r_stride_u > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride_u
    else:
        r_stride_l = -1

    # Check if no reflection step is available.
    if r_stride_l <= r_stride_u:
        a, b = build_1d_quadratic_function(J_h, diag_h, g_h, r_h, s0=p_h)
        r_stride, _ = minimize_quadratic(a, b, r_stride_l, r_stride_u)
        r_h = p_h + r_h * r_stride
    else:
        r_h = None

    # Now we want to correct p_h to make it strictly interior.
    p_h *= theta

    # If no reflection step just return p_h for convenience.
    if r_h is None:
        return p_h, p_h
    else:
        return p_h, r_h


def find_gradient_step(x, J_h, diag_h, g_h, d, Delta, l, u, theta):
    """Find a minimizer of a quadratic model along the scaled gradient."""
    to_bound, _ = step_size_to_bound(x, -g_h * d, l, u)
    to_bound *= theta

    to_tr = Delta / norm(g_h)
    g_stride = min(to_bound, to_tr)

    a, b = build_1d_quadratic_function(J_h, diag_h, g_h, -g_h)
    g_stride, _ = minimize_quadratic(a, b, 0.0, g_stride)

    return -g_stride * g_h


def trf(fun, jac, x0, bounds=(None, None), ftol=EPS**0.5, xtol=EPS**0.5,
        gtol=EPS**0.5, max_nfev=None, scaling=1.0):
    """Minimize the sum of squares with bounds on independent variables
    by Trust Region Reflective algorithm [1]_.

    Let f(x) maps from R^n to R^m, the function finds a local minimum of
    F(x) = ||f(x)||**2 = sum(f_i(x)**2, i = 1, ...,m),
    subject to bound constraints l <= x <= u

    Parameters
    ----------
    fun : callable
        Returns a 1-D array of residuals of size m.
    jac : callable
        Returns an m-by-n array containing partial derivatives of f with
        respect to x, known as Jacobian matrix.
    x0 : array-like, shape (n,)
        Initial guess on the independent variables.
    bounds : tuple of array-like, optional
        Lower and upper bounds on independent variables. None means that
        there is no lower/upper bound on any of the variables. To disable
        a bound on an individual variable use np.inf with the appropriate
        sign.
    ftol : float, optional
        Tolerance for termination by the change of the objective value.
        Default is square root of machine epsilon. The optimization process
        is stopped when ``dF < ftol * F``, where dF is the change of the
        objective value in the last iteration.
    xtol : float, optional
        Tolerance for termination by the change of the independent variables.
        Default is square root of machine epsilon. The optimization process
        is stopped when ``norm(dx) < xtol * max(EPS**0.5, norm(x))``,
        where dx is a step taken in the last iteration and EPS is machine
        epsilon.
    gtol : float, optional
        Tolerance for termination by the norm of scaled gradient. Default is
        square root of machine epsilon. The optimization process is stopped
        when ``norm(g_scaled, ord=np.inf) < gtol``, where g_scaled is
        properly scaled gradient to account for the presence of bounds as
        described in [1]_. The scaling imposed by `scaling` parameter
        (see below) is not considered.
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
        norm of Jacobian columns. This concept is irrelevant to scaling
        suggested in [1]_ for handling the bounds, from the experience it is
        generally not recommended to use ``scaling=auto`` in bounded problems.

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
        Firs-order optimality measure. Uniform norm of scaled gradient. This
        quantity was compared with `gtol` during iterations.
    active_mask : ndarray of bool, shape (n,)
        True means that the corresponding constraint is active at the solution.
        Might be somewhat arbitrary as the algorithm does strictly feasible
        iterations, thus `active_mask` is determined with tolerance threshold.
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
    The algorithm is motivated by the process of solving the equation, which
    constitutes the first-order optimality condition for a bound-constrained
    minimization problem as formulated in [1]_. The algorithm iteratively
    solves trust-region subproblems augmented by special diagonal quadratic
    term with trust-region shape determined by the distance from the bounds
    and the direction of the gradient. This enhancements help not to take
    steps directly into bounds and explore the whole variable space. To
    improve convergence speed the reflected from the first bound search
    direction is considered. To obey theoretical requirements the algorithm is
    making strictly feasible iterates.

    Trust-region subproblems are solved by exact method very similar to one
    described in [2]_ and implemented in MINPACK, but with the help of one
    per iteration singular value decomposition of Jacobian. The algorithm's
    performance is generally comparable to scipy.optimize.leastsq in unbounded
    case.

    References
    ----------
    .. [1] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior, and
           Conjugate Gradient Method for Large-Scale Bound-Constrained
           Minimization Problems," SIAM Journal on Scientific Computing,
           Vol. 21, Number 1, pp 1–23, 1999.

    .. [2] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
           and Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes
           in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    """
    x0 = np.asarray(x0, dtype=float)
    l, u, feasible = check_bounds(x0, bounds)
    if not feasible:
        raise ValueError("`x0` is infeasible.")

    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol)

    # We need strictly feasible guess to start with
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

    J_augmented = np.empty((m + n, n))
    f_augmented = np.zeros((m + n))

    obj_value = np.dot(f, f)
    alpha = 0.0  # "Levenberg-Marquardt" parameter

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

        # Compute Coleman-Li scaling parameters and "hat" variables.
        d_CL, jv = CL_scaling(x, g, l, u)
        d = d_CL * scale
        g_h = d * g
        diag_h = g * jv * scale**2

        g_norm = norm(g * d_CL**2, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if termination_status is not None:
            return prepare_OptimizeResult(x, f, J, l, u, obj_value, g_norm,
                                          nfev, njac, nit, termination_status)

        # Jacobian in "hat" space.
        J_h = J * d

        # J_augmented is used to solve a trust-region subproblem with
        # diagonal term diag_h.
        J_augmented[:m] = J_h
        J_augmented[m:] = np.diag(diag_h**0.5)
        f_augmented[:m] = f

        U, s, V = svd(J_augmented, full_matrices=False)
        V = V.T
        uf = U.T.dot(f_augmented)

        # theta controls step back size from the bounds.
        theta = max(0.995, 1 - g_norm)
        actual_reduction = -1

        # In the following: p - trust-region solution, r - reflected solution,
        # c - minimizer along the scaled gradient, _h means the variable
        # is computed in "hat" space.
        while nfev < max_nfev and actual_reduction < 0:
            p_h, alpha, n_iter = solve_lsq_trust_region(
                n, m, uf, s, V, Delta, initial_alpha=alpha)
            p = d * p_h

            to_bound, _ = step_size_to_bound(x, p, l, u)
            if to_bound >= 1:  # Trust region step is feasible.
                # Still step back from the bounds
                p_h *= min(theta * to_bound, 1)
                steps_h = np.atleast_2d(p_h)
            else:  # Otherwise consider a reflected and gradient steps.
                p_h, r_h = find_reflected_step(x, J_h, diag_h, g_h, p, p_h,
                                               d, Delta, l, u, theta)
                c_h = find_gradient_step(x, J_h, diag_h, g_h,
                                         d, Delta, l, u, theta)
                steps_h = np.array([p_h, r_h, c_h])

            qp_values = evaluate_quadratic_function(J_h, diag_h, g_h, steps_h)
            min_index = np.argmin(qp_values)
            step_h = steps_h[min_index]

            # qp-values are negative, also need to double it
            predicted_reduction = -2 * qp_values[min_index]

            step = d * step_h
            x_new = make_strictly_feasible(x + step, l, u)

            f_new = fun(x_new)
            nfev += 1

            # Usual trust-region step quality estimation.
            obj_value_new = np.dot(f_new, f_new)
            actual_reduction = obj_value - obj_value_new
            # Correction term is specific to the algorithm,
            # vanishes in unbounded case.
            correction = np.dot(step_h * diag_h, step_h)

            if predicted_reduction > 0:
                ratio = (actual_reduction - correction) / predicted_reduction
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

        if actual_reduction > 0:
            x = x_new
            f = f_new
            obj_value = obj_value_new

            J = jac(x)
            njac += 1

    return prepare_OptimizeResult(x, f, J, l, u, obj_value, g_norm,
                                  nfev, njac, nit, 0)
