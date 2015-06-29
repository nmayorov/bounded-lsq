from warnings import warn
import numpy as np
from numpy.linalg import norm
from scipy.optimize import approx_derivative, leastsq, OptimizeResult

from .bounds import in_bounds, prepare_bounds, find_active_constraints
from .trf import trf
from .dogbox import dogbox


EPS = np.finfo(float).eps


def check_tolerance(ftol, xtol, gtol):
    message = "{} is too low, setting to machine epsilon {}."
    if ftol < EPS:
        warn(message.format("`ftol`", EPS))
        ftol = EPS
    if xtol < EPS:
        warn(message.format("`xtol`", EPS))
        xtol = EPS
    if gtol < EPS:
        warn(message.format("`gtol`", EPS))
        gtol = EPS

    return ftol, xtol, gtol


TERMINATION_MESSAGES = {
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied."
}


FROM_MINPACK_TO_COMMON = {
    0: -1,  # 0 improper input parameters for MINPACK.
    1: 2,
    2: 3,
    3: 4,
    4: 1,
    5: 0
    # There are 6, 7, 8 for too small tolerance parameters,
    # but we guard against them by checking ftol, xtol, gtol beforehand.
}


def prepare_OptimizeResult(x, f, J, obj_value, g_norm, nfev, njev,
                           status, active_mask, x_covariance):
    r = OptimizeResult()
    r.x = x
    r.fun = f
    r.jac = J
    r.obj_value = obj_value
    r.optimality = g_norm
    r.active_mask = active_mask
    r.nfev = nfev
    r.njev = njev
    r.status = status
    r.success = status > 0
    r.message = TERMINATION_MESSAGES[status]
    r.x_covariance = x_covariance
    return r


def call_leastsq(fun, x0, jac, ftol, xtol, gtol, max_nfev, scaling,
                 diff_step, args, options):
    if jac == '3-point':
        warn("jac='3-point' works equivalently to '2-point' "
             "for 'lm' method.")

    if jac in ['2-point', '3-point']:
        jac = None

    if max_nfev is None:
        max_nfev = 0

    if diff_step is None:
        epsfcn = None
    else:
        epsfcn = diff_step**2

    if scaling == 'jac':
        scaling = None
    elif scaling is not None:
        scaling = np.asarray(scaling)
        if scaling.ndim == 0:
            scaling = np.resize(scaling, x0.shape)

    return leastsq(fun, x0, args=args, Dfun=jac, full_output=True,
                   ftol=ftol, xtol=xtol, gtol=gtol, maxfev=max_nfev,
                   epsfcn=epsfcn, diag=scaling, **options)


def least_squares(fun, x0, jac='2-point', bounds=(-np.inf, np.inf),
                  method='trf', ftol=EPS**0.5, xtol=EPS**0.5, gtol=EPS**0.5,
                  max_nfev=None, scaling=1.0, diff_step=None,
                  args=(), kwargs={}, options={}):
    """Minimize the sum of squares of nonlinear functions with bounds on
    independent variables.

    Let f(x) maps from R^n to R^m, the function finds a local minimum of
    F(x) = ||f(x)||**2 = sum(f_i(x)**2, i = 1, ..., m),
    subject to bound constraints l <= x <= u

    Parameters
    ----------
    fun : callable
        Returns a 1d-array of residuals of size m.
    x0 : array-like, shape (n,)
        Initial guess on the independent variables.
    jac : '2-point', '3-point' or callable, optional
        Method of computing partial derivatives of f with respect to x,
        which form m-by-n array called the Jacobian matrix. If set to '2-point'
        or '3-point', the Jacobian matrix is estimated by the corresponding
        finite difference scheme. The '3-point' scheme is more accurate,
        but requires twice as much operations compared to '2-point' (default).
        If callable then it should return a reasonable approximation of the
        Jacobian as ndarray. The typical use case is to implement Jacobian
        computation function by exact formulas (but make sure your
        implementation is correct running optimization.)
    bounds : tuple of array-like, optional
        Lower and upper bounds on independent variables. Default is not
        Each bound must match the size of `x0` or be a scalar, in the latter
        case the bound will be the same for all variables. Use ``np.inf``
        with an appropriate sign to disable bounds to some of the variables.
    method : {'trf', 'dogbox', 'lm'}, optional
        Determines the algorithm to perform optimization. Default is 'trf.
        See Notes and algorithm options to get information about each
        algorithm.
    ftol : float, optional
        Tolerance for termination by the change of the objective value.
        Default is square root of machine epsilon. See the exact meaning in
        documentation for a particular method.
    xtol : float. optional
        Tolerance for termination by the change of the independent variables.
        Default is square root of machine epsilon. See the exact meaning in
        documentation for a particular method.
    gtol : float, optional
        Tolerance for termination by the norm of gradient. Default is
        square root of machine epsilon. In presence of bounds more correctly
        to call it first-order optimality threshold, as raw gradient
        itself doesn't measure optimality. See the exact meaning in
        documentation for a particular method.
    max_nfev : None or int, optional
        Maximum number of function evaluations before the termination.
        If None (default) each algorithm uses its own default value.
    scaling : array-like or 'jac', optional
        Applies scaling to potentially improve algorithm convergence.
        Default is 1.0 which means no scaling. Scaling should be used to
        equalize the influence of each variable on the objective function.
        Alternatively you can think of `scaling` as diagonal elements of
        a matrix which determines the shape of a trust-region.
        Use smaller values for variables which have bigger characteristic
        scale compared to others. A scalar value won't affect the algorithm
        (except maybe fixing/introducing numerical issues and changing
        termination criteria). If 'jac', then scaling is proportional to the
        norms of Jacobian columns. The latter option is often helpful in
        unconstrained problems, but not so much in constrained ones.
        From experience usage of 'jac'-scaling is not recommended for bounded
        problems with 'trf' method.
    diff_step : None or array-like, optional
        Determines the step size for finite difference Jacobian approximation.
        The actual step is computed as ``x * diff_step``. If None (default),
        `diff_step` is assigned to a conventional "optimal" power of machine
        epsilon depending on finite difference approximation method [Press].
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun` and `jac`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)``. When
        ``method='lm'`` then `kwargs` is ignored.
    options : dict, optional
        Additional options passed to a chosen algorithm. Empty by default.
        The calling sequence is ``method(..., **options)``. Look for relative
        options in documentation for a particular method.

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
        First-order optimality measure. This quantity was compared with
        `gtol` during iterations.
    active_mask : ndarray of int, shape (n,)
        Each component shows whether the corresponding constraint is active:
             0 - a constraint is not active.
            -1 - a lower bound is active.
             1 - an upper bound is active.
        Might be somewhat arbitrary as the algorithm does strictly feasible
        iterates, thus `active_mask` is determined with tolerance threshold.
    nfev : int
        Number of function evaluations done. Methods 'trf' and 'dogbox' don't
        count function calls for numerical Jacobian approximation, opposed to
        'lm' method.
    njev : int
        Number of Jacobian evaluations done. If numerical Jacobian
        approximation is used in 'lm' method it is set to None.
    status : int
        Reason for algorithm termination:
            -1 - improper input parameters status returned from `leastsq`.
             0 - the maximum number of function evaluations is exceeded.
             1 - `gtol` termination condition is satisfied.
             2 - `ftol` termination condition is satisfied.
             3 - `xtol` convergence test is satisfied.
             4 - Both `ftol` and `xtol` termination conditions are satisfied.
    message : string
        Verbal description of the termination reason.
    success : int
        True if one of the convergence criteria is satisfied.
    x_covariance : ndarray, shape (m, m)
        Estimate of `x` covariance assuming that residuals are uncorrelated
        and have unity variance.

    Notes
    -----
    Method 'lm' (Levenberg-Marquardt) calls a wrapper over least-squares
    algorithms implemented in MINPACK (lmder, lmdif). It runs
    Levenberg-Marquadrd algorithm formulated as a trust-region type algorithm.
    The implementation is based on paper [JJMore], it is very robust and
    efficient with a lot of smart tricks. It should be your first choice
    for unconstrained problems. Note that it doesn't support bounds.

    Method 'trf' (Trust Region Reflective) is motivated by the process of
    solving the equation, which constitutes the first-order optimality
    condition for a bound-constrained minimization problem as formulated in
    [STIR]_. The algorithm iteratively solves trust-region subproblems
    augmented by special diagonal quadratic term with trust-region shape determined by
    the distance from the bounds and the direction of the gradient.
    This enhancements help not to take steps directly into bounds and explore
    the whole variable space. To improve convergence speed the reflected from
    the first bound search direction is considered. To obey theoretical
    requirements the algorithm keeps iterates strictly feasible. Trust-region
    subproblems are solved by exact method very similar to one described in
    [JJMore] and implemented in MINPACK, but with the help of one
    per iteration singular value decomposition of Jacobian. The algorithm's
    performance is generally comparable to MINPACK in unbounded case. The
    algorithm works quite robust in unbounded and bounded problems, thus
    it is set as default algorithm.

    Method 'dogbox' operates in a trust-region framework, but considers
    rectangular trust regions as opposed to conventional elliptical.
    The intersection of the current trust region and initial bounds is again
    rectangular, so on each iteration a quadratic minimization problem subject
    to bounds is solved. Powell's dogleg method [NumOpt]_ is applied to solve
    these subproblems. The algorithm is likely to exhibit slow convergence
    when the rank of Jacobian is less than the number of variables. For some
    problems though it performs better than 'trf' and 'lm', so you might want
    to try it. Although it's hard to say generally which problems are suitable
    for this method.

    References
    ----------
    [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
             and Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes
             in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior, and
           Conjugate Gradient Method for Large-Scale Bound-Constrained
           Minimization Problems," SIAM Journal on Scientific Computing,
           Vol. 21, Number 1, pp 1–23, 1999.
    [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region Dogleg
             Approach for Unconstrained and Bound Constrained Nonlinear
             Optimization", WSEAS International Conference on Applied
             Mathematics, Corfu, Greece, 2004.
    [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,
             2nd edition", Chapter 4.
    """

    if method not in ['trf', 'dogbox', 'lm']:
        raise ValueError("`method` must be 'trf', 'dogbox' or 'lm'.")

    if len(bounds) != 2:
        raise ValueError("`bounds` must contain 2 elements.")

    x0 = np.asarray(x0, dtype=float)

    if x0.ndim > 1:
        raise ValueError("`x0` must be at most 1-dimensional.")

    l, u = prepare_bounds(bounds, x0.size)

    if l.shape != x0.shape or u.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    bounded = not np.all((l == -np.inf) & (u == np.inf))

    if method == 'lm' and bounded:
        raise ValueError("Method 'lm' doesn't support bounds.")

    if jac not in ['2-point', '3-point'] and not callable(jac):
        raise ValueError("`jac` must be '2-point', '3-point' or callable.")

    if scaling != 'jac':
        scaling = np.asarray(scaling)
        if np.any(scaling <= 0):
            raise ValueError("`scaling` must be None, 'jac', "
                             "or array-like with positive elements.")

    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol)

    # Handle 'lm' separately
    if method == 'lm':
        if len(kwargs) > 0:
            warn("Method 'lm' can't use `kwargs`.")

        x, cov_x, info, message, status = call_leastsq(
            fun, x0, jac, ftol, xtol, gtol, max_nfev,
            scaling, diff_step, args, options)

        f = info['fvec']
        if callable(jac):
            J = jac(x, *args)
        else:
            J = approx_derivative(fun, x, args=args)
        obj_value = np.dot(f, f)

        # According to MINPACK compute optimality as
        # cosine between f and J columns.
        if obj_value == 0:
            g_norm = 0
        else:
            g = J.T.dot(f)
            g /= norm(f)
            J_norm = norm(J, axis=0)
            mask = J_norm > 0
            g[mask] /= J_norm[mask]
            g_norm = norm(g, ord=np.inf)

        nfev = info['nfev']
        njev = info.get('njev', None)
        status = FROM_MINPACK_TO_COMMON[status]
        active_mask = np.zeros_like(x0, dtype=int)

        return prepare_OptimizeResult(x, f, J, obj_value, g_norm,
                                      nfev, njev, status, active_mask, cov_x)

    if not in_bounds(x0, l, u):
        raise ValueError("`x0` is infeasible.")

    if max_nfev is None:
        max_nfev = x0.shape[0] * 100

    fun_wrapped = lambda x: fun(x, *args, **kwargs)

    if jac in ['2-point', '3-point']:
        # args, kwargs already passed to function
        jac_wrapped = lambda x, f: approx_derivative(
            fun, x, method=jac, f0=f, bounds=bounds)
    else:
        jac_wrapped = lambda x, f: jac(x, *args, **kwargs)

    if method == 'trf':
        x, f, J, obj_value, g_norm, nfev, njev, status = trf(
            fun_wrapped, jac_wrapped, x0, l, u,
            ftol, xtol, gtol, max_nfev, scaling)
        active_mask = find_active_constraints(x, l, u, rtol=xtol)

    elif method == 'dogbox':
        x, f, J, obj_value, g_norm, nfev, njev, status, active_mask = \
            dogbox(fun_wrapped, jac_wrapped, x0, l, u,
                   ftol, xtol, gtol, max_nfev, scaling)

    return prepare_OptimizeResult(x, f, J, obj_value, g_norm, nfev, njev,
                                  status, active_mask, None)
