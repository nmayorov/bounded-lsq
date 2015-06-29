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


def prepare_OptimizeResult(x, f, J, obj_value, g_norm, nfev, njev, nit,
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
    r.nit = nit
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
                  max_nfev=None, scaling=None, diff_step=None, args=(),
                  kwargs={}, options={}):
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

    if scaling is not None:
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
        njev = info['njev']
        nit = None
        status = FROM_MINPACK_TO_COMMON[status]
        active_mask = np.zeros_like(x0, dtype=int)

        return prepare_OptimizeResult(x, f, J, obj_value, g_norm, nfev, njev,
                                      nit, status, active_mask, cov_x)

    if not in_bounds(x0, l, u):
        raise ValueError("`x0` is infeasible.")

    if max_nfev is None:
        max_nfev = x0.shape[0] * 100

    fun_wrapped = lambda x: fun(x, *args, **kwargs)

    if jac in ['2-point', '3-point']:
        # args, kwargs already passed to function
        jac_wrapped = lambda x, f: approx_derivative(fun, x, method=jac, f0=f,
                                                     bounds=bounds)
    else:
        jac_wrapped = lambda x, f: jac(x, *args, **kwargs)

    if scaling is None:
        if bounded:
            scaling = 1.0
        else:
            scaling = 'jac'

    if method == 'trf':
        x, f, J, obj_value, g_norm, nfev, njev, nit, status = trf(
            fun_wrapped, jac_wrapped, x0, l, u,
            ftol, xtol, gtol, max_nfev, scaling)
        active_mask = find_active_constraints(x, l, u, rtol=xtol)

    elif method == 'dogbox':
        x, f, J, obj_value, g_norm, nfev, njev, nit, status, active_mask = \
            dogbox(fun_wrapped, jac_wrapped, x0, l, u,
                   ftol, xtol, gtol, max_nfev, scaling)

    return prepare_OptimizeResult(x, f, J, obj_value, g_norm, nfev, njev, nit,
                                  status, active_mask, None)
