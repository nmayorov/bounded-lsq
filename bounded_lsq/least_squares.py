from warnings import warn
import numpy as np

from scipy.optimize import approx_derivative, OptimizeResult

from .bounds import check_bounds, find_active_constraints
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
    1: "First order optimality measure is less than `gtol`.",
    2: "The relative reduction of objective value is less than `ftol`.",
    3: "The relative step size is less than `xtol`."
}


def prepare_OptimizeResult(x, f, J, obj_value, g_norm, nfev, njev, nit,
                           status, active_mask):
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
    return r


def jacobian_wrapper(fun, jac, bounds):
    if jac is None:
        def wrapped_jac(x, f):
            return approx_derivative(fun, x, method='2-point', f0=f,
                                     bounds=bounds)
    else:
        def wrapped_jac(x, f):
            return jac(x)

    return wrapped_jac


def least_squares(fun, x0, bounds=(None, None), method='trf', jac=None,
                  ftol=EPS**0.5, xtol=EPS**0.5, gtol=EPS**0.5,
                  max_nfev=None, scaling=1.0):
    if method not in ['trf', 'dogbox']:
        raise ValueError("`method` must be 'trf' or 'dogbox'.")

    x0 = np.asarray(x0, dtype=float)
    l, u, feasible = check_bounds(x0, bounds)
    if not feasible:
        raise ValueError("`x0` is infeasible.")

    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol)

    if max_nfev is None:
        max_nfev = x0.shape[0] * 100

    jac = jacobian_wrapper(fun, jac, bounds)

    if method == 'trf':
        x, f, J, obj_value, g_norm, nfev, njev, nit, status = trf(
            fun, jac, x0, l, u, ftol, xtol, gtol, max_nfev, scaling)
        active_mask = find_active_constraints(x, l, u, rtol=xtol)

    elif method == 'dogbox':
        x, f, J, obj_value, g_norm, nfev, njev, nit, status, active_mask = \
            dogbox(fun, jac, x0, l, u, ftol, xtol, gtol, max_nfev, scaling)

    return prepare_OptimizeResult(x, f, J, obj_value, g_norm, nfev, njev, nit,
                                  status, active_mask)
