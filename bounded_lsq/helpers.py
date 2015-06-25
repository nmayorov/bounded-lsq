"""Small infrastructure functions."""

from warnings import warn
import numpy as np
from scipy.optimize import OptimizeResult
from .bounds import find_active_constraints

EPS = np.finfo(float).eps


def check_tolerance(ftol, xtol, gtol):
    message = "{} is too low, setting to machine epsilon {}"
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


_TERMINATION_MESSAGES = {
    0: "The maximum number of function evaluations is exceeded.",
    1: "First order optimality measure is less than `gtol`.",
    2: "The relative reduction of objective value is less than `ftol`.",
    3: "The relative step size is less than `xtol`."
}


def prepare_OptimizeResult(x, f, J, l, u, obj_value, g_norm,
                           nfev, njac, nit, status, active_mask=None):
    r = OptimizeResult()
    r.x = x
    r.fun = f
    r.jac = J
    r.obj_value = obj_value
    r.optimality = g_norm
    if active_mask is None:
        r.active_mask = find_active_constraints(x, l, u)
    else:
        r.active_mask = active_mask
    r.nfev = nfev
    r.njac = njac
    r.nit = nit
    r.status = status
    r.success = status > 0
    r.message = _TERMINATION_MESSAGES[status]
    return r
