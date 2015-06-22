"""Functions related to a trust-region problem."""


from __future__ import division

import numpy as np
from numpy.linalg import norm
from math import copysign


def get_intersection(x, d, Delta):
    """Find the intersection of a line with a spherical trust region.

    The function just solves the quadratic equation with respect to t
    ||(x + t d)||**2 = Delta**2.

    Returns
    -------
    t_neg, t_pos : tuple of float
        The negative and positive roots.

    Raises
    ------
    ValueError
        If 'x' is not within the trust region.
    """
    a = np.dot(d, d)
    b = np.dot(x, d)
    c = np.dot(x, x) - Delta**2
    if c > 0:
        raise ValueError("`x` is not within the trust region.")
    delta = np.sqrt(b*b - a*c)
    q = -(b + copysign(delta, b))
    t1 = q / a
    t2 = c / q
    if t1 < t2:
        return t1, t2
    else:
        return t2, t1


def _phi_and_derivative(alpha, suf, s, Delta):
    """Used by solve_lsq_trust_region."""
    denom = s**2 + alpha
    p_norm = norm(suf / denom)
    phi = p_norm - Delta
    phi_prime = -np.sum(suf ** 2 / denom**3) / p_norm
    return phi, phi_prime


def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None,
                           rtol=0.01, max_iter=10):
    """Solve a trust-region problem arising in least-squares minimization by
    MINPACK approach.

    This function implements a method described by J. J. More, but it relies
    on SVD of Jacobian.

    Before running this function we compute
    ``U, s, VT = svd(J, full_matrices=False)``.  The vector of residuals we
    denote as f.

    Parameters
    ----------
    n : int
        Number of variables.
    m : int
        Number of residuals.
    uf : array
        Should be computed as U.T.dot(f).

    s : array
        Singular values of J.

    V : array
        Transpose of VT.

    Delta : float
        A trust-region radius.

    undetermined : bool
        Tells that we solve undetermined system. Unfortunately cannot be
        determined from other parameters, so pass ``m < n`` as this parameter.

    initial_alpha : float, optional
        The initial guess for alpha, which might be available from a
        previous iteration.

    rtol : float, optional
        The stopping tolerance for the root-finding procedure. Namely,
        the root `alpha` must satisfy ``|p(alpha) - Delta| < rtol * Delta``.

    max_iter : int, optional
        The maximum allowed number of iterations for the root-finding
        procedure.

    Returns
    -------
    p : array, shape (n,)
        The solution of a trust region problem

    alpha : float
        The corresponding scalar parameter (sometimes called
        Levenberg-Marquardt parameter).

    n_iter : int
        The number of iterations made by root-finding procedure. Zero means
        that Gauss-Newton step was selected as the solution.
    """
    suf = s * uf

    # Check if J has full rank and try Gauss-Newton step.
    if m >= n:
        threshold = np.finfo(float).eps * m * s[0]
        full_rank = s[-1] > threshold
    else:
        full_rank = False

    if full_rank:
        p = -V.dot(uf / s)
        if norm(p) <= Delta:
            return p, 0.0, 0

    alpha_upper = norm(suf) / Delta

    if full_rank:
        phi, phi_prime = _phi_and_derivative(0.0, suf, s, Delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0

    if initial_alpha is None or not full_rank and initial_alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)
    else:
        alpha = initial_alpha

    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)

        phi, phi_prime = _phi_and_derivative(alpha, suf, s, Delta)

        if np.abs(phi) < rtol * Delta:
            break

        if phi < 0:
            alpha_upper = alpha

        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta

    p = -V.dot(suf / (s**2 + alpha))
    if phi > 0:
        p *= Delta / norm(p)

    return p, alpha, it + 1
