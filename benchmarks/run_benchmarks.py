from __future__ import division, print_function

import argparse
from collections import OrderedDict
import sys

import numpy as np
from scipy.optimize import fmin_l_bfgs_b, leastsq
from bounded_lsq import (least_squares, leastsqbound, CL_optimality,
                         find_active_constraints, make_strictly_feasible)
from lsq_problems import extract_lsq_problems


def run_least_squares(problem, ftol, xtol, gtol, jac, **kwargs):
    l, u = problem.bounds
    if l is None:
        l = -np.inf
    if u is None:
        u = np.inf
    bounds = l, u
    if jac == 'exact':
        jac = problem.jac

    result = least_squares(problem.fun, problem.x0, jac=jac, bounds=bounds,
                           ftol=ftol, gtol=gtol, xtol=xtol, **kwargs)
    x = result.x
    g = 0.5 * problem.grad(x)
    optimality = CL_optimality(result.x, g, l, u)

    return (result.nfev, optimality, result.obj_value,
            np.sum(result.active_mask != 0), result.status)


def scipy_bounds(problem):
    n = problem.x0.shape[0]
    lb, ub = problem.bounds
    if lb is None:
        lb = np.full(n, -np.inf)
    else:
        lb = np.asarray(lb)
    if ub is None:
        ub = np.full(n, np.inf)
    else:
        ub = np.asarray(ub)

    bounds = []
    for li, ui in zip(lb, ub):
        if li == -np.inf:
            li = None
        if ui == np.inf:
            ui = None
        bounds.append((li, ui))
    return bounds, lb, ub


def run_leastsq_bound(problem, ftol, xtol, gtol, jac,
                      scaling=None, **kwargs):
    bounds, lb, ub = scipy_bounds(problem)

    if scaling is None:
        diag = np.ones_like(problem.x0)
    else:
        diag = None

    if jac in ['2-point', '3-point']:
        jac = None
    else:
        jac = problem.jac

    x, cov_x, info, mesg, ier = leastsqbound(
        problem.fun, problem.x0, bounds=bounds, full_output=True,
        Dfun=jac, ftol=ftol, xtol=xtol, gtol=gtol, diag=diag, **kwargs
    )
    x = make_strictly_feasible(x, lb, ub)
    f = problem.fun(x)
    g = 0.5 * problem.grad(x)
    optimality = CL_optimality(x, g, lb, ub)
    active = find_active_constraints(x, lb, ub)
    return info['nfev'], optimality, np.dot(f, f), np.sum(active != 0), ier


def run_l_bfgs_b(problem, ftol, gtol, xtol, jac):
    bounds, l, u = scipy_bounds(problem)
    factr = ftol / np.finfo(float).eps
    if jac in ['2-point', '3-point']:
        grad = None
        approx_grad = True
    else:
        grad = problem.grad
        approx_grad = False
    x, obj_value, info = fmin_l_bfgs_b(
        problem.obj_value, problem.x0, fprime=grad, bounds=bounds,
        approx_grad=approx_grad, m=100, factr=factr, pgtol=gtol, iprint=-1)
    g = 0.5 * problem.grad(x)
    optimality = CL_optimality(x, g, l, u)
    active = find_active_constraints(x, l, u)
    return (info['funcalls'], optimality, obj_value, np.sum(active != 0),
            info['warnflag'])


METHODS = OrderedDict([
    ("dogbox", (run_least_squares, dict(method='dogbox', scaling=1.0))),
    ("dogbox-s", (run_least_squares, dict(method='dogbox', scaling='jac'))),
    ("trf", (run_least_squares, dict(method='trf', scaling=1.0))),
    ("trf-s", (run_least_squares, dict(method='trf', scaling='jac'))),
    ("lm", (run_least_squares, dict(method='lm', scaling=1.0))),
    ("lm-s", (run_least_squares, dict(method='lm', scaling='jac'))),
    ('leastsqbound', (run_leastsq_bound, dict(scaling=None))),
    ("l-bfgs-b", (run_l_bfgs_b, dict())),
    ]
)


def run_benchmark(problems, ftol, xtol, gtol, jac,
                  methods=None, benchmark_name=None):
    header = "{:<25} {:<5} {:<5} {:<15} {:<5} {:<10} {:<10} {:<8} {:<8}".\
        format("problem", "n", "m", "solver", "nfev", "g norm",
               "value", "active", "status")

    if benchmark_name is not None:
        print(benchmark_name.center(len(header)))

    print(header)
    print("-" * len(header))

    report_format = "{:<25} {:<5} {:<5} {:<15} {:<5} {:<10.2e} {:<10.2e} " \
                    "{:<8} {:<8}"

    if methods is None:
        methods = METHODS.keys()

    for problem_name, problem in problems:
        if problem_name == 'ThermistorResistance':
            pass
        results = []
        used_methods = []
        for method_name in methods:
            used_methods.append(method_name)
            method, kwargs = METHODS[method_name]
            result = method(problem, ftol, xtol, gtol, jac, **kwargs)
            results.append(result)

        for i, (method_name, result) in enumerate(zip(used_methods, results)):
            nfev, opt, obj_value, active, status = result
            if "_B" in problem_name and method_name == 'leastsq':
                method_name += "bound"
            if i == 0:
                print(report_format.format(
                    problem_name, problem.n, problem.m, method_name,
                    nfev, opt, obj_value, active, status))
            else:
                print(report_format.format(
                    "", "", "", method_name, nfev, opt, obj_value, active,
                    status))
        print()


def parse_arguments():
    tol = np.finfo(float).eps**0.5
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs='?', type=str, help="Output file.")
    parser.add_argument("-jac", choices=['exact', '2-point', '3-point'],
                        default='exact', help="How to compute Jacobian.")
    parser.add_argument("-u", action='store_true', help="Benchmark unbounded")
    parser.add_argument("-b", action='store_true', help="Benchmark bounded.")
    parser.add_argument("-ftol", type=float, default=tol)
    parser.add_argument("-xtol", type=float, default=tol)
    parser.add_argument("-gtol", type=float, default=tol)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.output is not None:
        sys.stdout = open(args.output, "w")

    u, b = extract_lsq_problems()
    if not args.u and not args.b:
        args.u = True
        args.b = True
    if args.u:
        methods = None
        run_benchmark(u, args.ftol, args.xtol, args.gtol, args.jac,
                      methods=methods, benchmark_name="Unbounded problems")
    if args.b:
        methods = ['dogbox', 'trf', 'leastsqbound', 'l-bfgs-b']
        run_benchmark(b, args.ftol, args.xtol, args.gtol, args.jac,
                      methods=methods,  benchmark_name="Bounded problems")


if __name__ == '__main__':
    main()
