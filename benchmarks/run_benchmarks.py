from __future__ import division, print_function

import argparse
from collections import OrderedDict
import sys

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from bounded_lsq import (trf, dogbox, leastsqbound, CL_optimality,
                         find_active_constraints, make_strictly_feasible)
from lsq_problems import extract_lsq_problems


def run_dogbox(problem, ftol=1e-5, xtol=1e-5, gtol=1e-3, **kwargs):
    result = dogbox(problem.fun, problem.jac, problem.x0,
                    bounds=problem.bounds, ftol=ftol, gtol=gtol,
                    xtol=xtol, **kwargs)
    return (result.nfev, result.optimality, result.obj_value,
            np.sum(result.active_mask != 0), result.status)


def run_trf(problem, ftol=1e-5, xtol=1e-5, gtol=1e-3, **kwargs):
    result = trf(problem.fun, problem.jac, problem.x0,
                 bounds=problem.bounds, ftol=ftol, gtol=gtol,
                 xtol=xtol, **kwargs)
    return (result.nfev, result.optimality, result.obj_value,
            np.sum(result.active_mask != 0), result.status)


def scipy_bounds(problem):
    n = problem.x0.shape[0]
    l, u = problem.bounds
    if l is None:
        l = np.full(n, -np.inf)
    else:
        l = np.asarray(l)
    if u is None:
        u = np.full(n, np.inf)
    else:
        u = np.asarray(u)

    bounds = []
    for li, ui in zip(l, u):
        if li == -np.inf:
            li = None
        if ui == np.inf:
            ui = None
        bounds.append((li, ui))
    return bounds, l, u


def run_leastsq_bound(problem, ftol=1e-5, xtol=1e-5, gtol=1e-3,
                      scaling=None, **kwargs):
    bounds, l, u = scipy_bounds(problem)

    if scaling is None:
        diag = np.ones_like(problem.x0)
    else:
        diag = None

    x, cov_x, info, mesg, ier = leastsqbound(
        problem.fun, problem.x0, bounds=bounds, full_output=True,
        Dfun=problem.jac, ftol=ftol, xtol=xtol, gtol=gtol, diag=diag, **kwargs
    )
    x = make_strictly_feasible(x, l, u)
    f = problem.fun(x)
    g = 0.5 * problem.grad(x)
    optimality = CL_optimality(x, g, l, u)
    active = find_active_constraints(x, l, u)
    return info['nfev'], optimality, np.dot(f, f), np.sum(active != 0), ier


def run_l_bfgs_b(problem, ftol=1e-5, gtol=1e-3, xtol=None):
    bounds, l, u = scipy_bounds(problem)
    factr = ftol / np.finfo(float).eps
    x, obj_value, info = fmin_l_bfgs_b(
        problem.obj_value, problem.x0, fprime=problem.grad, bounds=bounds,
        m=100, factr=factr, pgtol=gtol, iprint=-1)
    g = 0.5 * problem.grad(x)
    optimality = CL_optimality(x, g, l, u)
    active = find_active_constraints(x, l, u)
    return (info['funcalls'], optimality, obj_value, np.sum(active != 0),
            info['warnflag'])


METHODS = OrderedDict([
    ("dogbox", (run_dogbox, dict())),
    ("dogbox-s", (run_dogbox, dict(scaling='auto'))),
    ("trf", (run_trf, dict())),
    ("trf-s", (run_trf, dict(scaling='auto'))),
    ("leastsq", (run_leastsq_bound, dict())),
    ("leastsq-s", (run_leastsq_bound, dict(scaling='auto'))),
    ("l-bfgs-b", (run_l_bfgs_b, dict())),
    ]
)


def run_benchmark(problems, ftol=1e-5, xtol=1e-5, gtol=1e-3,
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
            result = method(problem, ftol=ftol, xtol=xtol, gtol=gtol, **kwargs)
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
        run_benchmark(u, ftol=args.ftol, xtol=args.xtol, gtol=args.gtol,
                      methods=methods, benchmark_name="Unbounded problems")
    if args.b:
        methods = ['dogbox', 'trf', 'leastsq', 'l-bfgs-b']
        run_benchmark(b, ftol=args.ftol, xtol=args.xtol, gtol=args.gtol,
                      methods=methods,  benchmark_name="Bounded problems")


if __name__ == '__main__':
    main()
