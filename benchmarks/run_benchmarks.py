from __future__ import division, print_function

import argparse
from collections import OrderedDict
import sys

import numpy as np
from scipy.optimize import leastsq, fmin_l_bfgs_b
from bounded_lsq import trf, dogbox, leastsqbound
from lsq_problems import extract_lsq_problems


def run_dogbox(problem, ftol=1e-5, xtol=1e-5, gtol=1e-3, **kwargs):
    x, obj_value, nfev = dogbox(
        problem.fun, problem.jac, problem.x0,
        bounds=problem.bounds, ftol=ftol, gtol=gtol, xtol=xtol, **kwargs)
    return x, obj_value, nfev


def run_trf(problem, ftol=1e-5, xtol=1e-5, gtol=1e-3, **kwargs):
    x, obj_value, nfev = trf(
        problem.fun, problem.jac, problem.x0,
        bounds=problem.bounds, ftol=ftol, gtol=gtol, xtol=xtol, **kwargs)
    return x, obj_value, nfev


def scipy_bounds(problem):
    n = problem.x0.shape[0]
    lb, ub = problem.bounds
    if lb is None:
        lb = [None] * n
    if ub is None:
        ub = [None] * n
    bounds = []
    for l, u in zip(lb, ub):
        if l == -np.inf:
            l = None
        if u == np.inf:
            u = None
        bounds.append((l, u))
    return bounds


def run_leastsq_bound(problem, ftol=1e-5, xtol=1e-5, gtol=1e-3,
                      scaling=None, **kwargs):
    bounds = scipy_bounds(problem)

    if scaling is None:
        diag = np.ones_like(problem.x0)
    else:
        diag = None

    x, cov_x, info, _, _ = leastsqbound(
        problem.fun, problem.x0, bounds=bounds, full_output=True,
        Dfun=problem.jac, ftol=ftol, gtol=gtol, diag=diag, **kwargs
    )
    f = problem.fun(x)
    return x, np.dot(f, f), info['nfev']


def run_l_bfgs_b(problem, ftol=1e-5, gtol=1e-3, xtol=None):
    bounds = scipy_bounds(problem)
    factr = ftol / np.finfo(float).eps
    x, obj_value, info = fmin_l_bfgs_b(
        problem.obj_value, problem.x0, fprime=problem.grad, bounds=bounds,
        m=100, factr=factr, pgtol=gtol, iprint=-1)
    return x, obj_value, info['funcalls']


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
    header = "{:<25} {:<5} {:<5} {:<15} {:<5} {:<10} {:<10} {:<5}".\
        format("problem", "n", "m", "solver", "nfev", "g norm",
               "value", "active")

    if benchmark_name is not None:
        print(benchmark_name.center(len(header)))

    print(header)
    print("-" * len(header))

    report_format = "{:<25} {:<5} {:<5} {:<15} {:<5} {:<10.2e} {:<10.2e} {:<5}"

    if methods is None:
        methods = METHODS.keys()

    for problem_name, problem in problems:
        results = []
        used_methods = []
        for method_name in methods:
            used_methods.append(method_name)
            method, kwargs = METHODS[method_name]
            result = method(problem, ftol=ftol, xtol=xtol, gtol=gtol, **kwargs)
            results.append(result)

        for i, (method_name, result) in enumerate(zip(used_methods, results)):
            x, obj_value, nfev = result
            opt, active = problem.check_solution(x)
            if "_B" in problem_name and method_name == 'leastsq':
                method_name += "bound"
            if i == 0:
                print(report_format.format(
                    problem_name, problem.n, problem.m, method_name,
                    nfev, opt, obj_value, active))
            else:
                print(report_format.format(
                    "", "", "", method_name, nfev, opt, obj_value, active))
        print()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs='?', type=str, help="Output file.")
    parser.add_argument("-u", action='store_true', help="Benchmark unbounded")
    parser.add_argument("-b", action='store_true', help="Benchmark bounded.")
    parser.add_argument("-ftol", type=float, default=1e-10)
    parser.add_argument("-xtol", type=float, default=0)
    parser.add_argument("-gtol", type=float, default=0)
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
        run_benchmark(u, ftol=args.ftol, xtol=args.ftol, gtol=args.gtol,
                      benchmark_name="Unbounded problems")
    if args.b:
        run_benchmark(b, ftol=args.ftol, xtol=args.ftol, gtol=args.gtol,
                      methods=["dogbox", "trf", "leastsq", "l-bfgs-b"],
                      benchmark_name="Bounded problems")


if __name__ == '__main__':
    main()
