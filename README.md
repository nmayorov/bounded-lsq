Bounded Nonlinear Least-Squares Optimization
--------------------------------------------

This repository contains algorithms I'm preparing for scipy as part of my GSoC 2015 work. In order to use: clone this repository and add the path to the folder to PYTHONPATH. Tested in Python 3.4 and 2.7 with numpy 1.9.2 and scipy 0.15.1.

----------------

The algorithms is contained in package `bounded_lsq`.

1. `leastsqbound.py` is a wrapper over `scipy.otpimize.leastsq` which does bounded-to-unbounded variables transformation. It is the exact copy from [here](https://github.com/jjhelmus/leastsqbound-scipy), master branch, taken on 27 June 2015, commit 937c67500cf9340c31d5cfbbb274894e3b59bd89. This is the [fixed](https://github.com/jjhelmus/leastsqbound-scipy/pull/7) version and now it works well. 
2. `dogbox.py` implements a dogleg trust-region algorithm applied to a rectangular trust region. You can read the description in my [blog](https://nmayorov.wordpress.com/2015/06/19/dogbox-algorithm/).
3. `trf.py` implements a special Trust Region Reflective algorithm which combines several ideas. See the description in my [blog](https://nmayorov.wordpress.com/2015/06/19/trust-region-reflective-algorithm/).

------------

Run benchmarks from `benchmarks/run_benchmarks.py`. The default usage 

```
python run_benchmarks.py
```

runs all benchmarks with default tolerance parameters, analytical Jacobian and prints to stdout.  

The following example command run only bounded problems with custom tolerance settings using 2-point numerical Jacobian approximation and prints the result into a file: 

```
python run_benchmarks.py report.txt -b -ftol 1e-12 -xtol 1e-12 -gtol 1e-8 -jac 2-point 
```

Run `python run_benchmarks.py --help` to see full parameters signature.

For more information about this benchmarks read my [post](https://nmayorov.wordpress.com/2015/06/19/algorithm-benchmarks/). Your results can be somewhat different to ones reported in the blog, because I keep adjusting algorithms.