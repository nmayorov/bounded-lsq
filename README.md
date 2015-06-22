Bounded Nonlinear Least-Squares Optimization
--------------------------------------------

This repository contains algorithms I'm preparing for scipy as part of my GSoC 2015 work. In order to use: clone this repository and add the path to the folder to PYTHONPATH. Tested in Python 3.4 and 2.7 with numpy 1.9.2 and scipy 0.15.1.

----------------

The algorithms is contained in package `bounded_lsq`.

1. `leastsqbound.py` is a wrapper over `scipy.otpimize.leastsq` which does unbounded-to-bounded variables transformation. It is copied from [here](https://github.com/jjhelmus/leastsqbound-scipy) for convinience.
2. `dogbox.py` implements a dogleg trust-region algorithm applied to a rectangular trust region. You can read the description in my [blog](https://nmayorov.wordpress.com/2015/06/19/dogbox-algorithm/).
3. `trf.py` implements a special Trust Region Reflective algorithm which combines several ideas. See the description in my [blog](https://nmayorov.wordpress.com/2015/06/19/trust-region-reflective-algorithm/).

------------

Run benchmarks from `benchmarks/run_benchmarks.py`. For the default usage run 

```
python run_benchmarks.py
```

The following example command run only bounded problems with custom tolerance settings and writes the result into a file: 

```
python run_benchmarks.py report.txt -b -ftol 1e-8 -gtol 1e-5
```

For more information about this benchmarks read my [post](https://nmayorov.wordpress.com/2015/06/19/algorithm-benchmarks/). Note, that your results will be somewhat different to ones reported in the blog post, because I made small adjustments to the algorithms (and will keep making them).