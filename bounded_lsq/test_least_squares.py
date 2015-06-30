import numpy as np
from numpy.testing import run_module_suite, assert_, assert_allclose, TestCase

from least_squares import least_squares

def f11(x, a=0.):
    return (x - a)**2 + 5.

def jac11(x, a=0.):
    return 2. * (x - a)

#
# Parameterize basic smoke tests across methods
#
class BaseMixin(object):
    def test_basic(self):
        # test that the basic calling sequence works
        res = least_squares(f11, 2., method=self.meth)
        assert_allclose(res.x, 0.)
        assert_allclose(res.fun, f11(res.x))

    def test_fun_args(self):
        # test that f(x, *args) works
        res = least_squares(f11, 2., args=(3,), method=self.meth)
        assert_allclose(res.x, 3.)
        assert_allclose(res.fun, f11(res.x))
        
        # also **kwds work
        res = least_squares(f11, 2., kwds={'a': 3,}, method=self.meth)
        assert_allclose(res.x, 3.)
        assert_allclose(res.fun, f11(res.x))

    def test_fun_wrong_args(self):
        # inconsistent *args raise
        assert_raises(TypeError, least_squares, f11, 2.,
                                 **dict(args=(3, 4,), method=self.meth))
        assert_raises(TypeError, least_squares, f11, 2.,
                                 **dict(kwds={'kaboom': 3}, method=self.meth))

    ### TODO: repeat with jacobian specified

    def test_jac_kw(self):
        # known values for the jac keyword are whitelisted
        for jac in ['2-point', '3-point']:
            least_squares(f11, 2., jac=jac, method=self.meth)

        assert_raises(TypeError, least_squares, f11, 2., **{'jac': 'oops'})

    ### TODO: test that unknown options raise a TypeError
    ###       test that xtol, ftol, gtol are accepted
    ###       ditto for max_nfev
    ###       whitelist the values of scaling
    ###       check that diff_step is accepted (and maybe is checked to be positive)

    ### TODO: add a less trivial function, check that it works (just a smoke test).
    
    ### TODO: what do we do if a jacobian has wrong dimensionality:
    ###          check it or garbage in, garbage out?

    def test_x0_multidim(self):
        # we don't know what to do with x0.ndim > 1
        x0 = np.ones(4).reshape(2,2)
        assert_raises(ValueError, f11, x0, **dict(method=self.meth))


class BoundsMixin(object):
    # collect bounds-related tests here
    def test_infeasible(self):
        assert_raises(ValueError, least_squares, f11, 2.,
                                  **dict(bounds=(3., 4), method=self.meth))
                                                 
    def test_bounds_three_el(self):
        assert_raises(ValueError, least_squares, f11, 2.,
                                  **dict(bounds=(1., 2, 3), method=self.meth))

    def test_in_bounds(self):
        # TODO: test that a solution is in bounds for the minimum inside
        #       repear w/ a minimum @ the boundary
        raise AssertionError

    ### TODO: test various combinations of bounds: scalar & vector, two scalars etc


class TestDogbox(BaseMixin, BoundsMixin, TestCase):
    meth = 'dogbox'


class TestTRF(BaseMixin, BoundsMixin, TestCase):
    meth = 'trf'


class TestLM(Base, TestCase):
    meth = 'lm'

    ### TODO: test that options['epsfcn'] raises TypeError
    ###       test that non-default bounds raise


#
# One-off tests which do not need parameterization or are method-specific
#
def test_basic():
    # test that 'method' arg is really optional
    res = least_squares(f11, 2.)
    assert_allclose(res.x, 0.)
    assert_allclose(res.fun, f11(res.x))


if __name__ == "__main__":
    run_module_suite()
