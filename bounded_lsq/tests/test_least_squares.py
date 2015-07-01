from warnings import simplefilter
import numpy as np
from numpy.testing import (run_module_suite, assert_, assert_allclose,
                           assert_warns, TestCase, assert_raises, assert_equal)

from bounded_lsq import least_squares


simplefilter('ignore')


def fun(x, a=0):
    return (x - a)**2 + 5.0


def jac(x, a=0.0):
    return 2 * (x - a)


#
# Parametrize basic smoke tests across methods
#


class BaseMixin(object):
    def test_basic(self):
        # Test that the basic calling sequence works.
        res = least_squares(fun, 2., method=self.method)
        assert_allclose(res.x, 0, atol=1e-4)
        assert_allclose(res.fun, fun(res.x))

    def test_args_kwargs(self):
        # Test that args and kwargs are passed correctly to the functions.
        # And that kwargs are not supported by 'lm'.
        a = 3.0
        for j in ['2-point', '3-point', jac]:
            res = least_squares(fun, 2.0, j, args=(a,), method=self.method)
            assert_allclose(res.x, a, rtol=1e-4)
            assert_allclose(res.fun, fun(res.x, a))

            assert_raises(TypeError, least_squares, fun, 2.0,
                          args=(3, 4,), method=self.method)
        
            # Test that kwargs works for everything except 'lm.
            if self.method == 'lm':
                assert_raises(ValueError, least_squares, fun, 2.0,
                              kwargs={'a': a}, method=self.method)
            else:
                res = least_squares(fun, 2.0, j, kwargs={'a': a},
                                    method=self.method)
                assert_allclose(res.x, a, rtol=1e-4)
                assert_allclose(res.fun, fun(res.x, a))
                assert_raises(TypeError, least_squares, fun, 2.0,
                              kwargs={'kaboom': 3}, method=self.method)

    def test_jac_options(self):
        for j in ['2-point', '3-point', jac]:
            res = least_squares(fun, 2.0, j)
            assert_allclose(res.x, 0)
        assert_raises(ValueError, least_squares, fun, 2.0, jac='oops')

    def test_nfev_options(self):
        for max_nfev in [None, 20]:
            res = least_squares(fun, 2.0, max_nfev=max_nfev)
            assert_allclose(res.x, 0)

    def test_scaling_options(self):
        for scaling in [1.0, np.array([2.0]), 'jac']:
            res = least_squares(fun, 2.0, scaling=scaling)
            assert_allclose(res.x, 0)
        assert_raises(ValueError, least_squares, fun, 2.0, scaling='auto')
        assert_raises(ValueError, least_squares, fun, 2.0, scaling=-1.0)

    def test_diff_step(self):
        res1 = least_squares(fun, 2.0, xtol=1e-5, diff_step=1e-2)
        res2 = least_squares(fun, 2.0, xtol=1e-5, diff_step=-1e-2)
        res3 = least_squares(fun, 2.0, xtol=1e-5, diff_step=None)
        assert_allclose(res1.x, 0)
        assert_allclose(res2.x, 0)
        assert_allclose(res3.x, 0)
        assert_equal(res1.x, res2.x)
        assert_equal(res1.nfev, res2.nfev)
        assert_(res2.nfev > res3.nfev)

    def test_incorrect_options_usage(self):
        assert_raises(TypeError, least_squares, fun, 2.0,
                      options={'no_such_option': 100})
        assert_raises(TypeError, least_squares, fun, 2.0,
                      options={'max_nfev': 100})

    def test_tolerance_thresolds(self):
        assert_warns(UserWarning, least_squares, fun, 2.0, ftol=0.0)
        res = least_squares(fun, 2.0, ftol=1e-20, xtol=-1.0, gtol=0.0)
        assert_allclose(res.x, 0)

    ### TODO: add a less trivial function, check that it works
    # (just a smoke test).
    
    ### TODO: what do we do if a jacobian has wrong dimensionality:
    ###          check it or garbage in, garbage out?

    def test_x0_multidimensional(self):
        # we don't know what to do with x0.ndim > 1
        x0 = np.ones(4).reshape(2, 2)
        assert_raises(ValueError, least_squares, fun, x0, method=self.method)


class BoundsMixin(object):
    # collect bounds-related tests here
    def test_infeasible(self):
        assert_raises(ValueError, least_squares, fun, 2.,
                                  **dict(bounds=(3., 4), method=self.method))
                                                 
    def test_bounds_three_el(self):
        assert_raises(ValueError, least_squares, fun, 2.,
                                  **dict(bounds=(1., 2, 3),
                                         method=self.method))

    def test_in_bounds(self):
        # TODO: test that a solution is in bounds for the minimum inside
        #       repear w/ a minimum @ the boundary
        pass
        # raise AssertionError

    ### TODO: test various combinations of bounds: scalar & vector, two scalars etc


class TestDogbox(BaseMixin, BoundsMixin, TestCase):
    method = 'dogbox'


class TestTRF(BaseMixin, BoundsMixin, TestCase):
    method = 'trf'


class TestLM(BaseMixin, TestCase):
    method = 'lm'

    ### TODO: test that options['epsfcn'] raises TypeError
    ###       test that non-default bounds raise


#
# One-off tests which do not need parameterization or are method-specific
#
def test_basic():
    # test that 'method' arg is really optional
    res = least_squares(fun, 2.0)
    assert_allclose(res.x, 0, atol=1e-10)
    assert_allclose(res.fun, fun(res.x))


if __name__ == "__main__":
    run_module_suite()
