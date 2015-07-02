from warnings import simplefilter
import numpy as np
from numpy.testing import (run_module_suite, assert_, assert_allclose,
                           assert_warns, TestCase, assert_raises, assert_equal,
                           assert_almost_equal)

from bounded_lsq import least_squares


simplefilter('ignore')


def trivial_fun(x, a=0):
    return (x - a)**2 + 5.0


def trivial_jac(x, a=0.0):
    return 2 * (x - a)


def rosenbrock_fun(x):
    return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])


def rosenbrock_jac(x):
    return np.array([
        [-20 * x[0], 10],
        [-1, 0]
    ])

#
# Parametrize basic smoke tests across methods
#


class BaseMixin(object):
    def test_basic(self):
        # Test that the basic calling sequence works.
        res = least_squares(trivial_fun, 2., method=self.method)
        assert_allclose(res.x, 0, atol=1e-4)
        assert_allclose(res.fun, trivial_fun(res.x))

    def test_args_kwargs(self):
        # Test that args and kwargs are passed correctly to the functions.
        # And that kwargs are not supported by 'lm'.
        a = 3.0
        for jac in ['2-point', '3-point', trivial_jac]:
            res = least_squares(trivial_fun, 2.0, jac, args=(a,),
                                method=self.method)
            assert_allclose(res.x, a, rtol=1e-4)
            assert_allclose(res.fun, trivial_fun(res.x, a))

            assert_raises(TypeError, least_squares, trivial_fun, 2.0,
                          args=(3, 4,), method=self.method)
        
            # Test that kwargs works for everything except 'lm.
            if self.method == 'lm':
                assert_raises(ValueError, least_squares, trivial_fun, 2.0,
                              kwargs={'a': a}, method=self.method)
            else:
                res = least_squares(trivial_fun, 2.0, jac, kwargs={'a': a},
                                    method=self.method)
                assert_allclose(res.x, a, rtol=1e-4)
                assert_allclose(res.fun, trivial_fun(res.x, a))
                assert_raises(TypeError, least_squares, trivial_fun, 2.0,
                              kwargs={'kaboom': 3}, method=self.method)

    def test_jac_options(self):
        for jac in ['2-point', '3-point', trivial_jac]:
            res = least_squares(trivial_fun, 2.0, jac, method=self.method)
            assert_allclose(res.x, 0, atol=1e-4)
        assert_raises(ValueError, least_squares, trivial_fun, 2.0, jac='oops',
                      method=self.method)

    def test_nfev_options(self):
        for max_nfev in [None, 20]:
            res = least_squares(trivial_fun, 2.0, max_nfev=max_nfev,
                                method=self.method)
            assert_allclose(res.x, 0, atol=1e-4)

    def test_scaling_options(self):
        for scaling in [1.0, np.array([2.0]), 'jac']:
            res = least_squares(trivial_fun, 2.0, scaling=scaling)
            assert_allclose(res.x, 0)
        assert_raises(ValueError, least_squares, trivial_fun,
                      2.0, scaling='auto', method=self.method)
        assert_raises(ValueError, least_squares, trivial_fun,
                      2.0, scaling=-1.0, method=self.method)

    def test_diff_step(self):
        # res1 and res2 should be equivalent.
        # res2 and res3 should be different.
        res1 = least_squares(trivial_fun, 2.0, diff_step=1e-2,
                             method=self.method)
        res2 = least_squares(trivial_fun, 2.0, diff_step=-1e-2,
                             method=self.method)
        res3 = least_squares(trivial_fun, 2.0,
                             diff_step=None, method=self.method)
        assert_allclose(res1.x, 0, atol=1e-4)
        assert_allclose(res2.x, 0, atol=1e-4)
        assert_allclose(res3.x, 0, atol=1e-4)
        assert_equal(res1.x, res2.x)
        assert_equal(res1.nfev, res2.nfev)
        assert_(res2.nfev != res3.nfev)

    def test_incorrect_options_usage(self):
        assert_raises(TypeError, least_squares, trivial_fun, 2.0,
                      method=self.method, options={'no_such_option': 100})
        assert_raises(TypeError, least_squares, trivial_fun, 2.0,
                      method=self.method, options={'max_nfev': 100})

    def test_tolerance_thresholds(self):
        assert_warns(UserWarning, least_squares, trivial_fun, 2.0, ftol=0.0,
                     method=self.method)
        res = least_squares(trivial_fun, 2.0, ftol=1e-20, xtol=-1.0, gtol=0.0,
                            method=self.method)
        assert_allclose(res.x, 0, atol=1e-4)

    def test_full_result(self):
        res = least_squares(trivial_fun, 2.0, method=self.method)
        # Use assert_almost_equal to check shapes of arrays too.
        assert_almost_equal(res.x, np.array([0]), decimal=1)
        assert_almost_equal(res.obj_value, 25)
        assert_almost_equal(res.fun, np.array([5]))
        assert_almost_equal(res.jac, np.array([[0.0]]), decimal=2)
        # 'lm' works weired on this problem
        assert_almost_equal(res.optimality, 0, decimal=3)
        assert_equal(res.active_mask, np.array([0]))
        if self.method == 'lm':
            assert_(res.nfev < 25)
        else:
            assert_(res.nfev < 10)
        if self.method == 'lm':
            assert_(res.njev is None)
        else:
            assert_(res.njev < 10)
        assert_(res.status > 0)
        assert_(res.success)

    def test_rosenbrock(self):
        x0 = [-2, 1]
        x_opt = [1, 1]
        for scaling in [1.0, np.array([1.0, 5.0]), 'jac']:
            for jac in ['2-point', '3-point', rosenbrock_jac]:
                res = least_squares(rosenbrock_fun, x0, jac, scaling=scaling,
                                    method=self.method)
                assert_allclose(res.x, x_opt)

    ### TODO: what do we do if a jacobian has wrong dimensionality:
    ###          check it or garbage in, garbage out?

    def test_x0_multidimensional(self):
        # we don't know what to do with x0.ndim > 1
        x0 = np.ones(4).reshape(2, 2)
        assert_raises(ValueError, least_squares, trivial_fun, x0,
                      method=self.method)


class BoundsMixin(object):
    # collect bounds-related tests here
    def test_infeasible(self):
        assert_raises(ValueError, least_squares, trivial_fun, 2.,
                                  **dict(bounds=(3., 4), method=self.method))
                                                 
    def test_bounds_three_el(self):
        assert_raises(ValueError, least_squares, trivial_fun, 2.,
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
    res = least_squares(trivial_fun, 2.0)
    assert_allclose(res.x, 0, atol=1e-10)
    assert_allclose(res.fun, trivial_fun(res.x))


if __name__ == "__main__":
    run_module_suite()
