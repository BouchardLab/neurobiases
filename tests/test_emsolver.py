import numpy as np
import pytest

from neurobiases import EMSolver
from numpy.testing import assert_allclose, assert_array_equal

from utils import (generate_bf_cluster_model,
                   generate_dr_cluster_model)


def test_marginal_likelihood():
    """Tests the marginal likelihood for the linear TM in a simple case."""
    tm1 = generate_bf_cluster_model()
    tm2 = generate_dr_cluster_model()

    for tm in [tm1, tm2]:
        X, Y, y = tm.generate_samples(n_samples=1000)
        solver = EMSolver(X, Y, y, K=1)
        solver.set_params(L=np.zeros_like(solver.L))
        # test the marginal likelihood
        empirical_likelihood = solver.marginal_log_likelihood()
        true_likelihood = -0.5 * ((y.T @ y).item() + np.trace(Y @ Y.T))
        assert_allclose(true_likelihood, empirical_likelihood)


@pytest.mark.slow
def test_em_update():
    """Tests that the EM optimization always increases the marginal likelihood."""
    K = 1
    tm1 = generate_bf_cluster_model(K=K)
    tm2 = generate_dr_cluster_model(K=K)

    for tm in [tm1, tm2]:
        X, Y, y = tm.generate_samples(n_samples=100)
        solver = EMSolver(X=X, Y=Y, y=y, K=K, max_iter=50, tol=0)
        mlls = solver.fit_em(mll_curve=True)
        assert mlls.size == solver.max_iter
        assert np.all(np.ediff1d(mlls) > 0)


@pytest.mark.slow
def test_em_mask():
    """Tests that, when provided a mask, EMSolver maintains the selection
    profiles during fitting."""
    K = 2
    tm = generate_bf_cluster_model(K=K)

    X, Y, y = tm.generate_samples(n_samples=100)
    a_mask, b_mask, B_mask = tm.get_masks()
    solver = EMSolver(X=X, Y=Y, y=y, K=K, max_iter=10, tol=0,
                      a_mask=a_mask, b_mask=b_mask, B_mask=B_mask)
    solver.fit_em()
    assert_array_equal(a_mask, solver.a.ravel() != 0)
    assert_array_equal(b_mask, solver.b.ravel() != 0)
    assert_array_equal(B_mask, solver.B != 0)
