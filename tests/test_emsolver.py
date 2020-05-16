import numpy as np

from neurobiases import TriangularModel, EMSolver
from numpy.testing import assert_allclose


def load_tm():
    """Loads a random triangular model."""
    tuning_kwargs, coupling_kwargs, noise_kwargs, stim_kwargs = \
        TriangularModel.generate_piecewise_kwargs(
            M=50, N=23, K=1, corr_cluster=0.3, corr_back=0.1,
            tuning_sparsity=0.6, coupling_sparsity=0.5,
            tuning_random_state=233332, coupling_random_state=2)
    tm = TriangularModel(
        model='linear',
        parameter_design='piecewise',
        tuning_kwargs=tuning_kwargs,
        coupling_kwargs=coupling_kwargs,
        noise_kwargs=noise_kwargs,
        stim_kwargs=stim_kwargs)
    return tm


def test_marginal_likelihood():
    """Tests the marginal likelihood for the linear TM in a simple case."""
    tm = load_tm()
    X, Y, y = tm.generate_samples(n_samples=1000)
    solver = EMSolver(X, Y, y, K=1)
    solver.set_params(L=np.zeros_like(solver.L))
    # test the marginal likelihood
    empirical_likelihood = solver.marginal_log_likelihood()
    true_likelihood = -0.5 * ((y.T @ y).item() + np.trace(Y @ Y.T))
    assert_allclose(true_likelihood, empirical_likelihood)
