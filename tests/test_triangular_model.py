import numpy as np
import pytest

from neurobiases import TriangularModel, utils
from neurobiases.solver_utils import marginal_log_likelihood_linear_tm_wrapper
from numpy.testing import assert_allclose, assert_raises


def test_random_state():
    """Tests that the random seeds correctly set the coupling and tuning
    parameters."""
    tm1 = TriangularModel(coupling_rng=1234, tuning_rng=4321)
    tm2 = TriangularModel(coupling_rng=1234, tuning_rng=9876)
    tm3 = TriangularModel(coupling_rng=9876, tuning_rng=4321)
    tm4 = TriangularModel(coupling_rng=9876, tuning_rng=9876)

    # Random seed for parameters
    assert_allclose(tm1.a.ravel(), tm2.a.ravel())
    assert_raises(AssertionError, assert_allclose, tm1.b.ravel(), tm2.b.ravel())
    assert_allclose(tm3.a.ravel(), tm4.a.ravel())
    assert_raises(AssertionError, assert_allclose, tm3.b.ravel(), tm4.b.ravel())
    assert_allclose(tm1.b.ravel(), tm3.b.ravel())
    assert_allclose(tm1.B.ravel(), tm3.B.ravel())
    assert_raises(AssertionError, assert_allclose, tm1.a.ravel(), tm3.a.ravel())
    assert_allclose(tm2.b.ravel(), tm4.b.ravel())
    assert_allclose(tm2.B.ravel(), tm4.B.ravel())
    assert_raises(AssertionError, assert_allclose, tm2.a.ravel(), tm4.a.ravel())

    # Random seed for data
    X1, Y1, y1 = tm1.generate_samples(n_samples=50, rng=2332)
    X2, Y2, y2 = tm1.generate_samples(n_samples=50, rng=2332)
    assert_allclose(X1, X2)
    assert_allclose(Y1, Y2)
    assert_allclose(y1, y2)


def test_sparsity():
    """Test that the sparsity inputs correctly set the number of non-zero
    parameters."""
    N = 20
    coupling_sparsity = 0.25
    M = 40
    tuning_sparsity = 0.1
    tm = TriangularModel(N=N, coupling_sparsity=coupling_sparsity,
                         M=M, tuning_sparsity=tuning_sparsity,
                         coupling_rng=1234,
                         tuning_rng=4321)
    assert np.count_nonzero(tm.a.ravel() == 0) == int(N * coupling_sparsity)
    assert np.count_nonzero(tm.b.ravel() == 0) == int(M * tuning_sparsity)


def test_neural_variance():
    """Tests that the analytic expressions for the variance of the neural
    activity are correctly calculated."""
    tm1 = TriangularModel(parameter_design='direct_response', N=10, M=20, K=2)
    tm2 = TriangularModel(parameter_design='basis_functions', N=10, M=20, K=2)

    for tm in [tm1, tm2]:
        # Generate samples
        X, Y, y = tm.generate_samples(n_samples=100000, return_noise=False)

        # Non-target signal variance
        non_target_signal = np.dot(X, tm.B)
        empirical_variance = np.var(non_target_signal, axis=0)
        true_variance = tm.non_target_signal_variance(limits=(0, 1))
        assert_allclose(true_variance, empirical_variance, rtol=0.1)

        # Non-target variance
        empirical_variance = np.var(Y, axis=0)
        true_variance = tm.non_target_variance(limits=(0, 1))
        assert_allclose(true_variance, empirical_variance, rtol=0.1)

        # Target signal variance
        target_signal = np.dot(X, tm.b) + np.dot(Y, tm.a)
        empirical_variance = np.var(target_signal)
        true_variance = tm.target_signal_variance(limits=(0, 1))
        assert_allclose(true_variance, empirical_variance, rtol=0.1)

        # Target neuron variance
        empirical_variance = np.var(y)
        true_variance = tm.target_variance(limits=(0, 1))
        assert_allclose(true_variance, empirical_variance, rtol=0.1)


def test_snr():
    """Tests that the total variance of the neural activity reflects the
    signal-to-noise ratio."""
    tm1 = TriangularModel(parameter_design='direct_response', N=10, M=20, K=2)
    tm2 = TriangularModel(parameter_design='basis_functions', N=10, M=20, K=2)

    for tm in [tm1, tm2]:
        snr = tm.noise_kwargs['snr']
        # Generate samples
        X, Y, y = tm.generate_samples(n_samples=100000, return_noise=False)

        # Non-target neural activity
        true_signal_variance = tm.non_target_signal_variance(limits=(0, 1))
        true_total_variance = (1 + 1. / snr) * true_signal_variance
        empirical_variance = np.var(Y, axis=0)
        assert_allclose(true_total_variance, empirical_variance, rtol=0.1)

        # Target neural activity
        true_signal_variance = tm.target_signal_variance(limits=(0, 1))
        true_noise_variance = true_signal_variance / snr
        noise = y - np.dot(X, tm.b) - np.dot(Y, tm.a)
        empirical_noise_variance = np.var(noise)
        assert_allclose(true_noise_variance, empirical_noise_variance, rtol=0.1)


def test_noise_clusters():
    """Test that clustered noise correlations are generated correctly."""
    corr_cluster = 0.3
    # One latent factor
    tm = TriangularModel(parameter_design='direct_response',
                         K=1,
                         corr_cluster=corr_cluster)
    # Compare true and generated correlation matrices
    corr = utils.cov2corr(tm.get_noise_cov())
    true_corr = corr_cluster * np.ones((tm.N + 1, tm.N + 1))
    np.fill_diagonal(true_corr, np.ones(tm.N + 1))
    assert_allclose(corr, true_corr)

    # Three latent factors
    tm = TriangularModel(parameter_design='direct_response',
                         N=6,
                         K=3,
                         corr_cluster=corr_cluster)

    # compare true and generated correlation matrices
    corr = utils.cov2corr(tm.get_noise_cov())
    true_corr = np.array([
        [1, 0, 0, corr_cluster, corr_cluster, 0, 0],
        [0, 1, corr_cluster, 0, 0, 0, 0],
        [0, corr_cluster, 1, 0, 0, 0, 0],
        [corr_cluster, 0, 0, 1, corr_cluster, 0, 0],
        [corr_cluster, 0, 0, corr_cluster, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, corr_cluster],
        [0, 0, 0, 0, 0, corr_cluster, 1]
    ])
    assert_allclose(corr, true_corr, atol=1e-6)


def test_identifiability():
    """Tests that the identifiability transform leaves the likelihood
    unchanged."""
    tm1 = TriangularModel(parameter_design='direct_response', N=10, M=20, K=2)
    tm2 = TriangularModel(parameter_design='basis_functions', N=10, M=20, K=2)

    for tm in [tm1, tm2]:
        # Generate samples
        X, Y, y = tm.generate_samples(n_samples=10000, return_noise=False)
        # Likelihood with original parameters
        pre_ll = marginal_log_likelihood_linear_tm_wrapper(
            X=X, Y=Y, y=y, tm=tm
        )
        # Identifiability transform
        delta = np.random.normal(loc=0, scale=5.0, size=tm.K)
        tm.identifiability_transform(delta)
        # Likelihood with transfored parameters
        post_ll = marginal_log_likelihood_linear_tm_wrapper(
            X=X, Y=Y, y=y, tm=tm
        )
        assert_allclose(pre_ll, post_ll)


def test_identifiability_warning():
    """Tests that a warning is received if the model is not identifiable."""
    # Check that identifiability warning is raised
    with pytest.warns(RuntimeWarning) as record:
        TriangularModel(coupling_sparsity=0., tuning_sparsity=0., K=3)
    assert len(record) == 1
    # Check that identifiability warning is not raised
    with pytest.warns(None) as record:
        TriangularModel(coupling_sparsity=0.5, tuning_sparsity=0.5, K=3)
    assert len(record) == 0
