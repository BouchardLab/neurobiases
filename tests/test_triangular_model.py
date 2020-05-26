import numpy as np

from utils import (generate_bf_cluster_model,
                   generate_dr_cluster_model)
from neurobiases import TriangularModel, utils
from neurobiases.solver_utils import marginal_log_likelihood_linear_tm_wrapper
from numpy.testing import assert_allclose


def test_neural_variance():
    """Tests that the analytic expressions for the variance of the neural
    activity are correctly calculated."""
    tm1 = generate_bf_cluster_model()
    tm2 = generate_dr_cluster_model()

    for tm in [tm1, tm2]:
        # generate samples
        X, Y, y = tm.generate_samples(n_samples=100000, return_noise=False)

        # non-target signal variance
        non_target_signal = np.dot(X, tm.B)
        empirical_variance = np.var(non_target_signal, axis=0)
        true_variance = tm.non_target_signal_variance(limits=(0, 1))
        assert_allclose(true_variance, empirical_variance, rtol=0.1)

        # non-target variance
        empirical_variance = np.var(Y, axis=0)
        true_variance = tm.non_target_variance(limits=(0, 1))
        assert_allclose(true_variance, empirical_variance, rtol=0.1)

        # target signal variance
        target_signal = np.dot(X, tm.b) + np.dot(Y, tm.a)
        empirical_variance = np.var(target_signal)
        true_variance = tm.target_signal_variance(limits=(0, 1))
        assert_allclose(true_variance, empirical_variance, rtol=0.1)

        # target neuron variance
        empirical_variance = np.var(y)
        true_variance = tm.target_variance(limits=(0, 1))
        assert_allclose(true_variance, empirical_variance, rtol=0.1)


def test_snr():
    """Tests that the total variance of the neural activity reflects the
    signal-to-noise ratio."""
    tm1 = generate_bf_cluster_model()
    tm2 = generate_dr_cluster_model()

    for tm in [tm1, tm2]:
        snr = tm.noise_kwargs['snr']
        # generate samples
        X, Y, y = tm.generate_samples(n_samples=100000, return_noise=False)

        # non-target neural activity
        true_signal_variance = tm.non_target_signal_variance(limits=(0, 1))
        true_total_variance = (1 + 1. / snr) * true_signal_variance
        empirical_variance = np.var(Y, axis=0)
        assert_allclose(true_total_variance, empirical_variance, rtol=0.1)

        # target neural activity
        true_signal_variance = tm.target_signal_variance(limits=(0, 1))
        true_noise_variance = true_signal_variance / snr
        noise = y - np.dot(X, tm.b) - np.dot(Y, tm.a)
        empirical_noise_variance = np.var(noise)
        assert_allclose(true_noise_variance, empirical_noise_variance, rtol=0.1)


def test_noise_clusters():
    """Test that clustered noise correlations are generated correctly."""
    corr_cluster = 0.3
    # one latent factor
    # generate the triangular model
    tuning_kwargs, coupling_kwargs, noise_kwargs, stim_kwargs = \
        TriangularModel.generate_kwargs(
            parameter_design='direct_response', M=40, tuning_sparsity=0.2,
            tuning_noise_scale=0.20, N=3, K=1, corr_cluster=corr_cluster
        )
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        tuning_kwargs=tuning_kwargs,
        coupling_kwargs=coupling_kwargs,
        stim_kwargs=stim_kwargs,
        noise_kwargs=noise_kwargs)
    # compare true and generated correlation matrices
    corr = utils.cov2corr(tm.get_noise_cov())
    true_corr = corr_cluster * np.ones((tm.N + 1, tm.N + 1))
    np.fill_diagonal(true_corr, np.ones(tm.N + 1))
    assert_allclose(corr, true_corr)

    # three latent factors
    # generate the triangular model
    tuning_kwargs, coupling_kwargs, noise_kwargs, stim_kwargs = \
        TriangularModel.generate_kwargs(
            parameter_design='direct_response', M=40, tuning_sparsity=0.2,
            tuning_noise_scale=0.20, N=6, K=3, corr_cluster=corr_cluster
        )
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        tuning_kwargs=tuning_kwargs,
        coupling_kwargs=coupling_kwargs,
        stim_kwargs=stim_kwargs,
        noise_kwargs=noise_kwargs)
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
    tm1 = generate_bf_cluster_model()
    tm2 = generate_dr_cluster_model()

    for tm in [tm1, tm2]:
        # generate samples
        X, Y, y = tm.generate_samples(n_samples=10000, return_noise=False)
        pre_ll = marginal_log_likelihood_linear_tm_wrapper(
            X=X, Y=Y, y=y, tm=tm
        )
        delta = np.random.normal(loc=0, scale=5.0, size=tm.K)
        tm.identifiability_transform(delta)
        post_ll = marginal_log_likelihood_linear_tm_wrapper(
            X=X, Y=Y, y=y, tm=tm
        )
        assert_allclose(pre_ll, post_ll)
