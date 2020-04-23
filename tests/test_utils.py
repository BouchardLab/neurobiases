import numpy as np

from neurobiases import utils
from numpy.testing import (assert_allclose,
                           assert_equal)


def test_bf_mean_var():
    """Tests whether the mean and variance of a basis function conditioned on
    a uniform stimulus is correctly calculated."""
    # choose a basis function and stimuli
    center = np.random.uniform(low=0, high=1)
    scale = np.random.uniform(low=0.1, high=0.25)
    stimuli = np.random.uniform(low=0, high=1, size=100000)
    # calculate basis function output for stimuli
    features = utils.calculate_tuning_features(
        stimuli=stimuli,
        bf_centers=center,
        bf_scale=scale)
    # calculate true and empirical mean
    mean_true = utils.bf_mean(center, scale, limits=(0, 1))
    mean_empirical = np.mean(features)
    # calculate true and empirical variance
    var_true = utils.bf_variance(center, scale, limits=(0, 1))
    var_empirical = np.var(features)
    # validate that true estimates capture the empirical estimates
    assert_allclose(mean_true, mean_empirical, rtol=0.05)
    assert_allclose(var_true, var_empirical, rtol=0.05)


def test_bf_pairwise_cov():
    """Tests whether the covariance of two basis functions is correctly
    calculated."""
    # choose basis functions and stimuli.
    centers = np.random.uniform(low=0, high=1., size=2)
    scale = np.random.uniform(low=0.1, high=0.25)
    stimuli = np.random.uniform(low=0, high=1, size=100000)
    # calculate basis function output for stimuli
    features = utils.calculate_tuning_features(
        stimuli=stimuli,
        bf_centers=centers,
        bf_scale=scale)
    # calculate true and empirical covariances
    covar_true = utils.bf_pairwise_cov(centers, scale, limits=(0, 1))
    covar_empirical = np.mean(np.prod(features, axis=1)) - np.prod(np.mean(features, axis=0))
    # validate the empirical estimates
    assert_allclose(covar_true, covar_empirical, rtol=0.05)


def test_bf_cov():
    """Tests whether a full covariance matrix of basis function outputs is
    correctly calcaulted."""
    # choose basis functions and stimuli.
    centers = np.random.uniform(low=0, high=1., size=10)
    scale = np.random.uniform(low=0.1, high=0.25)
    stimuli = np.random.uniform(low=0, high=1, size=1000000)
    # calculate basis function output for stimuli
    features = utils.calculate_tuning_features(
        stimuli=stimuli,
        bf_centers=centers,
        bf_scale=scale)
    # calculate true and empirical covariances
    cov_true = utils.bf_cov(centers, scale, limits=(0, 1))
    cov_empirical = np.cov(features.T)
    assert_allclose(cov_true, cov_empirical, rtol=0.2)


def test_bf_sum_var():
    """Tests whether the variance of the linear combination of basis functions
    is calculated correctly."""
    # choose basis functions and stimuli.
    weights = np.random.uniform(low=0, high=5, size=10)
    centers = np.random.uniform(low=0, high=1., size=10)
    scale = np.random.uniform(low=0.1, high=0.25)
    stimuli = np.random.uniform(low=0, high=1, size=100000)
    # calculate basis function output for stimuli
    features = utils.calculate_tuning_features(
        stimuli=stimuli,
        bf_centers=centers,
        bf_scale=scale)
    # calculate true and empirical weights
    var_true = utils.bf_sum_var(weights, centers, scale, limits=(0, 1))
    var_empirical = np.var(np.dot(features, weights))
    # assert empirical variance
    assert_allclose(var_true, var_empirical, rtol=0.1)


def test_noise_corr_matrix():
    """Tests whether the noise correlation matrix is generated correctly."""
    tuning_prefs = np.array([0.25, 0.50, 0.75, 1.0])
    n_neurons = tuning_prefs.size
    corr_max = 1.0
    L = 0.25

    # test without circular stimulus
    noise_corrs = utils.noise_correlation_matrix(tuning_prefs=tuning_prefs,
                                                 corr_max=corr_max,
                                                 L=L)
    # test diagonal of noise correlation matrix
    assert_equal(np.diag(noise_corrs), np.ones(n_neurons))
    # test that off-diagonal terms are equal
    assert np.unique(np.diag(noise_corrs, k=1)).size == 1
    # test with circular stimulus
    noise_corrs = utils.noise_correlation_matrix(tuning_prefs=tuning_prefs,
                                                 corr_max=corr_max,
                                                 L=L,
                                                 circular_stim=1)
    reverse_diagonal = np.diag(np.flip(noise_corrs, axis=1))
    assert np.unique(reverse_diagonal).size == 1


def test_cov_corr_functions():
    """Test the covariance to correlation functions."""
    X = np.random.normal(loc=0, scale=0.25, size=(100, 50))
    cov = np.cov(X, rowvar=False)
    corr = np.corrcoef(X, rowvar=False)

    assert_allclose(corr, utils.cov2corr(cov))
    assert_allclose(cov, utils.corr2cov(corr, np.diag(cov)))
