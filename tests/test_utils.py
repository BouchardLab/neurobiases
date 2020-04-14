import numpy as np

from neurobiases import utils
from numpy.testing import assert_allclose


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
        bf_center=center,
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
        bf_center=centers,
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
        bf_center=centers,
        bf_scale=scale)
    # calculate true and empirical covariances
    cov_true = utils.bf_cov(centers, scale, limits=(0, 1))
    cov_empirical = np.cov(features.T)
    assert_allclose(cov_true, cov_empirical, rtol=0.1)


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
        bf_center=centers,
        bf_scale=scale)
    # calculate true and empirical weights
    var_true = utils.bf_sum_var(weights, centers, scale, limits=(0, 1))
    var_empirical = np.var(np.dot(features, weights))
    # assert empirical variance
    assert_allclose(var_true, var_empirical, rtol=0.1)
