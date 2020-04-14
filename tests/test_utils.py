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
