import numpy as np

from neurobiases import TriangularModel
from numpy.testing import assert_allclose


def generate_model():
    """Generates a triangular model."""
    tuning_kwargs = {
        'M': 10,
        'sparsity': 0.,
        'random_state': 2332,
        'bf_scale': 1/100,
        'add_noise': True,
        'noise_scale': 0.20,
    }
    coupling_kwargs = {
        'N': 13,
        'sparsity': 0.5,
        'distribution': 'symmetric_lognormal',
        'loc': -1,
        'scale': 0.5,
        'random_state': 2332
    }
    noise_kwargs = {
        'K': 2,
        'snr': 3,
        'corr_max': 0.3,
        'corr_min': -0.1,
        'L': 1
    }
    tm = TriangularModel(
        model='linear',
        parameter_design='basis_functions',
        tuning_kwargs=tuning_kwargs,
        coupling_kwargs=coupling_kwargs,
        noise_kwargs=noise_kwargs)
    return tm


def test_neural_variance():
    """Tests that the analytic expressions for the variance of the neural
    activity are correctly calculated."""
    tm = generate_model()
    # generate samples
    _, X, Y, y = tm.generate_samples(n_samples=100000, return_noise=False)

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
    tm = generate_model()
    snr = tm.noise_kwargs['snr']

    # generate samples
    _, _, Y, y = tm.generate_samples(n_samples=100000, return_noise=False)

    # non-target neural activity
    true_signal_variance = tm.non_target_signal_variance(limits=(0, 1))
    true_total_variance = (1 + 1. / snr) * true_signal_variance
    empirical_variance = np.var(Y, axis=0)
    assert_allclose(true_total_variance, empirical_variance, rtol=0.1)

    # target neural activity
    true_signal_variance = tm.target_signal_variance(limits=(0, 1))
    true_total_variance = (1 + 1. / snr) * true_signal_variance
    empirical_variance = np.var(y)
    assert_allclose(true_total_variance, empirical_variance, rtol=0.1)
