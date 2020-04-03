import numpy as np


def read_attribute_dict(attributes):
    copy = {}
    for key, val in attributes.items():
        if val == '':
            copy[key] = None
        else:
            copy[key] = val
    return copy


def calculate_tuning_features(stimuli, bf_pref_tuning, bf_scale):
    """Get basis function tuning features given a set of input stimuli.

    Parameters
    ----------
    stimuli : np.ndarray, shape (n_samples,)
        The input stimuli.

    bf_pref_tuning : np.ndarray, shape (n_parameters,)
        The locations of each basis function.

    bf_scale : float
        The spread of each basis function.

    Returns
    -------
    tuning_features : np.ndarray, shape (n_samples, n_features)
        The tuning features for each stimulus.
    """
    tuning_features = np.exp(
        -0.5 * np.subtract.outer(stimuli, bf_pref_tuning)**2 / bf_scale
    )
    return tuning_features


def calculate_tuning_curves(
    B, bf_pref_tuning, bf_scale, n_stimuli=10000, limits=(0, 1), intercepts=None
):
    """Calculates the tuning curves using Gaussian basis functions for a provided
    set of tuning parameters.

    Parameters
    ----------
    B : np.ndarray, shape (n_parameters, n_neurons)
        The tuning parameters for a set of neurons, with each column referring
        to a neuron.

    bf_pref_tuning : np.ndarray, shape (n_parameters,)
        The locations of each basis function.

    bf_scale : float
        The spread of each basis function.

    n_stimuli : int
        The number of stimuli to tile the plane.

    limits : tuple
        The lower and upper bounds to sample points from the stimulus space.

    intercepts : np.ndarray, shape (n_neurons,), default None
        The intercepts of each tuning curve. If None, intercepts are assumed
        to be zero.

    Returns
    -------
    tuning_curves : np.ndarray, shape (n_stimuli, n_neurons)
        The tuning curve for each neuron.
    """
    # check if the tuning parameters are in the right shape
    if B.ndim == 1:
        B = B[..., np.newaxis]
    # handle intercepts
    if intercepts is None:
        intercepts = np.zeros((1, B.shape[1]))
    if intercepts.ndim == 1:
        intercepts = intercepts[np.newaxis]
    # get stimuli tiling the plane
    stimuli = np.linspace(limits[0], limits[1], n_stimuli)
    # get responses for each basis function
    bf_responses = np.exp(
        -0.5 * np.subtract.outer(stimuli, bf_pref_tuning)**2 / bf_scale
    )
    # calculate responses for each neuron
    tuning_curves = intercepts + np.dot(bf_responses, B)
    return tuning_curves


def calculate_pref_tuning(
    B, bf_pref_tuning, bf_scale, n_stimuli=10000, limits=(0, 1)
):
    """Calculate the preferred tuning using tuning parameters.

    Parameters
    ----------
    B : np.ndarray, shape (n_parameters, n_neurons)
        The tuning parameters for a set of neurons, with each column referring
        to a neuron.

    bf_pref_tuning : np.ndarray, shape (n_parameters,)
        The locations of each basis function.

    bf_scale : float
        The spread of each basis function.

    n_stimuli : int
        The number of stimuli to tile the plane.

    limits : tuple
        The lower and upper bounds to sample points from the stimulus space.

    intercepts : np.ndarray, shape (n_neurons,), default None
        The intercepts of each tuning curve. If None, intercepts are assumed
        to be zero.

    Returns
    -------
    tuning_prefs : np.ndarray, shape (n_neurons,)
        The tuning preference for each neuron.
    """
    stimuli = np.linspace(limits[0], limits[1], n_stimuli)
    # calculate tuning curves
    tuning_curves = calculate_tuning_curves(
        B=B,
        bf_pref_tuning=bf_pref_tuning,
        bf_scale=bf_scale,
        n_stimuli=n_stimuli,
        limits=limits)
    # get tuning preference for each neuron by looking at tuning curve max
    tuning_prefs = stimuli[np.argmax(tuning_curves, axis=0)]
    return tuning_prefs
