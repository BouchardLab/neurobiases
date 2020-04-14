import numpy as np

from scipy.special import erf


def read_attribute_dict(attributes):
    copy = {}
    for key, val in attributes.items():
        if val == '':
            copy[key] = None
        else:
            copy[key] = val
    return copy


def sigmoid(x, phase=0, b=1):
    return 1./(1 + np.exp(-b * (x - phase)))


def bf_mean(center, scale, limits=(0, 1)):
    """Calculates the mean of a Gaussian basis function feature where the
    input behaves as a uniform distribution.

    Parameters
    ----------
    center : float or np.ndarray
        The center of the Gaussian basis function.

    scale : float
        The variance of the Gaussian basis function.

    limits : tuple
        The lower and upper bound of the incoming stimulus.

    Returns
    -------
    mean : float
        The mean of the basis function over the provided limits.
    """
    # constants
    a, b = limits
    norm = np.sqrt(2 * scale)
    erf_const = 0.5 * np.sqrt(np.pi)
    # calculate mean
    mean = erf_const * norm / (b - a) * (
        erf((b - center) / norm) - erf((a - center) / norm)
    )
    return mean


def bf_variance(center, scale, limits=(0, 1)):
    """Calculates the variance of a Gaussian basis function feature where the
    input behaves as a uniform distribution.

    Parameters
    ----------
    center : float or np.ndarray
        The center of the Gaussian basis function.

    scale : float
        The variance of the Gaussian basis function.

    limits : tuple
        The lower and upper bound of the incoming stimulus.

    Returns
    -------
    mean : float
        The mean of the basis function over the provided limits.
    """
    # calculate expected value of squared output and mean
    sq_mean = bf_mean(center, scale / 2., limits)
    mean = bf_mean(center, scale, limits)
    # calculate variance
    variance = sq_mean - mean**2
    return variance


def bf_pairwise_cov(centers, scale, limits=(0, 1)):
    """Calculates covariance between two Gaussian basis functions given an
    input stimulus following a uniform distribution, assuming both basis
    functions have the same scale.

    Parameters
    ----------
    centers : np.ndarray, shape (2,)
        The means of the basis functions.

    scale : float
        The variances of the basis functions.

    limits : tuple
        The lower and upper bound of the incoming stimulus.

    Returns
    -------
    covariance : float
        The covariance between the two basis functions across the stimulus
        distribution.
    """
    # calculate means of bf
    bf_means = bf_mean(centers, scale, limits=limits)
    # calculated expected value of the product
    constant = np.exp(-1./scale * (0.25 * np.sum(centers**2) - 0.5 * np.prod(centers)))
    product_mean = constant * bf_mean(np.mean(centers), scale / 2)
    # calculate covariance
    covariance = product_mean - np.prod(bf_means)
    return covariance


def bf_cov(centers, scale, limits=(0, 1)):
    """Calculates covariance between a set of Gaussian basis functions given an
    input stimulus following a uniform distribution, assuming all basis
    functions have the same scale.

    Parameters
    ----------
    centers : np.ndarray
        The means of the basis functions.

    scale : float
        The variances of the basis functions.

    limits : tuple
        The lower and upper bound of the incoming stimulus.

    Returns
    -------
    covariance : float
        The covariance matrix of the basis functions.
    """
    bf_means = bf_mean(centers, scale, limits=limits)
    constant = np.exp(-1. / scale * (
        0.25 * np.add.outer(centers**2, centers**2) - 0.5 * np.outer(centers, centers)
    ))
    product_mean = bf_mean(0.5 * np.add.outer(centers, centers),
                           scale / 2.,
                           limits=(0, 1))
    cov = constant * product_mean - np.outer(bf_means, bf_means)
    return cov


def bf_sum_var(weights, centers, scale, limits=(0, 1)):
    """Calculates variance of a linear combination of basis functions.

    Parameters
    ----------
    weights : np.ndarray, shape (n_bf,)
        The weights of each basis function.

    centers : np.ndarray, shape (n_bf,)
        The means of the basis functions.

    scale : float
        The variances of the basis functions.

    limits : tuple
        The lower and upper bound of the incoming stimulus.

    Returns
    -------
    covariance : float
        The covariance matrix of the basis functions.
    """
    cov = bf_cov(centers, scale, limits)
    var = np.dot(weights, np.dot(cov, weights))
    return var


def calculate_tuning_features(stimuli, bf_center, bf_scale):
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
        -0.5 * np.subtract.outer(stimuli, bf_center)**2 / bf_scale
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
    return stimuli, tuning_curves


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
