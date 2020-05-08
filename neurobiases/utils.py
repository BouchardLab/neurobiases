import numpy as np

from scipy.special import erf


def marginal_log_likelihood_linear_tm(
    X, Y, y, a, b, B, L, Psi, a_mask=None, b_mask=None, B_mask=None
):
    """Calculates the marginal log-likelihood of a parameter set under data
    generated from the linear triangular model.

    Parameters
    ----------
    X : np.ndarray, shape (D, M)
        Design matrix for tuning features.
    Y : np.ndarray, shape (D, N)
        Design matrix for coupling features.
    y : np.ndarray, shape (D, 1)
        Neural response vector.
    a : np.ndarray, shape (N,)
        The coupling parameters.
    b : np.ndarray, shape (N,)
        The target tuning parameters.
    B : np.ndarray, shape (M, N)
        The non-target tuning parameters.
    L : np.ndarray, shape (K, N + 1)
        The latent factors
    Psi : np.ndarray, shape (N + 1,)
        The private variances.
    a_mask : np.ndarray, shape (N,)
        Mask for coupling features.
    b_mask : nd-array, shape (M,)
        Mask for tuning features.
    B_mask : nd-array, shape (N, M)
        Mask for non-target neuron tuning features.

    Returns
    -------
    ll : float
        The marginal log-likelihood according to the linear triangular model.
    """
    D, M = X.shape
    N = Y.shape[1]

    # dimensionality checks
    if a.ndim == 2:
        a = a.ravel()
    if b.ndim == 2:
        b = b.ravel()
    # check masks
    if a_mask is None:
        a_mask = np.ones(N)
    elif a_mask.ndim == 2:
        a_mask = a_mask.ravel()

    if b_mask is None:
        b_mask = np.ones(M)
    elif b_mask.ndim == 2:
        b_mask = np.ones(M)

    # apply masks
    a = a * a_mask
    b = b * b_mask

    # split up into target and non-target components
    L_nt, l_t = np.split(L, [N], axis=1)
    l_t = l_t.ravel()
    Psi_nt, Psi_t = np.split(Psi, [N])
    Psi_t = Psi_t.item()

    # mean and covariance matrices of the gaussian expression
    mu = np.zeros((D, N + 1))
    sigma = np.zeros((N + 1, N + 1))

    # calculate mean of marginal
    mu[:, 1:] = X @ B
    mu[:, 0] = X @ (b + B @ a)

    # combine data matrices
    Y_all = np.concatenate((y, Y), axis=1)

    # useful terms to store for later
    coupled_L = l_t + L_nt @ a
    cross_coupling = Psi_nt * a + L_nt.T @ coupled_L

    # fill covariance matrix
    sigma[0, 0] = Psi_t + Psi_nt @ a**2 + coupled_L @ coupled_L
    sigma[1:, 0] = cross_coupling
    sigma[0, 1:] = cross_coupling
    sigma[1:, 1:] = np.diag(Psi_nt) + L_nt.T @ L_nt

    # calculate log-likelihood
    residual = Y_all - mu
    ll = -D / 2. * np.linalg.slogdet(sigma)[1] \
        + -0.5 * np.sum(residual.T * np.linalg.solve(sigma, residual.T))

    return ll


def marginal_log_likelihood_linear_tm_wrapper(X, Y, y, tm):
    """Calculates the marginal log-likelihood of the parameters in a
    TriangularModel instance under data generated from the linear triangular
    model.

    Parameters
    ----------
    X : np.ndarray, shape (D, M)
        Design matrix for tuning features.
    Y : np.ndarray, shape (D, N)
        Design matrix for coupling features.
    y : np.ndarray, shape (D, 1)
        Neural response vector.
    tm : TriangularModel instance
        A TriangularModel instance.

    Returns
    -------
    ll : float
        The marginal log-likelihood according to the linear triangular model.
    """
    ll = marginal_log_likelihood_linear_tm(
        X=X, Y=Y, y=y, a=tm.a, b=tm.b, B=tm.B, L=tm.L, Psi=tm.Psi
    )
    return ll


def copy_attribute_dict(attributes):
    """Reads in and copies an attribute dictionary from an H5 file.

    Parameters
    ----------
    attributes : dict
        The attribute dictionary.

    Returns
    -------
    copy : dict
        A copy of attributes, with empty values replaced with None.
    """
    copy = {}
    for key, val in attributes.items():
        # replace empty values with None
        if val == '':
            copy[key] = None
        else:
            copy[key] = val
    return copy


def sigmoid(x, phase=0, b=1):
    """Calculates the sigmoid of input data.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    phase : float
        The center of the sigmoid.
    b : float
        The growth rate. Larger values implies a steeper sigmoid function.

    Returns
    -------
    sigmoid : np.ndarray
    """
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
    var : float
        The variance of the sum.
    """
    cov = bf_cov(centers, scale, limits)
    var = np.dot(weights, np.dot(cov, weights))
    return var


def calculate_tuning_features(stimuli, bf_centers, bf_scale):
    """Get basis function tuning features given a set of input stimuli.

    Parameters
    ----------
    stimuli : np.ndarray, shape (n_samples,)
        The input stimuli.
    bf_centers : np.ndarray, shape (n_parameters,)
        The locations of each basis function.
    bf_scale : float
        The spread of each basis function.

    Returns
    -------
    tuning_features : np.ndarray, shape (n_samples, n_features)
        The tuning features for each stimulus.
    """
    tuning_features = np.exp(
        -0.5 * np.subtract.outer(stimuli, bf_centers)**2 / bf_scale
    )
    return tuning_features


def calculate_tuning_curves(
    B, bf_centers, bf_scale, n_stimuli=10000, limits=(0, 1), intercepts=None
):
    """Calculates the tuning curves using Gaussian basis functions for a provided
    set of tuning parameters.

    Parameters
    ----------
    B : np.ndarray, shape (n_parameters, n_neurons)
        The tuning parameters for a set of neurons, with each column referring
        to a neuron.
    bf_centers : np.ndarray, shape (n_parameters,)
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
        -0.5 * np.subtract.outer(stimuli, bf_centers)**2 / bf_scale
    )
    # calculate responses for each neuron
    tuning_curves = intercepts + np.dot(bf_responses, B)
    return stimuli, tuning_curves


def calculate_pref_tuning(
    B, bf_centers, bf_scale, n_stimuli=10000, limits=(0, 1)
):
    """Calculate the preferred tuning using tuning parameters.

    Parameters
    ----------
    B : np.ndarray, shape (n_parameters, n_neurons)
        The tuning parameters for a set of neurons, with each column referring
        to a neuron.
    bf_centers : np.ndarray, shape (n_parameters,)
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
    _, tuning_curves = calculate_tuning_curves(
        B=B,
        bf_centers=bf_centers,
        bf_scale=bf_scale,
        n_stimuli=n_stimuli,
        limits=limits)
    # get tuning preference for each neuron by looking at tuning curve max
    tuning_prefs = stimuli[np.argmax(tuning_curves, axis=0)]
    return tuning_prefs


def noise_correlation_matrix(
    tuning_prefs, corr_max, corr_min=0, L=1, circular_stim=None
):
    """Create noise correlation matrix according to tuning preference.
    Correlations fall off with exponential decay.

    Parameters
    ----------
    tuning_prefs : np.ndarray, shape (n_neurons,)
        The tuning preferences for each neuron.
    corr_max : float
        The maximum possible noise correlation.
    corr_min : float
        The minimum possible noise correlation.
    L : float
        The exponential decay constant. Lower values imply that correlations
        fall off more steeply.
    circular_stim : None or float
        If None, stimulus is treated normally. Otherwise, stimulus is treated
        circularly, with this value taking on the maximum value of the stimulus.

    Returns
    -------
    corrs : np.ndarray, shape (n_neurons, n_neurons)
        The correlation matrix.
    """
    # calculate tuning differences
    diffs = np.abs(np.subtract.outer(tuning_prefs, tuning_prefs))
    # if stimulus is circular, modify differences
    if circular_stim is not None:
        thres = circular_stim / 2
        diffs[diffs > thres] = circular_stim - diffs[diffs > thres]
    # calculate correlation values
    corrs = corr_min + corr_max * np.exp(-diffs / L)
    # replace diagonal with ones
    np.fill_diagonal(corrs, 1)
    return corrs


def cov2corr(cov):
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : np.ndarray, shape (n_units, n_units)
        The covariance matrix.

    Returns
    -------
    corr : np.ndarray, shape (n_units, n_units)
        The correlation matrix.
    """
    stdevs = np.sqrt(np.diag(cov))
    outer = np.outer(stdevs, stdevs)
    corr = cov / outer
    return corr


def corr2cov(corr, var):
    """Converts a correlation matrix to a covariance matrix, given a set of
    variances.

    Parameters
    ----------
    corr : np.ndarray, shape (n_units, n_units)
        The correlation matrix.
    var : np.ndarray, shape (n_units,)
        A vector of variances for the units in the correlation matrix.

    Returns
    -------
    cov : np.ndarray, shape (n_units, n_units)
        The covariance matrix.
    """
    stdevs = np.sqrt(var)
    outer = np.outer(stdevs, stdevs)
    cov = corr * outer
    return cov


def symmetric_low_rank_approx(X, k):
    """Calculates a low-rank approximation of a symmetric matrix using its
    eigenvalue decomposition.

    Parameters
    ----------
    X : np.ndarray
        Symmetric matrix.
    k : int
        The rank.

    Returns
    -------
    L : np.ndarray, shape (n_units, k)
        The low-rank approximation, i.e. the matrix such that LL^T is a rank-k
        approximation to X.
    """
    # eigenvalue decomposition
    u, v = np.linalg.eigh(X)
    # truncate eigenmodes
    u_trunc = u[-k:]
    v_trunc = v[:, -k:]
    # form low-rank basis
    L = v_trunc @ np.diag(np.sqrt(u_trunc))
    return L
