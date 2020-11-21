import numpy as np

from scipy.special import erf


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


def inv_softplus(x, cap=100):
    """Calculates the inverse softplus function log(exp(x) - 1).

    Parameters
    ----------
    x : np.ndarray
        Input array.
    cap: float
        A large enough number such that for each value v > cap, we approximate
        inv_softmax(v) = v. Helps avoid overflow errors with exp(v).

    Returns
    -------
    inv_softplus : np.ndarray
        The inverse softplus of the input.
    """
    # Note: there are still cases with {underflow in exp & invalid in log}
    # when input value v = x[i] is too small; in orders like v < (10 ** -15)
    y = np.copy(x)
    mask = (x < cap)
    y[mask] = np.log(np.exp(x[mask]) - 1)
    return y


def softplus(x, cap=100):
    """Calculates the softplus function log(1 + exp(x)).

    Parameters
    ----------
    x : np.ndarray
        Input array.
    cap: float
        A large enough number such that for each value v > cap, we approximate
        softmax(v) = v. Helps avoid overflow errors with exp(v).

    Returns
    -------
    y : np.ndarray
        The softplus of the input.
    """
    y = np.copy(x)
    mask = (x < cap)
    y[mask] = np.log(np.exp(x[mask]) + 1)
    return y


def selection_accuracy(mask1, mask2):
    """Calculates the selection accuracy (set overlap).

    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Selection profiles as numpy arrays of coefficients or masks.

    Returns
    -------
    selection_accuracy : float
        The selection accuracy.
    """
    idx1 = np.argwhere(mask1 != 0).ravel()
    idx2 = np.argwhere(mask2 != 0).ravel()
    difference = np.setdiff1d(idx1, idx2).size + np.setdiff1d(idx2, idx1).size
    selection_accuracy = 1 - difference / (idx1.size + idx2.size)
    return selection_accuracy


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


def process_tc_results(results):
    # Get true and estimated parameters
    a_trues = results['a_trues']
    b_trues = results['b_trues']
    a_hats = results['a_hats']
    b_hats = results['b_hats']

    # Get number of non-zero parameters
    n_nz_a = np.unique(np.count_nonzero(a_trues, axis=-1)).item()
    n_nz_b = np.unique(np.count_nonzero(b_trues, axis=-1)).item()
    n_hyparams, n_models, n_datasets = a_hats.shape[:-1]

    # Calculate biases
    a_bias = np.zeros((n_hyparams, n_models, n_datasets, n_nz_a))
    b_bias = np.zeros((n_hyparams, n_models, n_datasets, n_nz_b))
    a_bias_norm = np.zeros_like(a_bias)
    b_bias_norm = np.zeros_like(b_bias)

    for hyp_idx in range(n_hyparams):
        for model_idx in range(n_models):
            current_a = a_trues[hyp_idx, model_idx]
            current_b = b_trues[hyp_idx, model_idx]
            a_true_nz = current_a[current_a != 0]
            b_true_nz = current_b[current_b != 0]
            for dataset_idx in range(n_datasets):
                # Calculate biases per model and hyperparameters
                a_bias_temp = a_hats[hyp_idx, model_idx, dataset_idx][current_a != 0] - a_true_nz
                b_bias_temp = b_hats[hyp_idx, model_idx, dataset_idx][current_b != 0] - b_true_nz
                a_bias[hyp_idx, model_idx, dataset_idx] = a_bias_temp
                b_bias[hyp_idx, model_idx, dataset_idx] = b_bias_temp
                # Calculate normalized biases
                a_bias_norm[hyp_idx, model_idx, dataset_idx] = a_bias_temp / np.abs(a_true_nz)
                b_bias_norm[hyp_idx, model_idx, dataset_idx] = b_bias_temp / np.abs(b_true_nz)

    return a_trues, b_trues, a_bias, b_bias, a_bias_norm, b_bias_norm


def process_tc_double_results(results):
    # Get true and estimated parameters
    a_trues = results['a_trues']
    b_trues = results['b_trues']
    a_hats = results['a_hats']
    b_hats = results['b_hats']

    # Get number of non-zero parameters
    n_nz_a = np.unique(np.count_nonzero(a_trues, axis=-1)).item()
    n_nz_b = np.unique(np.count_nonzero(b_trues, axis=-1)).item()
    n_hyparams1, n_hyparams2, n_models, n_datasets = a_hats.shape[:-1]

    # Calculate biases
    a_bias = np.zeros((n_hyparams1, n_hyparams2, n_models, n_datasets, n_nz_a))
    b_bias = np.zeros((n_hyparams1, n_hyparams2, n_models, n_datasets, n_nz_b))
    a_bias_norm = np.zeros_like(a_bias)
    b_bias_norm = np.zeros_like(b_bias)

    for hyp1_idx in range(n_hyparams1):
        for hyp2_idx in range(n_hyparams2):
            for model_idx in range(n_models):
                current_a = a_trues[hyp1_idx, hyp2_idx, model_idx]
                current_b = b_trues[hyp1_idx, hyp2_idx, model_idx]
                a_true_nz = current_a[current_a != 0]
                b_true_nz = current_b[current_b != 0]
                for dataset_idx in range(n_datasets):
                    # Calculate biases per model and hyperparameters
                    a_bias_temp = \
                        a_hats[hyp1_idx, hyp2_idx, model_idx, dataset_idx][current_a != 0] \
                        - a_true_nz
                    b_bias_temp = \
                        b_hats[hyp1_idx, hyp2_idx, model_idx, dataset_idx][current_b != 0] \
                        - b_true_nz
                    a_bias[hyp1_idx, hyp2_idx, model_idx, dataset_idx] = a_bias_temp
                    b_bias[hyp1_idx, hyp2_idx, model_idx, dataset_idx] = b_bias_temp
                    # Calculate normalized biases
                    a_bias_norm[hyp1_idx, hyp2_idx, model_idx, dataset_idx] = \
                        a_bias_temp / np.abs(a_true_nz)
                    b_bias_norm[hyp1_idx, hyp2_idx, model_idx, dataset_idx] = \
                        b_bias_temp / np.abs(b_true_nz)

    return a_trues, b_trues, a_bias, b_bias, a_bias_norm, b_bias_norm


def check_identifiability_conditions(Psi_nt, L_nt, B, a_mask, b_mask):
    """Checks the conditions for clamping identifiability.

    Parameters
    ----------
    Psi_nt : np.ndarray, shape (N,)
        The non-target private variances.
    L_nt : np.ndarray, shape (K, N)
        The non-target latent factors.
    B : np.ndarray, shape (M, N)
        The non-target tuning parameters.
    a_mask : np.ndarray
        The selection profile for the coupling parameters.
    b_mask : np.ndarray
        The selection profile for the tuning parameters.

    Returns
    -------
    check : bool
        Whether the check passed.
    """
    # Get dimensionalities
    K = L_nt.shape[0]
    N = a_mask.size
    M = b_mask.size
    # Number of zero parameters
    sparsity = N + M - a_mask.sum() - b_mask.sum()

    # Dimensionality condition
    if K > sparsity:
        return False

    # Rank condition
    P = np.linalg.solve(np.diag(Psi_nt) + L_nt.T @ L_nt, L_nt.T)
    P_sub = P[a_mask]
    Q = np.dot(B, P)
    Q_sub = Q[b_mask]
    R = np.concatenate((P_sub, Q_sub), axis=0)
    rank = np.linalg.matrix_rank(R)
    if rank < K:
        return False
    return True
