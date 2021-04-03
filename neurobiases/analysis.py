import neuropacks as packs
import numpy as np

from sklearn.linear_model import LassoCV, LinearRegression


def bic_linear(y_true, y_pred, n_features):
    """Calculates the BIC of a linear model.

    Parameters
    ----------
    y_true : np.ndarray
        The true response variable values.
    y_pred : np.ndarray
        The predicted response variable values.
    n_features : int
        The number of features.

    Returns
    -------
    bic : float
        The Bayesian information criterion.
    """
    n_samples = y_true.size
    rss_mean = np.mean((y_true - y_pred)**2)
    bic = n_samples * np.log(rss_mean) + n_features * np.log(n_samples)
    return bic


def get_fitter(**kwargs):
    """Creates a scikit-learn fitting object using keyword arguments."""
    fit = kwargs.get('fit')
    # Fit an ordinary linear regression
    if fit == 'ols':
        fitter = LinearRegression(
            fit_intercept=kwargs.get('fit_intercept', True),
            normalize=kwargs.get('normalize', False))
    # Fit a cross-validated lasso regression
    elif fit == 'lasso':
        fitter = LassoCV(
            eps=kwargs.get('eps', 1e-3),
            n_alphas=kwargs.get('n_alphas', 100),
            fit_intercept=kwargs.get('fit_intercept', True),
            normalize=kwargs.get('normalize', False),
            max_iter=kwargs.get('max_iter', 5000),
            tol=kwargs.get('tol', 1e-4),
            cv=kwargs.get('cv', 5))
    return fitter


def get_modulations_and_preferences(form, coefs, **kwargs):
    """Calculates the tuning modulations and preferences for a set of tuning
    coefficients.

    Additional kwargs detail the tuning curve structure.

    Parameters
    ----------
    form : str
        The form of the tuning coefficients. Currently supports 'cosine2'
        (cosine basis functions with a period of pi), and 'gbf', or gaussian
        basis functions.
    coefs : np.ndarray
        The tuning coefficients, with first dimension spanning different
        tuning curves.

    Returns
    -------
    modulations : np.ndarray
        The tuning modulation for each tuning coefficient set.
    preferences : np.ndarray
        The tuning preference for each tuning coefficient set.
    """
    # Cosine basis functions
    if form == 'cosine2':
        c1 = coefs[..., 0]
        c2 = coefs[..., 1]
        # Convert preferences to degrees and ensure it's within [0, 360)
        preferences = np.arctan2(c2, c1) * (180/np.pi)
        preferences[preferences < 0] += 360
        preferences_rad = np.deg2rad(preferences)
        # Divide by 2 and take the modulus to ensure its lies within [0, 180)
        preferences = (preferences / 2) % 180
        # Calculate modulation
        modulations = 2 * (c2 - c1)/(np.sin(preferences_rad) - np.cos(preferences_rad))
    # Gaussian basis functions
    elif form == 'gbf':
        # Determine the tuning curve structure
        inputs = kwargs.get('inputs')
        means = kwargs.get('means')
        var = kwargs.get('var')
        # Calculate tuning curve
        norm = 1. / np.sqrt(2 * np.pi * var)
        tuning_curve = np.sum(coefs * norm * np.exp(
            -np.subtract.outer(inputs, means)**2 / (2 * var)
        ), axis=1)
        modulations = np.max(tuning_curve) - np.min(tuning_curve)
        preference_idx = np.argmax(tuning_curve).ravel()
        preferences = inputs[preference_idx]

    return modulations, preferences


def import_data(dataset, data_path, n_gaussians=7, transform='square_root'):
    # Electrocorticography, rat auditory cortex
    if dataset == 'ecog':
        # Create neuropack
        pack = packs.ECOG(data_path=data_path)
        # Get design matrix
        X = pack.get_design_matrix(form='bf', n_gaussians=n_gaussians)
        # Get response matrix
        Y = pack.get_response_matrix(
            bounds=(40, 60),
            band='HG',
            electrodes=None,
            transform=None)
        class_labels = pack.get_design_matrix(form='id')
    # Single-unit recordings, macaque primary visual cortex
    elif dataset == 'pvc11':
        # Create neuropack
        pack = packs.PVC11(data_path=data_path)
        # Get design matrix
        X = pack.get_design_matrix(form='cosine2')
        # get response matrix
        Y = pack.get_response_matrix(transform=transform)
        class_labels = pack.get_design_matrix(form='label')
    else:
        raise ValueError('Dataset not available.')

    return X, Y, class_labels, pack


def tuning_fit(X, Y, fitter, X_test=None, Y_test=None):
    """Perform a tuning fit across a set of neurons.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The design matrix.
    Y : np.ndarray, shape (n_samples, n_units)
        The response matrix, or neural activity matrix.
    fitter : object
        The fitting object.
    X_test : np.ndarray, shape (n_samples, n_features), optional
        Test design matrix for calculating generalization performance. If
        None, no test evaluation is performed.
    Y_test : np.ndarray, shape (n_samples, n_features), optional
        Test response matrix for calculating generalization performance. If
        None, no test evaluation is performed.

    Returns
    -------
    intercepts : np.ndarray, shape (n_units,)
        The intercept for each fit.
    tuning_coefs : np.ndarray, shape (n_units, n_features)
        The tuning coefficient for each fit.
    train_scores : np.ndarray, shape (n_units,)
        The score of the model on training data.
    test_scores : np.ndarray, shape (n_units,)
        The score of the model on test data. Only returned if X_test and
        Y_test are provided.
    bics : np.ndarray, shape (n_units,)
        The BIC evaluated on the training data.
    """
    n_features = X.shape[1]
    n_units = Y.shape[1]
    # Create storage arrays
    intercepts = np.zeros(n_units)
    tuning_coefs = np.zeros((n_units, n_features))
    train_scores = np.zeros(n_units)
    # Check if we need to provide test performance
    if (X_test is not None) and (Y_test is not None):
        test_set = True
        test_scores = np.zeros_like(train_scores)
    bics = np.zeros_like(train_scores)

    # Perform fits across units
    for unit in range(n_units):
        # Run tuning fit
        fitter.fit(X, Y[:, unit])
        # Store coefficients
        intercepts[unit] = fitter.intercept_
        tuning_coefs[unit] = fitter.coef_
        # Evaluate model
        train_scores[unit] = fitter.score(X, Y[:, unit])
        if test_set:
            test_scores[unit] = fitter.score(X_test, Y_test[:, unit])
        n_features = 1 + np.count_nonzero(fitter.coef_)
        bics[unit] = bic_linear(Y[:, unit], fitter.predict(X), n_features)

    if test_set:
        return intercepts, tuning_coefs, train_scores, test_scores, bics
    else:
        return intercepts, tuning_coefs, train_scores, bics
