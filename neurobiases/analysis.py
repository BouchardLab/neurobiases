import neuropacks as packs
import numpy as np

from neurobiases import TCSolver
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


def import_data(dataset, data_path, n_gaussians=7, transform='square_root'):
    """Interface with neuropacks to obtain design and response matrices.

    Parameters
    ----------
    dataset : string
        The neural dataset.
    data_path : string
        The path to the data.
    n_gaussians : int
        The number of gaussians in the basis function set (used for ecog pack).
    transform : string
        The transformation (used for pvc11 pack).

    Returns
    -------
    X, Y : np.ndarray
        The design and response matrices.
    class_labels : np.ndarray
        The class labels for the design matrix.
    pack : neuropack object
        The neuropack for the dataset.
    """
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


def coupling_fit(Y, fitter, Y_test=None):
    """Perform a coupling fit across a set of neurons.

    Parameters
    ----------
    Y : np.ndarray, shape (n_samples, n_units)
        The response matrix, or neural activity matrix.
    fitter : object
        The fitting object.
    Y_test : np.ndarray, shape (n_samples, n_features), optional
        Test response matrix for calculating generalization performance. If
        None, no test evaluation is performed.

    Returns
    -------
    intercepts : np.ndarray, shape (n_units,)
        The intercept for each fit.
    coupling_coefs : np.ndarray, shape (n_units, n_features)
        The tuning coefficient for each fit.
    train_scores : np.ndarray, shape (n_units,)
        The score of the model on training data.
    test_scores : np.ndarray, shape (n_units,)
        The score of the model on test data. Only returned if X_test and
        Y_test are provided.
    bics : np.ndarray, shape (n_units,)
        The BIC evaluated on the training data.
    """
    n_units = Y.shape[1]
    n_features = n_units - 1
    # Create storage arrays
    intercepts = np.zeros(n_units)
    coupling_coefs = np.zeros((n_units, n_features))
    train_scores = np.zeros(n_units)
    # Check if we need to provide test performance
    if Y_test is not None:
        test_set = True
        test_scores = np.zeros_like(train_scores)
    bics = np.zeros_like(train_scores)

    # Perform fits across units
    for unit in range(n_units):
        X_train = np.delete(Y, unit, axis=1)
        y_train = Y[:, unit]
        # Run tuning fit
        fitter.fit(X_train, y_train)
        # Store coefficients
        intercepts[unit] = fitter.intercept_
        coupling_coefs[unit] = fitter.coef_
        # Evaluate model
        train_scores[unit] = fitter.score(X_train, y_train)
        if test_set:
            X_test = np.delete(Y_test, unit, axis=1)
            y_test = Y_test[:, unit]
            test_scores[unit] = fitter.score(X_test, y_test)
        n_features = 1 + np.count_nonzero(fitter.coef_)
        bics[unit] = bic_linear(y_true=y_train,
                                y_pred=fitter.predict(X_train),
                                n_features=n_features)

    if test_set:
        return intercepts, coupling_coefs, train_scores, test_scores, bics
    else:
        return intercepts, coupling_coefs, train_scores, bics


def tuning_and_coupling_fit(
    X, Y, X_test=None, Y_test=None, solver='cd', c_tuning=0.,
    c_coupling='cv', initialization='random', max_iter=10000, tol=1e-4,
    refit=False, rng=None
):
    """Perform a tuning and coupling fit across a set of neurons.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The design matrix.
    Y : np.ndarray, shape (n_samples, n_units)
        The response matrix, or neural activity matrix.
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
    n_units = Y.shape[1]
    n_tuning_features = X.shape[1]
    n_coupling_features = n_units - 1
    # Create storage arrays
    intercepts = np.zeros(n_units)
    tuning_coefs = np.zeros((n_units, n_tuning_features))
    coupling_coefs = np.zeros((n_units, n_coupling_features))
    train_scores = np.zeros(n_units)
    # Check if we need to provide test performance
    if (X_test is not None) and (Y_test is not None):
        test_set = True
        test_scores = np.zeros_like(train_scores)
    bics = np.zeros_like(train_scores)

    # Perform fits across units
    for unit in range(n_units):
        fitter = TCSolver(
            X=X,
            Y=np.delete(Y, unit, axis=1),
            y=Y[:, unit],
            solver=solver,
            c_tuning=c_tuning,
            c_coupling=c_coupling,
            fit_intercept=True,
            initialization=initialization,
            max_iter=max_iter,
            tol=tol,
            rng=rng).fit(refit=refit)
        # Store coefficients
        intercepts[unit] = fitter.intercept
        tuning_coefs[unit] = fitter.b
        coupling_coefs[unit] = fitter.a
        # Evaluate model
        train_scores[unit] = fitter.mse()
        if test_set:
            test_scores[unit] = fitter.mse(X=X_test,
                                           Y=np.delete(Y_test, unit, axis=1),
                                           y=Y_test[:, unit])
        bics[unit] = fitter.bic()

    if test_set:
        return intercepts, coupling_coefs, tuning_coefs, train_scores, test_scores, bics
    else:
        return intercepts, coupling_coefs, tuning_coefs, train_scores, bics


def coupling_coefs_to_weight_matrix(coupling_coefs):
    """Converts a set of coupling coefficients to a weight matrix.

    Parameters
    ----------
    coupling_coefs : np.array
        The set of coupling coefficients.

    Returns
    -------
    weight_matrix : np.array
        A weight matrix, with self connections inputted.
    """
    n_units = coupling_coefs.shape[0]
    weight_matrix = np.zeros((n_units, n_units))

    for unit in range(n_units):
        weight_matrix[unit] = np.insert(coupling_coefs[unit], unit, 0)

    return weight_matrix
