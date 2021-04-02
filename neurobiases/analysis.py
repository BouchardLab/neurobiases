import neuropacks as packs
import numpy as np


def bic_linear(y_true, y_pred, n_features):
    n_samples = y_true.size
    rss_mean = np.mean((y_true - y_pred)**2)
    bic = n_samples * np.log(rss_mean) + n_features * np.log(n_samples)
    return bic


def get_modulations_and_preferences(form, coefs, **kwargs):
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
    elif form == 'gbf':
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
    n_features = X.shape[1]
    n_neurons = Y.shape[1]
    # Create storage arrays
    intercepts = np.zeros(n_neurons)
    tuning_coefs = np.zeros((n_neurons, n_features))
    train_scores = np.zeros(n_neurons)
    if (X_test is not None) and (Y_test is not None):
        test_set = True
        test_scores = np.zeros_like(train_scores)
    bics = np.zeros_like(train_scores)

    for neuron in range(n_neurons):
        # Run tuning fit
        fitter.fit(X, Y[:, neuron])
        # Store coefficients
        intercepts[neuron] = fitter.intercept_
        tuning_coefs[neuron] = fitter.coef_
        # Evaluate model
        train_scores[neuron] = fitter.score(X, Y[:, neuron])
        if test_set:
            test_scores[neuron] = fitter.score(X_test, Y_test[:, neuron])
        bics[neuron] = bic_linear(Y[:, neuron], fitter.predict(X))

    if test_set:
        return intercepts, tuning_coefs, train_scores, test_scores, bics
    else:
        return intercepts, tuning_coefs, train_scores, bics
