import argparse
import h5py
import numpy as np

from neurobiases import analysis
from sklearn.model_selection import StratifiedKFold


def main(args):
    # Check random state
    if args.rng == -1:
        rng = None
    else:
        rng = np.random.default_rng(args.rng)

    model = args.model
    dataset = args.dataset
    n_folds = args.n_folds
    standardize = args.standardize
    verbose = args.verbose

    # Get dataset
    X, Y, class_labels, pack = analysis.import_data(
        dataset=dataset,
        data_path=args.data_path,
        n_gaussians=args.n_gaussians,
        transform=args.transform)

    # Clear out empty units
    Y = Y[:, np.argwhere(Y.sum(axis=0) != 0).ravel()]
    n_targets = Y.shape[1]

    # Create stratified folds
    skfolds = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=rng)
    train_folds = {}
    test_folds = {}

    # Create storage arrays
    intercepts = np.zeros((n_folds, n_targets))
    if model == 'tuning':
        tuning_coefs = np.zeros((n_folds, n_targets, n_features))
    elif model == 'coupling':
        coupling_coefs = np.zeros((n_folds, n_targets, n_targets - 1))
    elif model == 'tc':
        tuning_coefs = np.zeros((n_folds, n_targets, n_features))
        coupling_coefs = np.zeros((n_folds, n_targets, n_targets - 1))

    r2s_train = np.zeros((n_folds, n_targets))
    r2s_test = np.zeros_like(r2s_train)
    bics = np.zeros(r2s_train)

    # Outer loop: iterate over cross-validation folds
    for fold_idx, (train_idx, test_idx) in enumerate(
        skfolds.split(y=class_labels, X=class_labels)
    ):
        if verbose:
            print(f'Fold {fold_idx}', flush=True)
        # Save train and test indices
        train_folds[f'fold_{fold_idx}'] = train_idx
        test_folds[f'fold_{fold_idx}'] = test_idx
        # Extract train and test sets
        X_train = X[train_idx, :]
        Y_train = Y[train_idx, :]
        X_test = X[test_idx, :]
        Y_test = Y[test_idx, :]

        if model == 'tuning':
            (intercepts[fold_idx], tuning_coefs[fold_idx],
             train_scores[fold_idx], test_scores[fold_idx], bics[fold_idx]) = \
                analysis.tuning_fit(
                    X=X_train,
                    Y=Y_train,
                    fitter=fitter,
                    X_test=X_test,
                    Y_test=Y_test)