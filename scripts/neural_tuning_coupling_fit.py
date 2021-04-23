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
        rng = args.rng
    if args.rng == -1:
        fitter_rng = None
    else:
        fitter_rng = args.rng
    model = args.model
    dataset = args.dataset
    data_path = args.data_path
    write_path = args.write_path
    write_group = args.write_group
    n_folds = args.n_folds
    cv_verbose = args.cv_verbose
    # Get fitting object
    fitter = analysis.get_fitter(**vars(args))

    # Get dataset
    X, Y, class_labels, pack = analysis.import_data(
        dataset=dataset,
        data_path=data_path,
        n_gaussians=args.n_gaussians,
        transform=args.transform)
    # Clear out empty units
    Y = Y[:, np.argwhere(Y.sum(axis=0) != 0).ravel()]
    n_tuning_features = X.shape[1]
    n_coupling_features = Y.shape[1] - 1
    n_units = Y.shape[1]

    # Create stratified folds
    skfolds = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=rng)
    train_folds = {}
    test_folds = {}

    # Create storage arrays
    intercepts = np.zeros((n_folds, n_units))
    if model == 't':
        tuning_coefs = np.zeros((n_folds, n_units, n_tuning_features))
        modulations = np.zeros((n_folds, n_units))
        preferences = np.zeros((n_folds, n_units))
    elif model == 'c':
        coupling_coefs = np.zeros((n_folds, n_units, n_coupling_features))
    elif model == 'tc':
        tuning_coefs = np.zeros((n_folds, n_units, n_tuning_features))
        coupling_coefs = np.zeros((n_folds, n_units, n_coupling_features))
        modulations = np.zeros((n_folds, n_units))
        preferences = np.zeros((n_folds, n_units))

    scores_train = np.zeros((n_folds, n_units))
    scores_test = np.zeros_like(scores_train)
    bics = np.zeros_like(scores_train)

    # Outer loop: iterate over cross-validation folds
    for fold_idx, (train_idx, test_idx) in enumerate(
        skfolds.split(y=class_labels, X=class_labels)
    ):
        if cv_verbose:
            print(f'Fold {fold_idx}', flush=True)
        # Save train and test indices
        train_folds[f'fold_{fold_idx}'] = train_idx
        test_folds[f'fold_{fold_idx}'] = test_idx
        # Extract train and test sets
        X_train = X[train_idx, :]
        Y_train = Y[train_idx, :]
        X_test = X[test_idx, :]
        Y_test = Y[test_idx, :]

        # Perform the fit across all units
        if model == 't':
            (intercepts[fold_idx], tuning_coefs[fold_idx],
             scores_train[fold_idx], scores_test[fold_idx], bics[fold_idx]) = \
                analysis.tuning_fit(
                    X=X_train,
                    Y=Y_train,
                    fitter=fitter,
                    X_test=X_test,
                    Y_test=Y_test)
            # Calculate modulation and preferences
            modulations[fold_idx], preferences[fold_idx] = \
                pack.get_tuning_modulation_and_preference(tuning_coefs[fold_idx])

        elif model == 'c':
            (intercepts[fold_idx], coupling_coefs[fold_idx],
             scores_train[fold_idx], scores_test[fold_idx], bics[fold_idx]) = \
                 analysis.coupling_fit(
                     Y=Y_train,
                     fitter=fitter,
                     Y_test=Y_test)

        elif model == 'tc':
            (intercepts[fold_idx], coupling_coefs[fold_idx], tuning_coefs[fold_idx],
             scores_train[fold_idx], scores_test[fold_idx], bics[fold_idx]) = \
                 analysis.tuning_and_coupling_fit(
                     X=X_train,
                     Y=Y_train,
                     X_test=X_test,
                     Y_test=Y_test,
                     solver=args.solver,
                     c_tuning=args.c_tuning,
                     c_coupling=args.c_coupling,
                     initialization=args.initialization,
                     max_iter=args.max_iter,
                     tol=args.tol,
                     refit=args.refit,
                     rng=fitter_rng)
            # Calculate modulation and preferences
            modulations[fold_idx], preferences[fold_idx] = \
                pack.get_tuning_modulation_and_preference(tuning_coefs[fold_idx])

    # Write results to file
    with h5py.File(write_path, 'a') as results:
        if 'X' not in list(results):
            results['X'] = X
        if 'Y' not in list(results):
            results['Y'] = Y
        if 'class_labels' not in list(results):
            results['class_labels'] = class_labels

        group = results.create_group(write_group)
        group['intercepts'] = intercepts
        if model == 't':
            group['tuning_coefs'] = tuning_coefs
            group['tuning_modulations'] = modulations
            group['tuning_preferences'] = preferences
        elif model == 'c':
            group['coupling_coefs'] = coupling_coefs
        elif model == 'tc':
            group['coupling_coefs'] = coupling_coefs
            group['tuning_coefs'] = tuning_coefs
            group['tuning_modulations'] = modulations
            group['tuning_preferences'] = preferences
        group['scores_train'] = scores_train
        group['scores_test'] = scores_test
        group['bic'] = bics
        group.attrs['model'] = model
        group.attrs['dataset'] = dataset
        group.attrs['n_folds'] = n_folds
        group.attrs['fitter'] = args.fit
        # Write train and test indices
        train_group = group.create_group('train_idx')
        for key, val in train_folds.items():
            train_group[key] = val
        test_group = group.create_group('test_idx')
        for key, val in test_folds.items():
            test_group[key] = val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a tuning or coupling fit on data.')
    parser.add_argument('--model', choices=['t', 'c'])
    parser.add_argument('--dataset', choices=['pvc11', 'ecog'])
    parser.add_argument('--data_path')
    parser.add_argument('--write_path')
    parser.add_argument('--write_group')
    parser.add_argument('--fit', choices=['ols', 'lasso'])
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--rng', type=int, default=-1)
    # Data arguments
    parser.add_argument('--n_gaussians', type=int, default=7)
    parser.add_argument('--transform', default=None)
    # Other arguments
    parser.add_argument('--cv_verbose', action='store_true')
    args = parser.parse_args()

    main(args)
