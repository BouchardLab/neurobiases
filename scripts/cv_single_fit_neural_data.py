import argparse
import h5py
import numpy as np
import warnings

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from neurobiases import EMSolver
from neurobiases.solver_utils import cv_solver_oracle_selection
from sklearn.exceptions import ConvergenceWarning


def main(args):
    # Turn off convergence warning by default
    if not args.warn:
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Fit settings
    save_group = args.save_group
    selection = args.selection
    K_max = args.K_max
    Ks = np.arange(1, K_max + 1)
    criterion = args.criterion
    cv = args.cv
    solver = args.solver
    initialization = args.initialization
    max_iter = args.max_iter
    tol = args.tol
    verbose = args.verbose

    # Extract neural data
    file_path = args.file_path
    n_folds = None
    n_neurons = None
    a_masks = None
    if rank == 0:
        with h5py.File(file_path, 'r') as params:
            X = params['X'][:]
            Y_all = params['Y'][:]
            a_masks = params[selection]['coupling_coefs'][:]
            n_folds = len(params['train_idx'])
        n_neurons = Y_all.shape[1]
        M = X.shape[1]
        N = n_neurons - 1
        # Storage arrays
        intercepts = np.zeros((n_folds, n_neurons))
        coupling_coefs = np.zeros((n_folds, n_neurons, N))
        tuning_coefs = np.zeros((n_folds, n_neurons, M))
        Bs = np.zeros((n_folds, n_neurons, M, N))
        Psis = np.zeros((n_folds, n_neurons, N + 1))
        scores_train = np.zeros((n_folds, n_neurons))
        scores_test = np.zeros((n_folds, n_neurons))
        bics = np.zeros((n_folds, n_neurons))
        best_Ks = np.zeros((n_folds, n_neurons))
    n_folds = comm.bcast(n_folds, root=0)
    n_neurons = comm.bcast(n_neurons, root=0)
    a_masks = Bcast_from_root(a_masks, comm)
    a_masks = a_masks != 0

    for fold_idx in range(1):
        if verbose and rank == 0:
            print(f'Fold {fold_idx+1}')
        # Extract train and test indices
        X_train = None
        if rank == 0:
            with h5py.File(file_path, 'r') as params:
                train_idx = params[f'train_idx/fold_{fold_idx}'][:]
                test_idx = params[f'test_idx/fold_{fold_idx}'][:]
            X_train = X[train_idx]
            X_test = X[test_idx]
            Y_all_train = Y_all[train_idx]
            Y_all_test = Y_all[test_idx]
        X_train = Bcast_from_root(X_train, comm)
        # Iterate over neurons
        for neuron in range(1):
            if verbose and rank == 0:
                print(f'>>> Neuron {neuron}')
            Y_train = None
            y_train = None
            if rank == 0:
                # Create train/test sets for neural data
                Y_train = np.delete(Y_all_train, [neuron], axis=1)[:]
                Y_test = np.delete(Y_all_test, [neuron], axis=1)[:]
                y_train = Y_all_train[:, neuron][..., np.newaxis]
                y_test = Y_all_test[:, neuron][..., np.newaxis]
            Y_train = Bcast_from_root(Y_train, comm)
            y_train = Bcast_from_root(y_train, comm)
            # Extract fitted selection profile
            a_mask = a_masks[fold_idx, neuron]
            # Run single fit
            results = cv_solver_oracle_selection(
                X=X_train,
                Y=Y_train,
                y=y_train,
                Ks=Ks,
                cv=cv,
                a_mask=a_mask,
                solver=solver,
                initialization=initialization,
                numpy=False,
                max_iter=max_iter,
                tol=tol,
                Psi_transform='softplus',
                refit=False,
                comm=comm,
                fit_intercept=True,
                cv_verbose=args.cv_verbose,
                fitter_verbose=args.fitter_verbose,
                mstep_verbose=args.mstep_verbose,
                fitter_rng=9395062021)

            if rank == 0:
                # Choose best hyperparameter
                if criterion == 'aic':
                    avg_criterion = np.mean(results[2], axis=-1)
                elif criterion == 'bic':
                    avg_criterion = np.mean(results[3], axis=-1)
                elif criterion == 'score':
                    avg_criterion = -np.mean(results[1], axis=-1)
                else:
                    raise ValueError('Incorrect criterion specified.')
                K = Ks[np.argmin(avg_criterion)]
                best_Ks[fold_idx, neuron] = K
                # Refit to entire dataset
                print(f"Refitting with K = {K}.")
                fitter = EMSolver(
                    X=X_train,
                    Y=Y_train,
                    y=y_train,
                    K=K,
                    a_mask=a_mask,
                    b_mask=None,
                    Psi_transform='softplus',
                    c_tuning=0.,
                    c_coupling=0.,
                    solver=solver,
                    fit_intercept=True,
                    max_iter=max_iter,
                    tol=tol,
                    initialization=initialization,
                    rng=738279).fit_em(refit=False, index=True)
                intercepts[fold_idx, neuron] = fitter.y_intercept
                coupling_coefs[fold_idx, neuron] = fitter.a.ravel()
                tuning_coefs[fold_idx, neuron] = fitter.b.ravel()
                Bs[fold_idx, neuron] = fitter.B
                Psis[fold_idx, neuron] = fitter.Psi_tr_to_Psi()
                scores_train[fold_idx, neuron] = fitter.marginal_log_likelihood()
                scores_test[fold_idx, neuron] = fitter.marginal_log_likelihood(
                    X=X_test, Y=Y_test, y=y_test
                )
                bics[fold_idx, neuron] = fitter.bic()

    if rank == 0:
        with h5py.File(file_path, 'a') as params:
            if save_group in params:
                save = params[save_group]
                save['intercepts'][:] = intercepts
                save['coupling_coefs'][:] = coupling_coefs
                save['tuning_coefs'][:] = tuning_coefs
                save['Bs'][:] = Bs
                save['Psis'][:] = Psis
                save['scores_train'][:] = scores_train
                save['scores_test'][:] = scores_test
                save['bics'][:] = bics
                save['best_Ks'][:] = best_Ks
            else:
                save = params.create_group(save_group)
                save['intercepts'] = intercepts
                save['coupling_coefs'] = coupling_coefs
                save['tuning_coefs'] = tuning_coefs
                save['Bs'] = Bs
                save['Psis'] = Psis
                save['scores_train'] = scores_train
                save['scores_test'] = scores_test
                save['bics'] = bics
                save['best_Ks'] = best_Ks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--save_group', type=str)
    parser.add_argument('--selection', type=str)
    parser.add_argument('--K_max', type=int, default=10)
    parser.add_argument('--criterion', type=str, default='bic')
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--solver', type=str, default='scipy_lbfgs')
    parser.add_argument('--initialization', type=str, default='fits')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-5)

    # Verbosity flags
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--fitter_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    parser.add_argument('--warn', action='store_true')
    args = parser.parse_args()

    main(args)
