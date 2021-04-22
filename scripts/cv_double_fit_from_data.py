import argparse
import h5py
import numpy as np
import os
import time
import warnings

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from neurobiases import EMSolver
from neurobiases.solver_utils import cv_sparse_solver_single
from sklearn.exceptions import ConvergenceWarning


def main(args):
    # MPI communicator
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    if rank == 0:
        t0 = time.time()

    # Turn off convergence warning by default
    if not args.warn:
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    verbose = args.verbose
    # Job settings
    file_path = args.file_path
    params_group = args.params_group
    store_path = args.store_path
    model_fit = args.model_fit
    unit = args.unit
    fold = args.fold

    # Open up experiment settings
    X_train = None
    Y_train = None
    y_train = None
    X_test = None
    Y_test = None
    y_test = None
    if rank == 0:
        with h5py.File(file_path, 'r') as data:
            # Get the data
            X = data['X'][:]
            Y = data['Y'][:]
            # Get the fold indices
            train_idx = data[f'train_idx/fold_{fold}'][:]
            test_idx = data[f'train_idx/fold_{fold}'][:]
            X_train = X[train_idx]
            Y_train = np.delete(Y[train_idx], unit, axis=1)
            y_train = Y[train_idx][:, unit]
            X_test = X[test_idx]
            Y_test = np.delete(Y[test_idx], unit, axis=1)
            y_test = Y[test_idx][:, unit]

            # Get params
            params = data[params_group]
            # Random seed
            fitter_rng = params.attrs['fitter_rng']
            # Training hyperparameters
            Ks = params['Ks'][:]
            coupling_lambdas = params['coupling_lambdas'][:]
            n_coupling_lambdas = coupling_lambdas.size
            tuning_lambdas = params['tuning_lambdas'][:]
            n_tuning_lambdas = tuning_lambdas.size
            # Training settings
            cv = params.attrs['cv']
            criterion = params.attrs['criterion']
            fine_sweep_frac = params.attrs['fine_sweep_frac']
            solver = params.attrs['solver']
            initialization = params.attrs['initialization']
            max_iter = params.attrs['max_iter']
            tol = params.attrs['tol']
    # Broadcast data
    X_train = Bcast_from_root(X_train, comm)
    Y_train = Bcast_from_root(Y_train, comm)
    y_train = Bcast_from_root(y_train, comm)
    X_test = Bcast_from_root(X_test, comm)
    Y_test = Bcast_from_root(Y_test, comm)
    y_test = Bcast_from_root(y_test, comm)

    # Run coarse sweep CV
    coarse_sweep_results = \
        cv_sparse_solver_single(
            method=model_fit,
            X=X_train,
            Y=Y_train,
            y=y_train,
            coupling_lambdas=coupling_lambdas,
            tuning_lambdas=tuning_lambdas,
            Ks=Ks,
            cv=cv,
            solver=solver,
            initialization=initialization,
            max_iter=max_iter,
            tol=tol,
            refit=True,
            comm=comm,
            cv_verbose=args.cv_verbose,
            fitter_verbose=args.fitter_verbose,
            mstep_verbose=args.mstep_verbose,
            fit_intercept=True,
            fitter_rng=fitter_rng)

    if rank == 0:
        if model_fit == 'em':
            scores, aics, bics, a_est, b_est, B_est, Psi_est, L_est = coarse_sweep_results
        elif model_fit == 'tc':
            scores, aics, bics, a_est, b_est = coarse_sweep_results
        # Identify best hyperparameter set according to criterion
        if criterion == 'aic':
            median_criterion = np.median(aics, axis=-1)
        elif criterion == 'bic':
            median_criterion = np.median(bics, axis=-1)
        elif criterion == 'score':
            median_criterion = -np.median(scores, axis=-1)
        else:
            raise ValueError('Incorrect criterion specified.')
        best_hyps = np.unravel_index(np.argmin(median_criterion), median_criterion.shape)
        best_c_coupling = coupling_lambdas[best_hyps[0]]
        best_c_tuning = tuning_lambdas[best_hyps[1]]
        if model_fit == 'em':
            Ks = np.array([Ks[best_hyps[2]]])
        # Create new hyperparameter set
        coupling_lambda_lower = fine_sweep_frac * best_c_coupling
        coupling_lambda_upper = (1. / fine_sweep_frac) * best_c_coupling
        coupling_lambdas = np.linspace(coupling_lambda_lower,
                                       coupling_lambda_upper,
                                       num=n_coupling_lambdas)
        tuning_lambda_lower = fine_sweep_frac * best_c_tuning
        tuning_lambda_upper = (1. / fine_sweep_frac) * best_c_tuning
        tuning_lambdas = np.linspace(tuning_lambda_lower,
                                     tuning_lambda_upper,
                                     num=n_tuning_lambdas)

        if verbose:
            print(f'First sweep complete. Best coupling lambda: {best_c_coupling}'
                  f' and best tuning lambda: {best_c_tuning}')
    # Broadcast new lambdas out
    coupling_lambdas = Bcast_from_root(coupling_lambdas, comm)
    tuning_lambdas = Bcast_from_root(tuning_lambdas, comm)
    if model_fit == 'em':
        Ks = Bcast_from_root(Ks, comm)
    # Verbosity update
    if rank == 0:
        t1 = time.time()

    # Run broad sweep CV
    fine_sweep_results = \
        cv_sparse_solver_single(
            method=model_fit,
            X=X_train,
            Y=Y_train,
            y=y_train,
            coupling_lambdas=coupling_lambdas,
            tuning_lambdas=tuning_lambdas,
            Ks=Ks,
            cv=cv,
            solver=solver,
            initialization=initialization,
            max_iter=max_iter,
            tol=tol,
            refit=True,
            comm=comm,
            cv_verbose=args.cv_verbose,
            fitter_verbose=args.fitter_verbose,
            mstep_verbose=args.mstep_verbose,
            fitter_rng=fitter_rng)

    if rank == 0:
        if model_fit == 'em':
            scores, aics, bics, a_est, b_est, B_est, Psi_est, L_est = fine_sweep_results
        elif model_fit == 'tc':
            scores, aics, bics, a_est, b_est = fine_sweep_results
        # Get best overall fit
        if criterion == 'aic':
            median_criterion = np.median(aics, axis=-1)
        elif criterion == 'bic':
            median_criterion = np.median(bics, axis=-1)
        elif criterion == 'score':
            median_criterion = -np.median(scores, axis=-1)
        else:
            raise ValueError('Incorrect criterion specified.')
        best_hyps = np.unravel_index(np.argmin(median_criterion), median_criterion.shape)
        best_c_coupling = coupling_lambdas[best_hyps[0]]
        best_c_tuning = tuning_lambdas[best_hyps[1]]
        if model_fit == 'em':
            best_K = Ks[best_hyps[2]]

        final_solver = EMSolver(
            X=X_train,
            Y=Y_train,
            y=y_train,
            K=best_K,
            c_tuning=best_c_tuning,
            c_coupling=best_c_coupling,
            solver='ow_lbfgs',
            fit_intercept=True,
            max_iter=args.max_iter,
            tol=args.tol,
            penalize_B=False,
            initialization=args.initialization,
            rng=fitter_rng).fit_em(refit=True, numpy=True)

        # Save results
        with h5py.File(store_path, 'a') as results:
            results.attrs['score'] = final_solver.marginal_log_likelihood(
                X=X_test, Y=Y_test, y=y_test
            )
            results['bic'] = final_solver.bic()
            results['aic'] = final_solver.aic()
            # Estimated parameters
            results['a_est'] = final_solver.a.ravel()
            results['a_est_intercept'] = final_solver.a_intercept
            results['b_est'] = final_solver.b.ravel()
            results['b_est_intercept'] = final_solver.b_intercept
            results['B_est'] = final_solver.B
            results['B_est_intercept'] = final_solver.B_intercept
            results['Psi_est'] = final_solver.Psi_tr_to_Psi(final_solver.Psi_tr)
            results['L_est'] = final_solver.L
            # CV details
            results.attrs['best_coupling_lambda'] = best_c_coupling
            results.attrs['best_tuning_lambda'] = best_c_tuning

        t2 = time.time()
        print(
            "---------------------------------------------------------------\n"
            "Job complete: Performed a single CV coarse-fine sweep.\n"
            f"File: {os.path.basename(file_path)}\n"
            f"Number of processes: {size}\n"
            f"Fine sweep centered on {best_c_coupling:0.2E} (coupling) "
            f"and {best_c_tuning:0.2E} (tuning).\n"
            f"Coarse sweep time: {t1 - t0} seconds.\n"
            f"Total time elapsed: {t2 - t0} seconds.\n"
            "---------------------------------------------------------------"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--model_fit', type=str)
    # Verbosity flags
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--fitter_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    parser.add_argument('--warn', action='store_true')
    args = parser.parse_args()

    main(args)
