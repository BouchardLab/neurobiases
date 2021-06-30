import argparse
import h5py
import numpy as np
import os
import time
import warnings

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from neurobiases import TriangularModel, TCSolver
from neurobiases.solver_utils import cv_sparse_solver_single
from sklearn.exceptions import ConvergenceWarning


def main(args):
    # Turn off convergence warning by default
    if not args.warn:
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    verbose = args.verbose
    # Job settings
    file_path = args.file_path
    group = args.group

    # Open up experiment settings
    with h5py.File(file_path, 'r') as params:
        # Model settings
        # Triangular model hyperparameters
        N = params.attrs['N']
        M = params.attrs['M']
        K = params.attrs['K']
        D = params.attrs['D']
        corr_cluster = params.attrs['corr_cluster']
        corr_back = params.attrs['corr_back']
        coupling_distribution = params.attrs['coupling_distribution']
        coupling_sparsity = params.attrs['coupling_sparsity']
        coupling_loc = params.attrs['coupling_loc']
        coupling_scale = params.attrs['coupling_scale']
        tuning_distribution = params.attrs['tuning_distribution']
        tuning_sparsity = params.attrs['tuning_sparsity']
        tuning_loc = params.attrs['tuning_loc']
        tuning_scale = params.attrs['tuning_scale']
        # Random seeds
        coupling_rng = params.attrs['coupling_rng']
        tuning_rng = params.attrs['tuning_rng']
        dataset_rng = params.attrs['dataset_rng']
        fitter_rng = params.attrs['fitter_rng']

        # Fitting settings
        settings = params[group]
        model = settings.attrs['model']
        # Training hyperparameters
        Ks = settings['Ks'][:]
        coupling_lambdas = settings['coupling_lambdas'][:]
        n_coupling_lambdas = coupling_lambdas.size
        n_coupling_lambdas_fine = settings.attrs['n_coupling_lambdas_fine']
        tuning_lambdas = settings['tuning_lambdas'][:]
        n_tuning_lambdas = tuning_lambdas.size
        n_tuning_lambdas_fine = settings.attrs['n_tuning_lambdas_fine']
        # Training settings
        criterion = settings.attrs['criterion']
        cv = settings.attrs['cv']
        fine_sweep_frac = settings.attrs['fine_sweep_frac']
        solver = settings.attrs['solver']
        initialization = settings.attrs['initialization']
        max_iter = settings.attrs['max_iter']
        tol = settings.attrs['tol']
        Psi_transform = settings.attrs['Psi_transform']

    # MPI communicator
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    if rank == 0:
        t0 = time.time()
        n_total_tasks = n_coupling_lambdas * n_tuning_lambdas * cv

    # Generate triangular model
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        M=M,
        N=N,
        K=K,
        corr_cluster=corr_cluster,
        corr_back=corr_back,
        coupling_distribution=coupling_distribution,
        coupling_sparsity=coupling_sparsity,
        coupling_loc=coupling_loc,
        coupling_scale=coupling_scale,
        coupling_rng=coupling_rng,
        tuning_distribution=tuning_distribution,
        tuning_sparsity=tuning_sparsity,
        tuning_loc=tuning_loc,
        tuning_scale=tuning_scale,
        tuning_rng=tuning_rng,
        stim_distribution='uniform')
    # Generate data using seed
    X, Y, y = tm.generate_samples(n_samples=D, rng=int(dataset_rng))

    # Run coarse sweep CV
    coarse_sweep_results = \
        cv_sparse_solver_single(
            method=model,
            X=X,
            Y=Y,
            y=y,
            coupling_lambdas=coupling_lambdas,
            tuning_lambdas=tuning_lambdas,
            Ks=Ks,
            cv=cv,
            solver=solver,
            initialization=initialization,
            numpy=False,
            max_iter=max_iter,
            tol=tol,
            Psi_transform=Psi_transform,
            refit=True,
            comm=comm,
            cv_verbose=args.cv_verbose,
            fitter_verbose=args.fitter_verbose,
            mstep_verbose=args.mstep_verbose,
            fitter_rng=fitter_rng)

    if rank == 0:
        if model == 'tm':
            scores_train, scores_test, aics, bics, a_est, b_est, B_est, Psi_est, L_est = \
                coarse_sweep_results
        elif model == 'tc':
            scores_train, scores_test, aics, bics, a_est, b_est = coarse_sweep_results
        # Identify best hyperparameter set according to criterion
        if criterion == 'aic':
            mean_criterion = np.mean(aics, axis=-1)
        elif criterion == 'bic':
            mean_criterion = np.mean(bics, axis=-1)
        elif criterion == 'score':
            mean_criterion = -np.mean(scores_test, axis=-1)
        else:
            raise ValueError('Incorrect criterion specified.')
        best_hyps = np.unravel_index(np.argmin(mean_criterion), mean_criterion.shape)
        best_c_coupling = coupling_lambdas[best_hyps[0]]
        best_c_tuning = tuning_lambdas[best_hyps[1]]
        if model == 'tm':
            Ks = np.array([Ks[best_hyps[2]]])
        # Create new hyperparameter set
        coupling_lambda_lower = fine_sweep_frac * best_c_coupling
        coupling_lambda_upper = (1. / fine_sweep_frac) * best_c_coupling
        coupling_lambdas = np.linspace(coupling_lambda_lower,
                                       coupling_lambda_upper,
                                       num=n_coupling_lambdas_fine)
        tuning_lambda_lower = fine_sweep_frac * best_c_tuning
        tuning_lambda_upper = (1. / fine_sweep_frac) * best_c_tuning
        tuning_lambdas = np.linspace(tuning_lambda_lower,
                                     tuning_lambda_upper,
                                     num=n_tuning_lambdas_fine)
        # Save coarse results
        with h5py.File(file_path, 'a') as results:
            storage = results[group]
            storage['coupling_lambdas_fine'] = coupling_lambdas
            storage['tuning_lambdas_fine'] = tuning_lambdas
            # Metrics
            storage['aics_coarse'] = aics
            storage['bics_coarse'] = bics
            storage['scores_train_coarse'] = scores_train
            storage['scores_test_coarse'] = scores_test
            # Estimated parameters
            storage['a_est_coarse'] = a_est
            storage['b_est_coarse'] = b_est

        if verbose:
            print(f'First sweep complete. Best coupling lambda: {best_c_coupling}'
                  f' and best tuning lambda: {best_c_tuning}')

    # Broadcast new lambdas out
    coupling_lambdas = Bcast_from_root(coupling_lambdas, comm)
    tuning_lambdas = Bcast_from_root(tuning_lambdas, comm)
    if model == 'tm':
        Ks = Bcast_from_root(Ks, comm)
    # Verbosity update
    if rank == 0:
        t1 = time.time()

    # Run broad sweep CV
    fine_sweep_results = \
        cv_sparse_solver_single(
            method=model,
            X=X,
            Y=Y,
            y=y,
            coupling_lambdas=coupling_lambdas,
            tuning_lambdas=tuning_lambdas,
            Ks=Ks,
            cv=cv,
            solver=solver,
            initialization=initialization,
            numpy=False,
            max_iter=max_iter,
            tol=tol,
            Psi_transform=Psi_transform,
            refit=True,
            comm=comm,
            cv_verbose=args.cv_verbose,
            fitter_verbose=args.fitter_verbose,
            mstep_verbose=args.mstep_verbose,
            fitter_rng=fitter_rng)
    if model == 'tm':
        scores_train, scores_test, aics, bics, a_est, b_est, B_est, Psi_est, L_est = \
            fine_sweep_results
    elif model == 'tc':
        scores_train, scores_test, aics, bics, a_est, b_est = fine_sweep_results

    if rank == 0:
        # Get best overall fit
        if criterion == 'aic':
            mean_criterion = np.mean(aics, axis=-1)
        elif criterion == 'bic':
            mean_criterion = np.mean(bics, axis=-1)
        elif criterion == 'score':
            mean_criterion = -np.mean(scores_test, axis=-1)
        else:
            raise ValueError('Incorrect criterion specified.')
        best_hyps = np.unravel_index(np.argmin(mean_criterion), mean_criterion.shape)
        best_coupling_lambda = coupling_lambdas[best_hyps[0]]
        best_tuning_lambda = tuning_lambdas[best_hyps[1]]

        final_solver = TCSolver(
            X=X,
            Y=Y,
            y=y,
            solver=solver,
            c_coupling=best_coupling_lambda,
            c_tuning=best_tuning_lambda,
            fit_intercept=False,
            initialization=initialization,
            max_iter=max_iter,
            tol=tol,
            rng=fitter_rng).fit(refit=False)

        # Save results
        with h5py.File(file_path, 'a') as results:
            storage = results[group]
            storage['aics_fine'] = aics
            storage['bics_fine'] = bics
            storage['scores_train_fine'] = scores_train
            storage['scores_test_fine'] = scores_test
            # Estimated parameters
            storage['a_est_fine'] = a_est
            storage['b_est_fine'] = b_est
            # Final results
            storage['a_est'] = final_solver.a.ravel()
            storage['b_est'] = final_solver.b.ravel()
            storage['aic'] = final_solver.aic()
            storage['bic'] = final_solver.bic()
            if model == 'tc':
                storage['scores_train'] = final_solver.mse()
            # CV details
            storage.attrs['best_coupling_lambda'] = best_coupling_lambda
            storage.attrs['best_tuning_lambda'] = best_tuning_lambda

        t2 = time.time()
        print(
            "---------------------------------------------------------------\n"
            "Job complete: Performed a single CV coarse-fine sweep.\n"
            f"File: {os.path.basename(file_path)}\n"
            f"Model: {N} coupling, {M} tuning, {K} latent, {D} samples\n"
            f"Number of processes: {size}\n"
            f"Total number of tasks: {n_total_tasks}\n"
            f"Fine sweep centered on {best_c_coupling:0.2E} (coupling) "
            f"and {best_c_tuning:0.2E} (tuning).\n"
            f"Coarse sweep time: {t1 - t0} seconds.\n"
            f"Total time elapsed: {t2 - t0} seconds.\n"
            "---------------------------------------------------------------"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--group', type=str)
    # Verbosity flags
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--fitter_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    parser.add_argument('--warn', action='store_true')
    args = parser.parse_args()

    main(args)
