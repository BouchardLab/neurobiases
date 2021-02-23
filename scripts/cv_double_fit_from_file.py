import argparse
import h5py
import numpy as np
import os
import time
import warnings

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from neurobiases import TriangularModel
from neurobiases.solver_utils import cv_sparse_solver_single
from sklearn.exceptions import ConvergenceWarning


def main(args):
    # Turn off convergence warning by default
    if not args.warn:
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    verbose = args.verbose
    # Job settings
    file_path = args.file_path
    model_fit = args.model_fit

    # Open up experiment settings
    with h5py.File(file_path, 'r') as params:
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
        coupling_rng = params.attrs['coupling_rng']
        tuning_distribution = params.attrs['tuning_distribution']
        tuning_sparsity = params.attrs['tuning_sparsity']
        tuning_loc = params.attrs['tuning_loc']
        tuning_scale = params.attrs['tuning_scale']
        tuning_rng = params.attrs['tuning_rng']
        # Random seeds
        coupling_rng = params.attrs['coupling_rng']
        tuning_rng = params.attrs['tuning_rng']
        dataset_rng = params.attrs['dataset_rng']
        fitter_rng = params.attrs['fitter_rng']
        # Training hyperparameters
        Ks = params['Ks'][:]
        coupling_lambdas = params['coupling_lambdas'][:]
        n_coupling_lambdas = coupling_lambdas.size
        tuning_lambdas = params['tuning_lambdas'][:]
        n_tuning_lambdas = tuning_lambdas.size
        # Training settings
        cv = params.attrs['cv']
        fine_sweep_frac = params.attrs['fine_sweep_frac']
        solver = params.attrs['solver']
        initialization = params.attrs['initialization']
        max_iter = params.attrs['max_iter']
        tol = params.attrs['tol']

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
        stim_distribution='uniform'
    )
    # Generate data using seed
    X, Y, y = tm.generate_samples(n_samples=D, rng=int(dataset_rng))

    # Run coarse sweep CV
    coarse_sweep_results = \
        cv_sparse_solver_single(
            method=model_fit,
            X=X,
            Y=Y,
            y=y,
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
            fitter_rng=fitter_rng
        )

    if rank == 0:
        # Identify best hyperparameter set according to BIC
        bics = coarse_sweep_results[1]
        median_bics = np.median(bics, axis=-1)
        best_hyps = np.unravel_index(np.argmin(median_bics), median_bics.shape)
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
            X=X,
            Y=Y,
            y=y,
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
            fitter_rng=fitter_rng
        )
    if model_fit == 'em':
        mlls, bics, a_est, b_est, B_est, Psi_est, L_est = fine_sweep_results
    elif model_fit == 'tc':
        mses, bics, a_est, b_est = fine_sweep_results

    if rank == 0:
        # Get best overall fit
        median_bics = np.median(bics, axis=-1)
        best_hyps = np.unravel_index(np.argmin(median_bics), median_bics.shape)
        a_est_best = a_est[best_hyps]
        b_est_best = b_est[best_hyps]
        if model_fit == 'em':
            B_est_best = B_est[best_hyps]
            Psi_est_best = Psi_est[best_hyps]
            L_est_best = L_est[best_hyps]

        # Save results
        with h5py.File(file_path, 'a') as results:
            if model_fit == 'em':
                results['mlls'] = np.squeeze(mlls[best_hyps])
            else:
                results['mses'] = np.squeeze(mses[best_hyps])
            results['bics'] = np.squeeze(bics[best_hyps])
            # True parameters
            results['a_true'] = tm.a.ravel()
            results['b_true'] = tm.b.ravel()
            results['B_true'] = tm.B
            results['Psi_true'] = tm.Psi
            results['L_true'] = tm.L
            # Estimated parameters
            results['a_est'] = a_est_best
            results['b_est'] = b_est_best
            if model_fit == 'em':
                results['B_est'] = B_est_best
                results['Psi_est'] = Psi_est_best
                results['L_est'] = L_est_best
            # CV details
            results.attrs['best_coupling_lambda'] = coupling_lambdas[best_hyps[0]]
            results.attrs['best_tuning_lambda'] = tuning_lambdas[best_hyps[1]]

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
    parser.add_argument('--model_fit', type=str)
    # Verbosity flags
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--fitter_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    parser.add_argument('--warn', action='store_true')
    args = parser.parse_args()

    main(args)
