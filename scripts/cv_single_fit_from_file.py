import argparse
import h5py
import os
import time
import warnings

from mpi4py import MPI
from neurobiases import TriangularModel
from neurobiases.solver_utils import cv_sparse_solver_single
from sklearn.exceptions import ConvergenceWarning


def main(args):
    # Turn off convergence warning by default
    if not args.warn:
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
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
        stim_distribution='uniform')
    # Generate data using seed
    X, Y, y = tm.generate_samples(n_samples=D, rng=int(dataset_rng))

    # Run coarse sweep CV
    sweep_results = \
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
            numpy=args.numpy,
            comm=comm,
            cv_verbose=args.cv_verbose,
            fitter_verbose=args.fitter_verbose,
            mstep_verbose=args.mstep_verbose,
            fitter_rng=fitter_rng)

    if model_fit == 'em':
        mlls, bics, a_est, b_est, B_est, Psi_est, L_est = sweep_results
    elif model_fit == 'tc':
        mses, bics, a_est, b_est = sweep_results

    if rank == 0:
        # Save results
        with h5py.File(file_path, 'a') as results:
            if model_fit == 'em':
                results['mlls'] = mlls
                results['B_est'] = B_est
                results['Psi_est'] = Psi_est
                results['L_est'] = L_est
            else:
                results['mses'] = mses
            results['bics'] = bics
            # True parameters
            results['a_true'] = tm.a.ravel()
            results['b_true'] = tm.b.ravel()
            results['B_true'] = tm.B
            results['Psi_true'] = tm.Psi
            results['L_true'] = tm.L
            # Estimated parameters
            results['a_est'] = a_est
            results['b_est'] = b_est
            # Data
            results['X'] = X
            results['Y'] = Y
            results['y'] = y

        t1 = time.time()
        print(
            "---------------------------------------------------------------\n"
            "Job complete: Performed a single CV coarse-fine sweep.\n"
            f"File: {os.path.basename(file_path)}\n"
            f"Model: {N} coupling, {M} tuning, {K} latent, {D} samples\n"
            f"Number of processes: {size}\n"
            f"Total number of tasks: {n_total_tasks}\n"
            f"Sweep time: {t1 - t0} seconds.\n"
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
    parser.add_argument('--numpy', action='store_true')
    parser.add_argument('--warn', action='store_true')
    args = parser.parse_args()

    main(args)
