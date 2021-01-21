import argparse
import h5py
import numpy as np
import time

from mpi4py import MPI
from neurobiases import TriangularModel
from neurobiases.solver_utils import cv_sparse_solver_single


def main(args):
    save_path = args.save_path
    model_fit = args.model_fit
    N = args.N
    M = args.M
    K = args.K
    D = args.D
    model_idx = args.model_idx
    dataset_idx = args.dataset_idx

    # Model hyperparameters
    coupling_locs = np.linspace(args.coupling_loc_min,
                                args.coupling_loc_max,
                                args.n_coupling_locs)
    coupling_loc = coupling_locs[args.coupling_loc_idx]
    tuning_locs = np.linspace(args.tuning_loc_min,
                              args.tuning_loc_max,
                              args.n_tuning_locs)
    tuning_loc = tuning_locs[args.tuning_loc_idx]
    # Create hyperparameters for CV fitting
    n_coupling_lambdas = args.n_coupling_lambdas
    coupling_lambdas = np.logspace(args.coupling_lambda_lower,
                                   args.coupling_lambda_upper,
                                   num=n_coupling_lambdas)
    n_tuning_lambdas = args.n_tuning_lambdas
    tuning_lambdas = np.logspace(args.tuning_lambda_lower,
                                 args.tuning_lambda_upper,
                                 num=n_tuning_lambdas)
    Ks = np.arange(args.max_K) + 1

    # Process random seeds for model
    coupling_rng = args.coupling_rng
    tuning_rng = args.tuning_rng
    dataset_rng = args.dataset_rng
    # Process fitter random state
    if args.fitter_rng == -1:
        fitter_rng = None
    else:
        fitter_rng = args.fitter_rng

    # MPI communicator
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    if rank == 0:
        t0 = time.time()
        print('--------------------------------------------------------------')
        print(f'{size} processes running, this is rank {rank}.')
        print('--------------------------------------------------------------')

    # Generate triangular model
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        M=M, N=N, K=K,
        corr_cluster=args.corr_cluster,
        corr_back=args.corr_back,
        tuning_distribution=args.tuning_distribution,
        tuning_sparsity=args.tuning_sparsity,
        tuning_loc=tuning_loc,
        tuning_scale=args.tuning_scale,
        tuning_rng=tuning_rng,
        coupling_distribution=args.coupling_distribution,
        coupling_sparsity=args.coupling_sparsity,
        coupling_loc=coupling_loc,
        coupling_scale=args.coupling_scale,
        coupling_rng=coupling_rng,
        stim_distribution='uniform'
    )
    # Generate data using seed
    X, Y, y = tm.generate_samples(n_samples=D, rng=int(dataset_rng))

    results = \
        cv_sparse_solver_single(
            method=model_fit,
            X=X,
            Y=Y,
            y=y,
            coupling_lambdas=coupling_lambdas,
            tuning_lambdas=tuning_lambdas,
            Ks=Ks,
            cv=args.cv,
            solver=args.solver,
            initialization=args.initialization,
            max_iter=args.max_iter,
            tol=args.tol,
            refit=args.refit,
            comm=comm,
            cv_verbose=args.cv_verbose,
            fitter_verbose=args.fitter_verbose,
            mstep_verbose=args.mstep_verbose,
            fitter_rng=fitter_rng
        )

    if model_fit == 'em':
        mlls, bics, a, a_est, b, b_est, B, B_est, Psi, Psi_est, L, L_est = \
            results
    elif model_fit == 'tc':
        mses, bics, a, a_est, b, b_est, B, Psi, L = \
            results

    # TODO: second stage CV fit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--model_fit', default='em')
    # Fixed model hyperparameters
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--model_idx', type=int, default=0)
    parser.add_argument('--dataset_idx', type=int, default=0)
    # Variable model hyperparameters
    parser.add_argument('--n_coupling_locs', type=int, default=30)
    parser.add_argument('--coupling_loc_min', type=float, default=-3)
    parser.add_argument('--coupling_loc_max', type=float, default=3)
    parser.add_argument('--coupling_loc_idx', type=int, default=0)
    parser.add_argument('--n_tuning_locs', type=int, default=30)
    parser.add_argument('--tuning_loc_min', type=float, default=-3)
    parser.add_argument('--tuning_loc_max', type=float, default=3)
    parser.add_argument('--tuning_loc_idx', type=int, default=0)
    # CV fitting hyperparameters
    parser.add_argument('--n_coupling_lambdas', type=int, default=30)
    parser.add_argument('--coupling_lambda_lower', type=float, default=-5)
    parser.add_argument('--coupling_lambda_upper', type=float, default=-2)
    parser.add_argument('--n_tuning_lambdas', type=int, default=30)
    parser.add_argument('--tuning_lambda_lower', type=float, default=-5)
    parser.add_argument('--tuning_lambda_upper', type=float, default=-2)
    parser.add_argument('--max_K', type=int, default=1)
    parser.add_argument('--cv', type=int, default=3)
    # Model parameters
    parser.add_argument('--coupling_distribution', default='gaussian')
    parser.add_argument('--coupling_sparsity', type=float, default=0.5)
    parser.add_argument('--coupling_scale', type=float, default=1.)
    parser.add_argument('--tuning_distribution', default='gaussian')
    parser.add_argument('--tuning_sparsity', type=float, default=0.5)
    parser.add_argument('--tuning_scale', type=float, default=1.)
    parser.add_argument('--corr_cluster', type=float, default=0.25)
    parser.add_argument('--corr_back', type=float, default=0.10)
    # Fitter arguments
    parser.add_argument('--solver', default='ow_lbfgs')
    parser.add_argument('--initialization', default='fits')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--refit', action='store_true')
    # Random seeds
    parser.add_argument('--coupling_rng', type=int, default=1)
    parser.add_argument('--tuning_rng', type=int, default=1)
    parser.add_argument('--dataset_rng', type=int, default=1)
    parser.add_argument('--fitter_rng', type=int, default=-1)
    # Verbosity flags
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--fitter_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    args = parser.parse_args()

    main(args)
