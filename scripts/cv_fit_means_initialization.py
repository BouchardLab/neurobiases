import argparse
import h5py
import numpy as np
import time

from mpi4py import MPI
from neurobiases.solver_utils import cv_solver_full


def main(args):
    save_path = args.save_path
    selection = args.selection
    N = args.N
    M = args.M
    K = args.K
    D = args.D
    n_datasets = args.n_datasets
    n_models = args.n_models

    # Model hyperparameters
    n_coupling_locs = args.n_coupling_locs
    n_tuning_locs = args.n_tuning_locs
    coupling_locs = np.linspace(args.coupling_loc_min,
                                args.coupling_loc_max,
                                n_coupling_locs)
    tuning_locs = np.linspace(args.tuning_loc_min,
                              args.tuning_loc_max,
                              n_tuning_locs)
    # Create hyperparameters for CV fitting
    coupling_lambdas = np.array([0.])
    tuning_lambdas = np.array([0.])
    Ks = np.arange(args.max_K) + 1

    # Process coupling random states
    if args.coupling_rng == -1:
        coupling_rng = np.random.default_rng()
    else:
        coupling_rng = np.random.default_rng(args.coupling_rng)
    # Make sure we repeat the coupling seeds so that we get the same model
    coupling_rngs = np.repeat(coupling_rng.integers(low=0, high=2**32-1), n_models)
    # Process tuning random states
    if args.tuning_rng == -1:
        tuning_rng = np.random.default_rng()
    else:
        tuning_rng = np.random.default_rng(args.tuning_rng)
    # Make sure we repeat the tuning seeds so that we get the same model
    tuning_rngs = np.repeat(tuning_rng.integers(low=0, high=2**32-1), n_models)
    # Process dataset random states
    if args.dataset_rng == -1:
        dataset_rng = np.random.default_rng()
    else:
        dataset_rng = np.random.default_rng(args.dataset_rng)
    dataset_rngs = dataset_rng.integers(
        low=0,
        high=2**32 - 1,
        size=n_datasets)
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

    mlls, bics, a, a_est, b, b_est, B, B_est, Psi, Psi_est, L, L_est = \
        cv_solver_full(
            method='em',
            selection=selection,
            M=M, N=N, K=K, D=D,
            coupling_distribution=args.coupling_distribution,
            coupling_sparsities=np.array([args.coupling_sparsity]),
            coupling_locs=coupling_locs,
            coupling_scale=args.coupling_scale,
            coupling_rngs=coupling_rngs,
            tuning_distribution=args.tuning_distribution,
            tuning_sparsities=np.array([args.tuning_sparsity]),
            tuning_locs=tuning_locs,
            tuning_scale=args.tuning_scale,
            tuning_rngs=tuning_rngs,
            corr_clusters=np.array([args.corr_cluster]),
            corr_back=args.corr_back,
            dataset_rngs=dataset_rngs,
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

    if rank == 0:
        if selection == 'sparse':
            shape_key = np.array(['tuning_loc',
                                  'coupling_loc',
                                  'model_idx',
                                  'dataset_idx',
                                  'split_idx'])
        elif selection == 'oracle':
            shape_key = np.array(['tuning_loc',
                                  'coupling_loc',
                                  'model_idx',
                                  'dataset_idx',
                                  'split_idx'])
        results = h5py.File(save_path, 'w')
        shape_key_h5 = results.create_dataset(
            'shape_key',
            (len(shape_key),),
            dtype=h5py.special_dtype(vlen=str)
        )
        shape_key_h5[:] = shape_key
        results['coupling_rngs'] = coupling_rngs
        results['tuning_rngs'] = tuning_rngs
        results['dataset_rngs'] = dataset_rngs
        results['mlls'] = np.squeeze(mlls)
        results['bics'] = np.squeeze(bics)
        results['a_true'] = np.squeeze(a)
        results['a_est'] = np.squeeze(a_est)
        results['b_true'] = np.squeeze(b)
        results['b_est'] = np.squeeze(b_est)
        results['B_true'] = np.squeeze(B)
        results['B_est'] = np.squeeze(B_est)
        results['Psi_true'] = np.squeeze(Psi)
        results['Psi_est'] = np.squeeze(Psi_est)
        results['L_true'] = np.squeeze(L, axis=(0, 2, 5))
        results['L_est'] = np.squeeze(L_est, axis=(0, 2, 5))
        results['coupling_locs'] = coupling_locs
        results['tuning_locs'] = tuning_locs
        results.attrs['N'] = N
        results.attrs['M'] = M
        results.attrs['K'] = K
        results.attrs['D'] = D
        results.attrs['n_datasets'] = n_datasets
        results.attrs['n_models'] = n_models
        results.attrs['n_splits'] = args.cv
        results.attrs['coupling_distribution'] = args.coupling_distribution
        results.attrs['coupling_sparsity'] = args.coupling_sparsity
        results.attrs['coupling_scale'] = args.coupling_scale
        results.attrs['tuning_distribution'] = args.tuning_distribution
        results.attrs['tuning_sparsity'] = args.tuning_sparsity
        results.attrs['tuning_scale'] = args.tuning_scale
        results.attrs['corr_cluster'] = args.corr_cluster
        results.attrs['corr_back'] = args.corr_back
        results.attrs['solver'] = args.solver
        results.attrs['initialization'] = args.initialization
        results.attrs['max_iter'] = args.max_iter
        results.attrs['tol'] = args.tol
        results.attrs['coupling_rng'] = args.coupling_rng
        results.attrs['tuning_rng'] = args.tuning_rng
        results.attrs['dataset_rng'] = args.dataset_rng
        results.close()

    if rank == 0:
        print('Successfully Saved.')
        t2 = time.time()
        print('Job complete. Total time: ', t2 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--selection', default='sparse')
    # Fixed model hyperparameters
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--n_datasets', type=int, default=10)
    parser.add_argument('--n_models', type=int, default=10)
    # Variable model hyperparameters
    parser.add_argument('--n_coupling_locs', type=int, default=30)
    parser.add_argument('--coupling_loc_min', type=float, default=-3)
    parser.add_argument('--coupling_loc_max', type=float, default=3)
    parser.add_argument('--n_tuning_locs', type=int, default=30)
    parser.add_argument('--tuning_loc_min', type=float, default=-3)
    parser.add_argument('--tuning_loc_max', type=float, default=3)
    # CV fitting hyperparameters
    parser.add_argument('--max_K', type=int, default=1)
    parser.add_argument('--cv', type=int, default=3)
    # Model parameters
    parser.add_argument('--coupling_distribution', default='gaussian')
    parser.add_argument('--coupling_sparsity', type=float, default=0.5)
    parser.add_argument('--coupling_scale', type=float, default=0.5)
    parser.add_argument('--tuning_distribution', default='gaussian')
    parser.add_argument('--tuning_sparsity', type=float, default=0.5)
    parser.add_argument('--tuning_scale', type=float, default=0.5)
    parser.add_argument('--corr_cluster', type=float, default=0.25)
    parser.add_argument('--corr_back', type=float, default=0.10)
    # Fitter arguments
    parser.add_argument('--solver', default='ow_lbfgs')
    parser.add_argument('--initialization', default='fits')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--refit', action='store_true')
    # Random seeds
    parser.add_argument('--coupling_rng', type=int, default=-1)
    parser.add_argument('--tuning_rng', type=int, default=-1)
    parser.add_argument('--dataset_rng', type=int, default=-1)
    parser.add_argument('--fitter_rng', type=int, default=-1)
    # Verbosity flags
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--fitter_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    args = parser.parse_args()

    main(args)
