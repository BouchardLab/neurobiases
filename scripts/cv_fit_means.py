import argparse
import h5py
import numpy as np
import time

from mpi4py import MPI
from neurobiases.solver_utils import (cv_sparse_em_solver_full,
                                      cv_sparse_tc_solver_full)


def main(args):
    save_path = args.save_path
    model_fit = args.model_fit
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
    n_coupling_lambdas = args.n_coupling_lambdas
    coupling_lambdas = np.logspace(args.coupling_lambda_lower,
                                   args.coupling_lambda_upper,
                                   num=n_coupling_lambdas)
    n_tuning_lambdas = args.n_tuning_lambdas
    tuning_lambdas = np.logspace(args.tuning_lambda_lower,
                                 args.tuning_lambda_upper,
                                 num=n_tuning_lambdas)
    Ks = np.arange(args.max_K) + 1

    # Process coupling random states
    if args.coupling_random_state == -1:
        coupling_random_state = np.random.default_rng()
    else:
        coupling_random_state = np.random.default_rng(args.coupling_random_state)
    coupling_random_states = coupling_random_state.integers(
        low=0,
        high=2**32-1,
        size=n_models)
    # Process tuning random states
    if args.tuning_random_state == -1:
        tuning_random_state = np.random.default_rng()
    else:
        tuning_random_state = np.random.default_rng(args.tuning_random_state)
    tuning_random_states = tuning_random_state.integers(
        low=0,
        high=2**32-1,
        size=n_models)
    # Process dataset random states
    if args.dataset_random_state == -1:
        dataset_random_state = np.random.default_rng()
    else:
        dataset_random_state = np.random.default_rng(args.dataset_random_state)
    dataset_random_states = dataset_random_state.integers(
        low=0,
        high=2**32 - 1,
        size=n_datasets)
    # Process fitter random state
    if args.fitter_random_state == -1:
        fitter_random_state = None
    else:
        fitter_random_state = args.fitter_random_state

    # MPI communicator
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    if rank == 0:
        t0 = time.time()
        n_total_tasks = n_coupling_locs * n_tuning_locs * n_models * n_datasets \
            * n_coupling_lambdas * n_tuning_lambdas * Ks.size * args.cv
        print('--------------------------------------------------------------')
        print(f'{size} processes running, this is rank {rank}.')
        print(f'Number of total tasks: {n_total_tasks}.')
        print('--------------------------------------------------------------')

    # Fit parameters according to TM (using EM)
    if model_fit == 'em':
        mlls, bics, a, a_est, b, b_est, B, B_est, Psi, Psi_est, L, L_est = \
            cv_sparse_em_solver_full(
                M=M, N=N, K=K, D=D,
                coupling_distribution=args.coupling_distribution,
                coupling_sparsities=np.array([args.coupling_sparsity]),
                coupling_locs=coupling_locs,
                coupling_scale=args.coupling_scale,
                coupling_random_states=coupling_random_states,
                tuning_distribution=args.tuning_distribution,
                tuning_sparsities=np.array([args.tuning_sparsity]),
                tuning_locs=tuning_locs,
                tuning_scale=args.tuning_scale,
                tuning_random_states=tuning_random_states,
                corr_clusters=np.array([args.corr_cluster]),
                corr_back=args.corr_back,
                dataset_random_states=dataset_random_states,
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
                em_verbose=args.fitter_verbose,
                mstep_verbose=args.mstep_verbose,
                random_state=fitter_random_state
            )
        if rank == 0:
            np.savez(
                save_path,
                coupling_random_states=coupling_random_states,
                tuning_random_states=tuning_random_states,
                dataset_random_states=dataset_random_states,
                scores=mlls,
                bics=bics,
                a_est=a,
                a_true=a_est,
                b_est=b,
                b_true=b_est,
                B_est=B,
                B_true=B_est,
                Psi_est=Psi,
                Psi_true=Psi_est,
                L_est=L,
                L_true=L_est
            )

    # Fit parameters according to TCM (using sparse TC solver)
    elif model_fit == 'tc':
        mses, bics, a, a_est, b, b_est, B, Psi, L = \
            cv_sparse_tc_solver_full(
                M=M, N=N, K=K, D=D,
                coupling_distribution=args.coupling_distribution,
                coupling_sparsities=np.array([args.coupling_sparsity]),
                coupling_locs=coupling_locs,
                coupling_scale=args.coupling_scale,
                coupling_random_states=coupling_random_states,
                tuning_distribution=args.tuning_distribution,
                tuning_sparsities=np.array([args.tuning_sparsity]),
                tuning_locs=tuning_locs,
                tuning_scale=args.tuning_scale,
                tuning_random_states=tuning_random_states,
                corr_clusters=np.array([args.corr_cluster]),
                corr_back=args.corr_back,
                dataset_random_states=dataset_random_states,
                coupling_lambdas=coupling_lambdas,
                tuning_lambdas=tuning_lambdas,
                cv=args.cv,
                solver=args.solver,
                initialization=args.initialization,
                max_iter=args.max_iter,
                tol=args.tol,
                refit=args.refit,
                comm=comm,
                cv_verbose=args.cv_verbose,
                tc_verbose=args.fitter_verbose,
            )
        if rank == 0:
            shape_key = np.array(['tuning_loc',
                                  'coupling_loc',
                                  'model_idx',
                                  'dataset_idx',
                                  'split_idx',
                                  'coupling_lambda',
                                  'tuning_lambda'])
            results = h5py.File(save_path, 'w')
            shape_key_h5 = results.create_dataset(
                'shape_key',
                (len(shape_key),),
                dtype=h5py.special_dtype(vlen=str)
            )
            shape_key_h5[:] = shape_key
            results['coupling_random_states'] = coupling_random_states
            results['tuning_random_states'] = tuning_random_states
            results['dataset_random_states'] = dataset_random_states
            results['mses'] = np.squeeze(mses)
            results['bics'] = np.squeeze(bics)
            results['a_true'] = np.squeeze(a)
            results['a_est'] = np.squeeze(a_est)
            results['b_true'] = np.squeeze(b)
            results['b_est'] = np.squeeze(b_est)
            results['B'] = np.squeeze(B)
            results['Psi'] = np.squeeze(Psi)
            results['L'] = np.squeeze(L, axis=(0, 2, 5))
            results['coupling_locs'] = coupling_locs
            results['tuning_locs'] = tuning_locs
            results['coupling_lambdas'] = coupling_lambdas
            results['tuning_lambdas'] = tuning_lambdas
            results.attrs['model_fit'] = args.model_fit
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
    else:
        raise ValueError('Incorrect model fit input.')

    if rank == 0:
        print('Successfully Saved.')
        t2 = time.time()
        print('Job complete. Total time: ', t2 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--model_fit', default='em')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--n_datasets', type=int, default=10)
    parser.add_argument('--n_models', type=int, default=10)
    # Model hyperparameters
    parser.add_argument('--n_coupling_locs', type=int, default=30)
    parser.add_argument('--coupling_loc_min', type=float, default=-3)
    parser.add_argument('--coupling_loc_max', type=float, default=3)
    parser.add_argument('--n_tuning_locs', type=int, default=30)
    parser.add_argument('--tuning_loc_min', type=float, default=-3)
    parser.add_argument('--tuning_loc_max', type=float, default=3)
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
    # Random states
    parser.add_argument('--coupling_random_state', type=int, default=-1)
    parser.add_argument('--tuning_random_state', type=int, default=-1)
    parser.add_argument('--dataset_random_state', type=int, default=-1)
    parser.add_argument('--fitter_random_state', type=int, default=-1)
    # Verbosity flags
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--fitter_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    args = parser.parse_args()

    main(args)
