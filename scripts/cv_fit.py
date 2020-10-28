import argparse
import numpy as np
import time

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from neurobiases import TriangularModel
from neurobiases.solver_utils import cv_sparse_em_solver


def main(args):
    N = args.N
    M = args.M
    K = args.K
    D = args.D
    save_path = args.save_path
    tm_random_state = args.tm_random_state
    if args.em_random_state == -1:
        em_random_state = None
    else:
        em_random_state = args.em_random_state

    # MPI communicator
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    if rank == 0:
        t0 = time.time()
        print('--------------------------------------------------------------')
        print(f'{size} processes running, this is rank {rank}.')
        print(f'Using {N} neurons and {M} tuning dimensions, with {D} samples.')
        print('--------------------------------------------------------------')

    # Generate triangular model
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        M=M, N=N, K=K, corr_cluster=0.25, corr_back=0.0,
        tuning_distribution=args.tuning_distribution,
        tuning_sparsity=args.tuning_sparsity,
        tuning_loc=args.tuning_loc,
        tuning_scale=args.tuning_scale,
        tuning_random_state=tm_random_state,
        coupling_distribution=args.coupling_distribution,
        coupling_sparsity=args.coupling_sparsity,
        coupling_loc=args.coupling_loc,
        coupling_scale=args.coupling_scale,
        coupling_sum=args.coupling_sum,
        coupling_random_state=tm_random_state,
        stim_distribution=args.stim_distribution
    )

    X = None
    Y = None
    y = None
    # generate samples
    if rank == 0:
        X, Y, y = tm.generate_samples(n_samples=D, random_state=tm_random_state)

    X = Bcast_from_root(X, comm)
    Y = Bcast_from_root(Y, comm)
    y = Bcast_from_root(y, comm)

    if rank == 0:
        print('Broadcasted data.')

    # create hyperparameters
    coupling_lambdas = np.logspace(args.coupling_lower,
                                   args.coupling_upper,
                                   num=args.n_coupling)
    tuning_lambdas = np.logspace(args.tuning_lower,
                                 args.tuning_upper,
                                 num=args.n_tuning)
    Ks = np.arange(args.max_K) + 1

    mlls, bics, a, b, B, Psi, L, n_iterations = cv_sparse_em_solver(
        X=X, Y=Y, y=y,
        solver='ow_lbfgs', initialization=args.initialization,
        coupling_lambdas=coupling_lambdas, tuning_lambdas=tuning_lambdas, Ks=Ks,
        cv=args.cv, max_iter=args.max_iter, tol=args.tol, refit=args.refit, comm=comm,
        cv_verbose=args.cv_verbose, em_verbose=args.em_verbose,
        mstep_verbose=args.mstep_verbose,
        random_state=em_random_state,
    )
    if rank == 0:
        np.savez(save_path,
                 scores=mlls,
                 bics=bics,
                 a_est=a,
                 a_true=tm.a.ravel(),
                 b_est=b,
                 b_true=tm.b.ravel(),
                 B_est=B,
                 B_true=tm.B,
                 Psi_est=Psi,
                 Psi_true=tm.Psi,
                 L_est=L,
                 L_true=tm.L,
                 X=X, Y=Y, y=y,
                 n_iterations=n_iterations)
        print('Successfully Saved.')
        t2 = time.time()
        print('Job complete. Total time: ', t2 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--coupling_lower', type=float, default=-5)
    parser.add_argument('--coupling_upper', type=float, default=-2)
    parser.add_argument('--n_coupling', type=int, default=5)
    parser.add_argument('--tuning_lower', type=float, default=-5)
    parser.add_argument('--tuning_upper', type=float, default=-2)
    parser.add_argument('--n_tuning', type=int, default=5)
    parser.add_argument('--max_K', type=int, default=1)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--initialization', default='fits')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--tuning_distribution', default='gaussian')
    parser.add_argument('--tuning_sparsity', default=0.5)
    parser.add_argument('--tuning_loc', type=float, default=0.)
    parser.add_argument('--tuning_scale', type=float, default=1.)
    parser.add_argument('--coupling_distribution', default='gaussian')
    parser.add_argument('--coupling_sparsity', default=0.5)
    parser.add_argument('--coupling_loc', type=float, default=0.)
    parser.add_argument('--coupling_scale', type=float, default=1.)
    parser.add_argument('--coupling_sum', type=float, default=0.)
    parser.add_argument('--stim_distribution', default='uniform')
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--tm_random_state', type=int, default=2332)
    parser.add_argument('--em_random_state', type=int, default=-1)
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--em_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    args = parser.parse_args()

    main(args)
