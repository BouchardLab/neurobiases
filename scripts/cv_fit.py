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
    random_state = args.random_state

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

    # create triangular model kwargs
    tuning_kwargs, coupling_kwargs, noise_kwargs, stim_kwargs = \
        TriangularModel.generate_kwargs(
            parameter_design='direct_response',
            M=M, N=N, K=K, corr_cluster=0.4, corr_back=0.1,
            tuning_sparsity=0.5, coupling_sparsity=0.5,
            tuning_random_state=random_state, coupling_random_state=random_state)
    # generate triangular model
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        tuning_kwargs=tuning_kwargs,
        coupling_kwargs=coupling_kwargs,
        noise_kwargs=noise_kwargs,
        stim_kwargs=stim_kwargs)

    X = None
    Y = None
    y = None
    # generate samples
    if rank == 0:
        X, Y, y = tm.generate_samples(n_samples=D, random_state=random_state)

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

    scores, a, b, B, Psi_tr = cv_sparse_em_solver(
        X=X, Y=Y, y=y,
        coupling_lambdas=coupling_lambdas, tuning_lambdas=tuning_lambdas, Ks=Ks,
        cv=args.cv, max_iter=args.max_iter, tol=args.tol, comm=comm,
        verbose=True
    )
    if rank == 0:
        np.savez(save_path, scores=scores, a=a, b=b, B=B, Psi_tr=Psi_tr)
        print('Successfully Saved.')
        t2 = time.time()
        print('Job complete. Total time: ', t2 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run noise correlations analysis.')
    parser.add_argument('--save_path', type=str,
                        help='Path where results will be saved.')
    parser.add_argument('--N', type=int, default=15)
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
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--random_state', type=int, default=2332)
    args = parser.parse_args()

    main(args)
