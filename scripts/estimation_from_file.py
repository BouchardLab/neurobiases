import argparse
import h5py
import numpy as np
import os
import warnings

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from neurobiases.experiments import generate_data_from_file
from neurobiases.solver_utils import cv_solver_oracle_selection
from sklearn.exceptions import ConvergenceWarning


def main(args):
    verbose = args.verbose
    # Turn off convergence warning by default
    if not args.warn:
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # File path to folder or file
    file_path = args.file_path
    if os.path.isfile(file_path):
        paths = [file_path]
    elif os.path.isdir(file_path):
        paths = [os.path.join(file_path, f)
                 for f in sorted(os.listdir(file_path))]
    else:
        raise ValueError("File path must be valid file or directory.")

    for path in paths:
        if verbose:
            if rank == 0:
                print(f"Fitting to file {path}.")

        X = None
        Y = None
        y = None
        a_mask = None
        b_mask = None
        if rank == 0:
            X, Y, y = generate_data_from_file(path)
            with h5py.File(path, 'r') as results:
                a_mask = (results[args.a_mask_group][:] != 0).astype(int)
                b_mask = (results[args.b_mask_group][:] != 0).astype(int)
        X = Bcast_from_root(X, comm=comm)
        Y = Bcast_from_root(Y, comm=comm)
        y = Bcast_from_root(y, comm=comm)
        a_mask = Bcast_from_root(a_mask, comm=comm)
        b_mask = Bcast_from_root(b_mask, comm=comm)

        scores_train, scores_test, aics, bics, a_est, b_est, B_est, Psi_est, L_est = \
            cv_solver_oracle_selection(
                X=X, Y=Y, y=y,
                Ks=np.arange(1, args.max_K + 1),
                cv=args.cv,
                a_mask=a_mask,
                b_mask=b_mask,
                solver=args.solver,
                initialization=args.initialization,
                max_iter=args.max_iter,
                tol=args.tol,
                refit=False,
                fitter_rng=args.fitter_rng,
                numpy=False,
                comm=comm,
                fit_intercept=False)

        if rank == 0:
            with h5py.File(path, 'a') as results:
                group = results.create_group(args.results_group)
                group['scores_train'] = np.squeeze(scores_train)
                group['scores_test'] = np.squeeze(scores_test)
                group['aics'] = np.squeeze(aics)
                group['bics'] = np.squeeze(bics)
                group['a_est'] = np.squeeze(a_est)
                group['b_est'] = np.squeeze(b_est)
                group['B_est'] = np.squeeze(B_est)
                group['Psi_est'] = np.squeeze(Psi_est)
                group['L_est'] = np.squeeze(L_est)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--a_mask_group', type=str)
    parser.add_argument('--b_mask_group', type=str)
    parser.add_argument('--results_group', type=str)
    parser.add_argument('--max_K', type=int, default=1)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--solver', type=str, default='scipy_lbfgs')
    parser.add_argument('--initialization', type=str, default='fits')
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--fitter_rng', type=int, default=2332)
    parser.add_argument('--warn', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
