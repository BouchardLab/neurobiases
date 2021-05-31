import argparse
import h5py
import numpy as np
import os
import warnings

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from neurobiases.experiments import generate_data_from_file
from pyuoi.linear_model import UoI_Lasso
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
        if rank == 0:
            X, Y, y = generate_data_from_file(path)
        X = Bcast_from_root(X, comm=comm)
        Y = Bcast_from_root(Y, comm=comm)
        y = Bcast_from_root(y, comm=comm)

        # Create UoI Lasso fitting object
        fitter = UoI_Lasso(
            n_boots_sel=args.n_boots_sel,
            n_boots_est=args.n_boots_est,
            selection_frac=args.selection_frac,
            estimation_frac=args.estimation_frac,
            n_lambdas=args.n_lambdas,
            stability_selection=args.stability_selection,
            estimation_score=args.estimation_score,
            warm_start=False,
            max_iter=args.max_iter,
            comm=comm)
        # Perform tuning fit
        fitter.fit(X, y.ravel())
        if rank == 0:
            t_selection = fitter.coef_ != 0
        # Perform coupling fit
        fitter.fit(Y, y.ravel())
        if rank == 0:
            c_selection = fitter.coef_ != 0
        # Perform tuning and coupling fit
        Z = np.concatenate((X, Y), axis=1)
        fitter.fit(Z, y.ravel())
        if rank == 0:
            t_selection_tc, c_selection_tc = np.split(fitter.coef_ != 0, [X.shape[1]])
            # Save selection profiles
            with h5py.File(path, 'a') as results:
                results['c_selection_uoi'] = c_selection
                results['t_selection_uoi'] = t_selection
                results['c_selection_uoi_tc'] = c_selection_tc
                results['t_selection_uoi_tc'] = t_selection_tc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--selection_frac', type=float, default=0.80)
    parser.add_argument('--estimation_frac', type=float, default=0.80)
    parser.add_argument('--stability_selection', type=float, default=1.)
    parser.add_argument('--estimation_score', type=str, default='BIC')
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--warn', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
