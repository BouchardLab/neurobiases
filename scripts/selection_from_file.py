import argparse
import h5py
import warnings

from mpi4py import MPI
from neurobiases import TriangularModel
from pyuoi.linear_model import UoI_Lasso
from sklearn.exceptions import ConvergenceWarning


def main(args):
    # Turn off convergence warning by default
    if not args.warn:
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    # Job settings
    file_path = args.file_path

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
        # Training settings
        max_iter = params.attrs['max_iter']

    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.rank

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
    # Create UoI Lasso fitting object
    fitter = UoI_Lasso(
        n_boots_sel=30,
        n_boots_est=30,
        selection_frac=0.90,
        estimation_frac=0.90,
        n_lambdas=50,
        stability_selection=0.90,
        estimation_score='bic',
        warm_start=False,
        max_iter=max_iter,
        comm=comm)
    # Perform tuning fit
    fitter.fit(X, y.ravel())
    if rank == 0:
        tuning_selection = fitter.coef_ != 0
    fitter.fit(Y, y.ravel())
    if rank == 0:
        coupling_selection = fitter.coef_ != 0
    # Save coarse results
    with h5py.File(file_path, 'a') as results:
        results['coupling_selection_uoi'] = coupling_selection
        results['tuning_selection_uoi'] = tuning_selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run UoI solver on triangular model data.')
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--warn', action='store_true')
    args = parser.parse_args()

    main(args)
