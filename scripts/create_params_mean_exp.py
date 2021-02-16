import argparse
import h5py
import numpy as np


def main(args):
    save_folder = args.save_folder
    tag = args.tag

    # Experiment configuration
    N = args.N
    M = args.M
    K = args.K
    D = args.D
    n_models = args.n_models
    n_datasets = args.n_datasets
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
    if args.coupling_rng == -1:
        coupling_rng = np.random.default_rng()
    else:
        coupling_rng = np.random.default_rng(args.coupling_rng)
    coupling_rngs = coupling_rng.integers(
        low=0,
        high=2**32-1,
        size=n_models)
    # Process tuning random states
    if args.tuning_rng == -1:
        tuning_rng = np.random.default_rng()
    else:
        tuning_rng = np.random.default_rng(args.tuning_rng)
    tuning_rngs = tuning_rng.integers(
        low=0,
        high=2**32-1,
        size=n_models)
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
        fitter_rng = np.random.randint(low=0, high=2**32 - 1)
    else:
        fitter_rng = args.fitter_rng

    for ii, coupling_loc in enumerate(coupling_locs):
        for jj, tuning_loc in enumerate(tuning_locs):
            for kk, (coupling_rng, tuning_rng) in enumerate(zip(coupling_rngs, tuning_rngs)):
                for ll, dataset_rng in enumerate(dataset_rngs):
                    save_path = f"{save_folder}/{tag}_{kk}_{ii}_{jj}_{ll}.h5"
                    # Load the model configuration
                    with h5py.File(save_path, 'w') as params:
                        # Triangular model hyperparameters
                        params.attrs['N'] = N
                        params.attrs['M'] = M
                        params.attrs['K'] = K
                        params.attrs['D'] = D
                        params.attrs['corr_cluster'] = args.corr_cluster
                        params.attrs['corr_back'] = args.corr_back
                        params.attrs['coupling_distribution'] = args.coupling_distribution
                        params.attrs['coupling_sparsity'] = args.coupling_sparsity
                        params.attrs['coupling_loc'] = coupling_loc
                        params.attrs['coupling_scale'] = args.coupling_scale
                        params.attrs['coupling_rng'] = coupling_rng
                        params.attrs['tuning_distribution'] = args.tuning_distribution
                        params.attrs['tuning_sparsity'] = args.tuning_sparsity
                        params.attrs['tuning_loc'] = tuning_loc
                        params.attrs['tuning_scale'] = args.tuning_scale
                        params.attrs['tuning_rng'] = tuning_rng
                        # Random seeds
                        params.attrs['coupling_rng'] = coupling_rng
                        params.attrs['tuning_rng'] = tuning_rng
                        params.attrs['dataset_rng'] = dataset_rng
                        params.attrs['fitter_rng'] = fitter_rng
                        # Training hyperparameters
                        params['Ks'] = Ks
                        params['coupling_lambdas'] = coupling_lambdas
                        params['tuning_lambdas'] = tuning_lambdas
                        # Training settings
                        params.attrs['cv'] = args.cv
                        params.attrs['fine_sweep_frac'] = args.fine_sweep_frac
                        params.attrs['solver'] = args.solver
                        params.attrs['initialization'] = args.initialization
                        params.attrs['max_iter'] = args.max_iter
                        params.attrs['tol'] = args.tol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--tag', type=str)
    # Fixed model hyperparameters
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--n_datasets', type=int, default=30)
    # Variable model hyperparameters
    parser.add_argument('--n_coupling_locs', type=int, default=30)
    parser.add_argument('--coupling_loc_min', type=float, default=-2)
    parser.add_argument('--coupling_loc_max', type=float, default=2)
    parser.add_argument('--n_tuning_locs', type=int, default=30)
    parser.add_argument('--tuning_loc_min', type=float, default=-2)
    parser.add_argument('--tuning_loc_max', type=float, default=2)
    # CV fitting hyperparameters
    parser.add_argument('--n_coupling_lambdas', type=int, default=25)
    parser.add_argument('--coupling_lambda_lower', type=float, default=-7)
    parser.add_argument('--coupling_lambda_upper', type=float, default=-2)
    parser.add_argument('--n_tuning_lambdas', type=int, default=25)
    parser.add_argument('--tuning_lambda_lower', type=float, default=-7)
    parser.add_argument('--tuning_lambda_upper', type=float, default=-2)
    parser.add_argument('--fine_sweep_frac', type=float, default=0.1)
    parser.add_argument('--max_K', type=int, default=1)
    parser.add_argument('--cv', type=int, default=3)
    # Model parameters
    parser.add_argument('--coupling_distribution', default='gaussian')
    parser.add_argument('--coupling_sparsity', type=float, default=0.5)
    parser.add_argument('--coupling_scale', type=float, default=0.25)
    parser.add_argument('--tuning_distribution', default='gaussian')
    parser.add_argument('--tuning_sparsity', type=float, default=0.5)
    parser.add_argument('--tuning_scale', type=float, default=0.25)
    parser.add_argument('--corr_cluster', type=float, default=0.25)
    parser.add_argument('--corr_back', type=float, default=0.10)
    # Fitter arguments
    parser.add_argument('--solver', default='ow_lbfgs')
    parser.add_argument('--initialization', default='fits')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-8)
    # Random seeds
    parser.add_argument('--coupling_rng', type=int, default=-1)
    parser.add_argument('--tuning_rng', type=int, default=-1)
    parser.add_argument('--dataset_rng', type=int, default=-1)
    parser.add_argument('--fitter_rng', type=int, default=-1)
    args = parser.parse_args()

    main(args)
