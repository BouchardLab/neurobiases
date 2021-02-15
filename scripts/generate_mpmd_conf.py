import argparse
import numpy as np


def main(args):
    # Job related arguments
    script_path = args.script_path
    conf_path = args.conf_path
    save_folder = args.save_folder
    tag = args.tag
    n_tasks = args.n_tasks
    # Model hyperparameters
    model_fit = args.model_fit
    N = args.N
    M = args.M
    K = args.K
    D = args.D
    n_models = args.n_models
    n_datasets = args.n_datasets
    # Other options
    no_warn = args.no_warn

    # Model hyperparameters
    n_coupling_locs = args.n_coupling_locs
    coupling_loc_min = args.coupling_loc_min
    coupling_loc_max = args.coupling_loc_max
    n_tuning_locs = args.n_tuning_locs
    tuning_loc_min = args.tuning_loc_min
    tuning_loc_max = args.tuning_loc_max

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
        fitter_rng = np.random.default_rng()
        fitter_rng = fitter_rng.integers(low=0, high=2**32 - 1)
    else:
        fitter_rng = args.fitter_rng

    counter = 0
    with open(conf_path, 'w') as tasks:
        # Iterate over each job, writing it to the configuration file
        for ii in range(n_coupling_locs):
            for jj in range(n_tuning_locs):
                for kk, (coupling_rng, tuning_rng) in enumerate(zip(coupling_rngs, tuning_rngs)):
                    for ll, dataset_rng in enumerate(dataset_rngs):
                        save_path = f"{save_folder}/{tag}_{ii}_{jj}_{kk}_{ll}.h5"
                        command = \
                            f"{counter} shifter -E -e " \
                            + f"HOME=/global/homes/s/sachdeva/ mpirun -np {n_tasks} " \
                            + f"-launcher fork python -u {script_path} " \
                            + f"--save_path={save_path} " \
                            + f"--model_fit={model_fit} " \
                            + f"--N={N} --M={M} --K={K} --D={D} " \
                            + f"--n_coupling_locs={n_coupling_locs} " \
                            + f"--coupling_loc_min={coupling_loc_min} " \
                            + f"--coupling_loc_max={coupling_loc_max} " \
                            + f"--coupling_loc_idx={ii} " \
                            + f"--n_tuning_locs={n_tuning_locs} " \
                            + f"--tuning_loc_min={tuning_loc_min} " \
                            + f"--tuning_loc_max={tuning_loc_max} " \
                            + f"--tuning_loc_idx={jj} " \
                            + f"--n_coupling_lambdas={args.n_coupling_lambdas} " \
                            + f"--coupling_lambda_lower={args.coupling_lambda_lower} " \
                            + f"--coupling_lambda_upper={args.coupling_lambda_upper} " \
                            + f"--n_tuning_lambdas={args.n_tuning_lambdas} " \
                            + f"--tuning_lambda_lower={args.tuning_lambda_lower} " \
                            + f"--tuning_lambda_upper={args.tuning_lambda_upper} " \
                            + f"--fine_sweep_frac={args.fine_sweep_frac} " \
                            + f"--max_K={args.max_K} " \
                            + f"--cv={args.cv} " \
                            + f"--coupling_distribution={args.coupling_distribution} " \
                            + f"--coupling_sparsity={args.coupling_sparsity} " \
                            + f"--coupling_scale={args.coupling_scale} " \
                            + f"--tuning_distribution={args.tuning_distribution} " \
                            + f"--tuning_sparsity={args.tuning_sparsity} " \
                            + f"--coupling_scale={args.coupling_scale} " \
                            + f"--corr_cluster={args.corr_cluster} " \
                            + f"--corr_back={args.corr_back} " \
                            + f"--solver={args.solver} " \
                            + f"--initialization={args.initialization} " \
                            + f"--max_iter={args.max_iter} " \
                            + f"--tol={args.tol} " \
                            + "--refit " \
                            + f"--coupling_rng={coupling_rng} " \
                            + f"--tuning_rng={tuning_rng} " \
                            + f"--dataset_rng={dataset_rng} " \
                            + f"--fitter_rng={fitter_rng} "
                        if no_warn:
                            command += " --no_warn"
                        command += "\n"
                        tasks.write(command)
                        # Adjust counter for task list
                        counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--script_path', type=str)
    parser.add_argument('--conf_path', type=str)
    parser.add_argument('--save_folder', type=str)
    # NERSC options
    parser.add_argument('--tag', type=str)
    parser.add_argument('--n_tasks', type=int, default=32)
    # Fixed model hyperparameters
    parser.add_argument('--model_fit', default='em')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--n_datasets', type=int, default=30)
    # Variable model hyperparameters
    parser.add_argument('--n_coupling_locs', type=int, default=30)
    parser.add_argument('--coupling_loc_min', type=float, default=-3)
    parser.add_argument('--coupling_loc_max', type=float, default=3)
    parser.add_argument('--n_tuning_locs', type=int, default=30)
    parser.add_argument('--tuning_loc_min', type=float, default=-3)
    parser.add_argument('--tuning_loc_max', type=float, default=3)
    # CV fitting hyperparameters
    parser.add_argument('--n_coupling_lambdas', type=int, default=30)
    parser.add_argument('--coupling_lambda_lower', type=float, default=-5)
    parser.add_argument('--coupling_lambda_upper', type=float, default=5)
    parser.add_argument('--n_tuning_lambdas', type=int, default=30)
    parser.add_argument('--tuning_lambda_lower', type=float, default=-5)
    parser.add_argument('--tuning_lambda_upper', type=float, default=5)
    parser.add_argument('--fine_sweep_frac', type=float, default=0.1)
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
    # Random seeds
    parser.add_argument('--coupling_rng', type=int, default=1)
    parser.add_argument('--tuning_rng', type=int, default=1)
    parser.add_argument('--dataset_rng', type=int, default=1)
    parser.add_argument('--fitter_rng', type=int, default=-1)
    # Other options
    parser.add_argument('--no_warn', action='store_true')
    args = parser.parse_args()

    main(args)