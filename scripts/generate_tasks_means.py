import argparse
import numpy as np
import os


def main(args):
    tasks_path = args.tasks_path
    wrapper_path = args.wrapper_path
    save_folder = args.save_folder
    model_fit = args.model_fit
    N = args.N
    M = args.M
    K = args.K
    D = args.D
    n_models = args.n_models
    n_datasets = args.n_datasets

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

    with open(tasks_path, 'w') as tasks:
        tasks.write('#!/usr/bin/env bash\n\n')
        for ii in range(n_coupling_locs):
            for jj in range(n_tuning_locs):
                for kk, (coupling_rng, tuning_rng) in enumerate(zip(coupling_rngs, tuning_rngs)):
                    for ll, dataset_rng in enumerate(dataset_rngs):
                        save_name = \
                            model_fit + "_" \
                            + str(ii) + "_" \
                            + str(jj) + "_" \
                            + str(kk) + "_" \
                            + str(ll) + ".h5"
                        save_path = os.path.join(save_folder, save_name)
                        command = wrapper_path + " "
                        command += save_path + " "  # 1
                        command += model_fit + " "  # 2
                        command += str(N) + " "  # 3
                        command += str(M) + " "  # 4
                        command += str(K) + " "  # 5
                        command += str(D) + " "  # 6
                        command += str(n_coupling_locs) + " "  # 7
                        command += str(coupling_loc_min) + " "  # 8
                        command += str(coupling_loc_max) + " "  # 9
                        command += str(ii) + " "  # 10
                        command += str(n_tuning_locs) + " "  # 11
                        command += str(tuning_loc_min) + " "  # 12
                        command += str(tuning_loc_max) + " "  # 13
                        command += str(jj) + " "  # 14
                        command += str(args.n_coupling_lambdas) + " "  # 15
                        command += str(args.coupling_lambda_lower) + " "  # 16
                        command += str(args.coupling_lambda_upper) + " "  # 17
                        command += str(args.n_tuning_lambdas) + " "  # 18
                        command += str(args.tuning_lambda_lower) + " "  # 19
                        command += str(args.tuning_lambda_upper) + " "  # 20
                        command += str(args.fine_sweep_frac) + " "  # 21
                        command += str(args.solver) + " "  # 22
                        command += str(args.initialization) + " "  # 23
                        command += str(args.max_iter) + " "  # 24
                        command += str(args.tol) + " "  # 25
                        command += str(coupling_rng) + " "  # 26
                        command += str(tuning_rng) + " "  # 27
                        command += str(dataset_rng) + " "  # 28
                        command += str(fitter_rng) + " "  # 29
                        command += '\n'
                        tasks.write(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--tasks_path', type=str)
    parser.add_argument('--wrapper_path', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--model_fit', default='em')
    # Fixed model hyperparameters
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
    args = parser.parse_args()

    main(args)
