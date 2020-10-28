import argparse
import numpy as np
import time

from neurobiases import TCSolver
from neurobiases import TriangularModel


def main(args):
    save_path = args.save_path
    N = args.N
    M = args.M
    K = args.K
    D = args.D
    n_models = args.n_models
    n_datasets = args.n_datasets
    n_tuning_locs = args.n_tuning_locs
    tuning_locs = np.linspace(args.tuning_loc_min,
                              args.tuning_loc_max,
                              n_tuning_locs)
    a_trues = np.zeros((n_tuning_locs, n_models, N))
    b_trues = np.zeros((n_tuning_locs, n_models, M))
    a_hats = np.zeros((n_tuning_locs, n_models, n_datasets, N))
    b_hats = np.zeros((n_tuning_locs, n_models, n_datasets, M))
    t0 = time.time()

    # create triangular models
    for idx, tuning_loc in enumerate(tuning_locs):
        print(f'Hyperparameter {idx}')
        for model in range(n_models):
            # Generate triangular model
            tm = TriangularModel(
                model='linear',
                parameter_design='direct_response',
                M=M, N=N, K=K,
                tuning_distribution=args.tuning_distribution,
                tuning_sparsity=args.tuning_sparsity,
                tuning_loc=tuning_loc,
                tuning_scale=args.tuning_scale,
                tuning_random_state=None,
                coupling_distribution=args.coupling_distribution,
                coupling_sparsity=args.coupling_sparsity,
                coupling_loc=args.coupling_loc,
                coupling_scale=args.coupling_scale,
                coupling_random_state=10282020,
                coupling_sum=None,
                corr_cluster=0.25,
                corr_back=0.0)

            a_trues[idx, model] = tm.a.ravel()
            b_trues[idx, model] = tm.b.ravel()

            # Iterate over datasets
            for dataset in range(n_datasets):
                a_mask, b_mask, _ = tm.get_masks()
                X, Y, y = tm.generate_samples(n_samples=D, random_state=dataset)
                solver = TCSolver(X, Y, y, a_mask, b_mask)
                a_hats[idx, model, dataset], b_hats[idx, model, dataset] = solver.fit_ols()

    np.savez(save_path,
             a_trues=a_trues,
             b_trues=b_trues,
             a_hats=a_hats,
             b_hats=b_hats)
    print('Successfully Saved.')
    t2 = time.time()
    print('Job complete. Total time: ', t2 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--N', type=int, default=15)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--n_models', type=int, default=50)
    parser.add_argument('--n_datasets', type=int, default=30)
    parser.add_argument('--n_tuning_locs', type=int, default=30)
    parser.add_argument('--tuning_distribution', default='gaussian')
    parser.add_argument('--tuning_sparsity', type=float, default=0.5)
    parser.add_argument('--tuning_loc_min', type=float, default=-3)
    parser.add_argument('--tuning_loc_max', type=float, default=3)
    parser.add_argument('--tuning_scale', type=float, default=1.)
    parser.add_argument('--coupling_distribution', default='gaussian')
    parser.add_argument('--coupling_sparsity', type=float, default=0.5)
    parser.add_argument('--coupling_loc', type=float, default=0.)
    parser.add_argument('--coupling_scale', type=float, default=1.)

    args = parser.parse_args()

    main(args)
