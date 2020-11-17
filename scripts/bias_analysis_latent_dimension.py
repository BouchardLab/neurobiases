import argparse
import numpy as np
import time

from neurobiases import TCSolver
from neurobiases import TriangularModel


def main(args):
    save_path = args.save_path
    N = args.N
    M = args.M
    D = args.D
    n_models = args.n_models
    n_datasets = args.n_datasets
    n_Ks = args.n_Ks
    Ks = 1 + np.arange(n_Ks)
    a_trues = np.zeros((n_models, N))
    b_trues = np.zeros((n_models, M))
    a_hats = np.zeros((n_models, n_Ks, n_datasets, N))
    b_hats = np.zeros((n_models, n_Ks, n_datasets, M))
    t0 = time.time()

    # create triangular models
    for model in range(n_models):
        print(f'Model {model}')
        # Generate triangular model
        tm = TriangularModel(
            model='linear',
            parameter_design='direct_response',
            M=M, N=N, K=1,
            tuning_distribution=args.tuning_distribution,
            tuning_sparsity=args.tuning_sparsity,
            tuning_loc=args.tuning_loc,
            tuning_scale=args.tuning_scale,
            tuning_random_state=None,
            coupling_distribution=args.coupling_distribution,
            coupling_sparsity=args.coupling_sparsity,
            coupling_loc=args.coupling_loc,
            coupling_scale=args.coupling_scale,
            coupling_random_state=None,
            coupling_sum=None,
            corr_cluster=args.corr_cluster,
            corr_back=args.corr_back)
        a_trues[model] = tm.a.ravel()
        b_trues[model] = tm.b.ravel()
        # Iterate over noise correlations
        for idx, K in enumerate(Ks):
            tm.K = K
            tm.generate_noise_structure()
            # Iterate over datasets
            for dataset in range(n_datasets):
                a_mask, b_mask, _ = tm.get_masks()
                X, Y, y = tm.generate_samples(n_samples=D, random_state=dataset)
                solver = TCSolver(X, Y, y, a_mask, b_mask).fit_ols()
                a_hats[model, idx, dataset], b_hats[model, idx, dataset] = \
                    solver.a, solver.b

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
    parser.add_argument('--D', type=int, default=1000)
    parser.add_argument('--n_models', type=int, default=50)
    parser.add_argument('--n_datasets', type=int, default=30)
    parser.add_argument('--n_Ks', type=int, default=10)
    parser.add_argument('--tuning_distribution', default='gaussian')
    parser.add_argument('--tuning_sparsity', type=float, default=0.5)
    parser.add_argument('--tuning_loc', type=float, default=0.)
    parser.add_argument('--tuning_scale', type=float, default=1.)
    parser.add_argument('--coupling_distribution', default='gaussian')
    parser.add_argument('--coupling_sparsity', type=float, default=0.5)
    parser.add_argument('--coupling_loc', type=float, default=0.)
    parser.add_argument('--coupling_scale', type=float, default=1.)
    parser.add_argument('--corr_cluster', type=float, default=0.25)
    parser.add_argument('--corr_back', type=float, default=0.0)

    args = parser.parse_args()

    main(args)
