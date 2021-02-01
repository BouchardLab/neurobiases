import argparse
import h5py
import numpy as np
import time
import warnings

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from neurobiases import TriangularModel
from neurobiases.solver_utils import cv_sparse_solver_single
from sklearn.exceptions import ConvergenceWarning


def main(args):
    if args.no_warn:
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    save_path = args.save_path
    model_fit = args.model_fit
    N = args.N
    M = args.M
    K = args.K
    D = args.D

    # Model hyperparameters
    coupling_locs = np.linspace(args.coupling_loc_min,
                                args.coupling_loc_max,
                                args.n_coupling_locs)
    coupling_loc = coupling_locs[args.coupling_loc_idx]
    tuning_locs = np.linspace(args.tuning_loc_min,
                              args.tuning_loc_max,
                              args.n_tuning_locs)
    tuning_loc = tuning_locs[args.tuning_loc_idx]
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

    # Process random seeds for model
    coupling_rng = args.coupling_rng
    tuning_rng = args.tuning_rng
    dataset_rng = args.dataset_rng
    # Process fitter random state
    if args.fitter_rng == -1:
        fitter_rng = None
    else:
        fitter_rng = args.fitter_rng

    # MPI communicator
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    if rank == 0:
        t0 = time.time()
        n_total_tasks = args.n_coupling_lambdas * args.n_tuning_lambdas * args.cv

    # Generate triangular model
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        M=M, N=N, K=K,
        corr_cluster=args.corr_cluster,
        corr_back=args.corr_back,
        tuning_distribution=args.tuning_distribution,
        tuning_sparsity=args.tuning_sparsity,
        tuning_loc=tuning_loc,
        tuning_scale=args.tuning_scale,
        tuning_rng=tuning_rng,
        coupling_distribution=args.coupling_distribution,
        coupling_sparsity=args.coupling_sparsity,
        coupling_loc=coupling_loc,
        coupling_scale=args.coupling_scale,
        coupling_rng=coupling_rng,
        stim_distribution='uniform'
    )
    # Generate data using seed
    X, Y, y = tm.generate_samples(n_samples=D, rng=int(dataset_rng))

    # Run coarse sweep CV
    coarse_sweep_results = \
        cv_sparse_solver_single(
            method=model_fit,
            X=X,
            Y=Y,
            y=y,
            coupling_lambdas=coupling_lambdas,
            tuning_lambdas=tuning_lambdas,
            Ks=Ks,
            cv=args.cv,
            solver=args.solver,
            initialization=args.initialization,
            max_iter=args.max_iter,
            tol=args.tol,
            refit=args.refit,
            comm=comm,
            cv_verbose=args.cv_verbose,
            fitter_verbose=args.fitter_verbose,
            mstep_verbose=args.mstep_verbose,
            fitter_rng=fitter_rng
        )
    if rank == 0:
        # Identify best hyperparameter set according to BIC
        bics = coarse_sweep_results[1]
        median_bics = np.median(bics, axis=-1)
        best_hyps = np.unravel_index(np.argmin(median_bics), median_bics.shape)
        best_c_coupling = coupling_lambdas[best_hyps[0]]
        best_c_tuning = tuning_lambdas[best_hyps[1]]
        if model_fit == 'em':
            Ks = np.array([best_hyps[2]])
        # Create new hyperparameter set
        coupling_lambda_lower = args.fine_sweep_frac * best_c_coupling
        coupling_lambda_upper = (1. / args.fine_sweep_frac) * best_c_coupling
        coupling_lambdas = np.linspace(coupling_lambda_lower,
                                       coupling_lambda_upper,
                                       num=n_coupling_lambdas)
        tuning_lambda_lower = args.fine_sweep_frac * best_c_tuning
        tuning_lambda_upper = (1. / args.fine_sweep_frac) * best_c_tuning
        tuning_lambdas = np.linspace(tuning_lambda_lower,
                                     tuning_lambda_upper,
                                     num=n_tuning_lambdas)
    # Broadcast new lambdas out
    coupling_lambdas = Bcast_from_root(coupling_lambdas, comm)
    tuning_lambdas = Bcast_from_root(tuning_lambdas, comm)
    if model_fit == 'em':
        Ks = Bcast_from_root(Ks, comm)
    # Verbosity update
    if rank == 0:
        t1 = time.time()

    # Run broad sweep CV
    fine_sweep_results = \
        cv_sparse_solver_single(
            method=model_fit,
            X=X,
            Y=Y,
            y=y,
            coupling_lambdas=coupling_lambdas,
            tuning_lambdas=tuning_lambdas,
            Ks=Ks,
            cv=args.cv,
            solver=args.solver,
            initialization=args.initialization,
            max_iter=args.max_iter,
            tol=args.tol,
            refit=args.refit,
            comm=comm,
            cv_verbose=args.cv_verbose,
            fitter_verbose=args.fitter_verbose,
            mstep_verbose=args.mstep_verbose,
            fitter_rng=fitter_rng
        )
    if model_fit == 'em':
        mlls, bics, a_est, b_est, B_est, Psi_est, L_est = fine_sweep_results
    elif model_fit == 'tc':
        mses, bics, a_est, b_est = fine_sweep_results

    if rank == 0:
        # Get best overall fit
        median_bics = np.median(bics, axis=-1)
        best_hyps = np.unravel_index(np.argmin(median_bics), median_bics.shape)
        a_est_best = a_est[best_hyps]
        b_est_best = b_est[best_hyps]
        if model_fit == 'em':
            B_est_best = B_est[best_hyps]
            Psi_est_best = Psi_est[best_hyps]
            L_est_best = L_est[best_hyps]

        # Save results
        with h5py.File(save_path, 'w') as results:
            if model_fit == 'em':
                results['mlls'] = np.squeeze(mlls[best_hyps])
            else:
                results['mses'] = np.squeeze(mses[best_hyps])
            results['bics'] = np.squeeze(bics[best_hyps])
            # True parameters
            results['a_true'] = tm.a.ravel()
            results['b_true'] = tm.b.ravel()
            results['B_true'] = tm.B
            results['Psi_true'] = tm.Psi
            results['L_true'] = tm.L
            # Estimated parameters
            results['a_est'] = a_est_best
            results['b_est'] = b_est_best
            if model_fit == 'em':
                results['B_est'] = B_est_best
                results['Psi_est'] = Psi_est_best
                results['L_est'] = L_est_best
            # CV details
            results['coupling_lambdas'] = coupling_lambdas
            results['tuning_lambdas'] = tuning_lambdas
            results.attrs['best_coupling_lambda'] = coupling_lambdas[best_hyps[0]]
            results.attrs['best_tuning_lambda'] = tuning_lambdas[best_hyps[1]]
            # Model hyperparameters
            results.attrs['model_fit'] = args.model_fit
            results.attrs['N'] = N
            results.attrs['M'] = M
            results.attrs['K'] = K
            results.attrs['D'] = D
            results.attrs['n_splits'] = args.cv
            results.attrs['coupling_distribution'] = args.coupling_distribution
            results.attrs['coupling_sparsity'] = args.coupling_sparsity
            results.attrs['coupling_loc'] = coupling_loc
            results.attrs['coupling_scale'] = args.coupling_scale
            results.attrs['tuning_distribution'] = args.tuning_distribution
            results.attrs['tuning_sparsity'] = args.tuning_sparsity
            results.attrs['tuning_loc'] = tuning_loc
            results.attrs['tuning_scale'] = args.tuning_scale
            results.attrs['corr_cluster'] = args.corr_cluster
            results.attrs['corr_back'] = args.corr_back
            results.attrs['solver'] = args.solver
            results.attrs['initialization'] = args.initialization
            results.attrs['max_iter'] = args.max_iter
            results.attrs['tol'] = args.tol
            results.attrs['coupling_rng'] = args.coupling_rng
            results.attrs['tuning_rng'] = args.tuning_rng
            results.attrs['dataset_rng'] = args.dataset_rng
            results.attrs['fitter_rng'] = fitter_rng

        t2 = time.time()
        print(
            "---------------------------------------------------------------" +
            "Job complete: Performed a single CV coarse-fine sweep." +
            f"Model: {N} coupling, {M} tuning, {K} latent, {D} samples" +
            f"Number of processes: {size}" +
            f"Total number of tasks: {n_total_tasks}" +
            f"Fine sweep centered on {best_c_coupling} (coupling) and {best_c_tuning} (tuning)." +
            f"Coarse sweep time: {t1 - t0} seconds." +
            f"Total time elapsed: {t2 - t0} seconds." +
            "---------------------------------------------------------------"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CV solver on triangular model.')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--model_fit', default='em')
    # Fixed model hyperparameters
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--D', type=int, default=1000)
    # Variable model hyperparameters
    parser.add_argument('--n_coupling_locs', type=int, default=30)
    parser.add_argument('--coupling_loc_min', type=float, default=-3)
    parser.add_argument('--coupling_loc_max', type=float, default=3)
    parser.add_argument('--coupling_loc_idx', type=int, default=0)
    parser.add_argument('--n_tuning_locs', type=int, default=30)
    parser.add_argument('--tuning_loc_min', type=float, default=-3)
    parser.add_argument('--tuning_loc_max', type=float, default=3)
    parser.add_argument('--tuning_loc_idx', type=int, default=0)
    # CV fitting hyperparameters
    parser.add_argument('--n_coupling_lambdas', type=int, default=30)
    parser.add_argument('--coupling_lambda_lower', type=float, default=-5)
    parser.add_argument('--coupling_lambda_upper', type=float, default=-2)
    parser.add_argument('--n_tuning_lambdas', type=int, default=30)
    parser.add_argument('--tuning_lambda_lower', type=float, default=-5)
    parser.add_argument('--tuning_lambda_upper', type=float, default=-2)
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
    parser.add_argument('--refit', action='store_true')
    # Random seeds
    parser.add_argument('--coupling_rng', type=int, default=1)
    parser.add_argument('--tuning_rng', type=int, default=1)
    parser.add_argument('--dataset_rng', type=int, default=1)
    parser.add_argument('--fitter_rng', type=int, default=-1)
    # Verbosity flags
    parser.add_argument('--cv_verbose', action='store_true')
    parser.add_argument('--fitter_verbose', action='store_true')
    parser.add_argument('--mstep_verbose', action='store_true')
    parser.add_argument('--no_warn', action='store_true')
    args = parser.parse_args()

    main(args)
