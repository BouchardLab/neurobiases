import numpy as np
import time

from .EMSolver import EMSolver
from .ITSFASolver import ITSFASolver
from .TCSolver import TCSolver
from .TriangularModel import TriangularModel
from sklearn.model_selection import check_cv
from sklearn.utils.extmath import cartesian


def cv_sparse_solver_single(
    method, X, Y, y, coupling_lambdas, tuning_lambdas, Ks=np.array([1]), cv=5,
    solver='ow_lbfgs', initialization='fits', max_iter=1000, tol=1e-4,
    refit=False, fitter_rng=None, comm=None, cv_verbose=False,
    fitter_verbose=False, mstep_verbose=False
):
    """Performs a cross-validated, sparse EM fit on the triangular model.

    This function is parallelized with MPI. It parallelizes coupling, tuning,
    and latent hyperparameters across cores. Fits across cross-validation folds
    are performed within a core.

    Parameters
    ----------
    X : np.ndarray, shape (D, M)
        Design matrix for tuning features.
    Y : np.ndarray, shape (D, N)
        Design matrix for coupling features.
    y : np.ndarray, shape (D, 1)
        Neural response vector.
    coupling_lambdas : np.ndarray
        The coupling sparsity penalties to apply to the optimization.
    tuning_lambdas : np.ndarray
        The tuning sparsity penalties to apply to the optimization.
    Ks : np.ndarray
        The latent factors to iterate over.
    cv : int, or cross-validation object
        The number of cross-validation folds, if int. Can also be its own
        cross-validator object.
    solver : string
        The sparse solver to use. Defaults to orthant-wise LBFGS.
    max_iter : int
        The maximum number of EM iterations.
    tol : float
        Convergence criteria for relative decrease in marginal log-likelihood.
    random_state : random state object
        Used for EM solver.
    comm : MPI communicator
        For MPI runs. If None, assumes that MPI is not used.
    verbose : bool
        If True, prints out updates during hyperparameter folds.

    Returns
    -------
    mlls : np.ndarray
        The marginal log-likelihood of the trained model on the held out data.
    a : np.ndarray
        The coupling parameters.
    b : np.ndarray
        The tuning parameters.
    B : np.ndarray
        The non-target tuning parameters.
    Psi_tr : np.ndarray
        The transformed private variances.
    """
    # Get dimensions
    M = X.shape[1]
    N = Y.shape[1]

    # Handle MPI communicators
    rank = 0
    size = 1
    if comm is not None:
        from mpi_utils.ndarray import Gatherv_rows
        rank = comm.rank
        size = comm.size

    # Get CV object
    cv = check_cv(cv=cv)
    n_splits = cv.get_n_splits()
    splits = np.arange(n_splits)

    # Assign tasks
    if method == 'em':
        hyperparameters = cartesian((coupling_lambdas,
                                     tuning_lambdas,
                                     Ks,
                                     splits))
    elif method == 'itsfa':
        hyperparameters = cartesian((Ks, splits))
    elif method == 'tc':
        hyperparameters = cartesian((coupling_lambdas, tuning_lambdas, splits))
    else:
        raise ValueError('Method is invalid.')

    tasks = np.array_split(hyperparameters, size)[rank]
    n_tasks = len(tasks)

    # Create storage arrays
    if method == 'em':
        mlls = np.zeros(n_tasks)
        bics = np.zeros(n_tasks)
        a_est = np.zeros((n_tasks, N))
        b_est = np.zeros((n_tasks, M))
        B_est = np.zeros((n_tasks, M, N))
        Psi_est = np.zeros((n_tasks, N + 1))
        L_est = np.zeros((n_tasks, Ks.max(), N + 1))
    elif method == 'itsfa':
        mses = np.zeros(n_tasks)
        bics = np.zeros(n_tasks)
        a_est = np.zeros((n_tasks, N))
        b_est = np.zeros((n_tasks, M))
        B_est = np.zeros((n_tasks, M, N))
    elif method == 'tc':
        mses = np.zeros(n_tasks)
        bics = np.zeros(n_tasks)
        a_est = np.zeros((n_tasks, N))
        b_est = np.zeros((n_tasks, M))

    # Iterate over tasks for this rank
    for task_idx, values in enumerate(tasks):
        if method == 'em':
            c_coupling, c_tuning, K_cv, split_idx = values
        elif method == 'itsfa':
            K_cv, split_idx = values
        elif method == 'tc':
            c_coupling, c_tuning, split_idx = values

        # Pull out the indices for the current fold
        train_idx, test_idx = list(cv.split(X))[int(split_idx)]
        X_train = X[train_idx]
        Y_train = Y[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]
        y_test = y[test_idx]

        if cv_verbose:
            t0 = time.time()

        if method == 'em':
            fitter = EMSolver(
                X=X_train,
                Y=Y_train,
                y=y_train,
                K=int(K_cv),
                solver=solver,
                initialization=initialization,
                max_iter=max_iter,
                tol=tol,
                c_tuning=c_tuning,
                c_coupling=c_coupling,
                rng=fitter_rng,
                fa_rng=2332).fit_em(
                    verbose=fitter_verbose,
                    mstep_verbose=mstep_verbose,
                    refit=refit
                )

            # Store parameter fits
            a_est[task_idx] = fitter.a.ravel()
            b_est[task_idx] = fitter.b.ravel()
            B_est[task_idx] = fitter.B
            Psi_est[task_idx] = fitter.Psi_tr_to_Psi()
            L_est[task_idx, :int(K_cv), :] = fitter.L
            # Score the resulting fit
            mlls[task_idx] = fitter.marginal_log_likelihood(
                X=X_test,
                Y=Y_test,
                y=y_test)
            # Calculate BIC
            bics[task_idx] = fitter.bic()
        elif method == 'itsfa':
            raise NotImplementedError()
        elif method == 'tc':
            fitter = TCSolver(
                X=X_train,
                Y=Y_train,
                y=y_train,
                solver=solver,
                c_tuning=c_tuning,
                c_coupling=c_coupling,
                initialization=initialization,
                max_iter=max_iter,
                tol=tol,
                rng=fitter_rng).fit_lasso(
                    refit=refit,
                    verbose=fitter_verbose
                )
            # Store parameter fits
            a_est[task_idx] = fitter.a.ravel()
            b_est[task_idx] = fitter.b.ravel()
            # Score the resulting fit
            mses[task_idx] = fitter.mse(X=X_test, Y=Y_test, y=y_test)
            # Calculate BIC
            bics[task_idx] = fitter.bic()

        if cv_verbose:
            elapsed = time.time() - t0
            print(f"Rank {rank}, task {task_idx+1}/{len(tasks)} complete. "
                  f"Elapsed time: {elapsed:0.2f}. "
                  f"Coupling lambda: {c_coupling:0.3E}. "
                  f"Tuning lambda: {c_tuning:0.3E}. "
                  f"Split idx: {split_idx}.")

    if comm is not None:
        # Gather tasks across all storage arrays
        if method == 'em':
            mlls = Gatherv_rows(mlls, comm)
            bics = Gatherv_rows(bics, comm)
            a_est = Gatherv_rows(a_est, comm)
            b_est = Gatherv_rows(b_est, comm)
            B_est = Gatherv_rows(B_est, comm)
            Psi_est = Gatherv_rows(Psi_est, comm)
            L_est = Gatherv_rows(L_est, comm)
        elif method == 'itsfa':
            raise NotImplementedError()
        elif method == 'tc':
            mses = Gatherv_rows(mses, comm)
            bics = Gatherv_rows(bics, comm)
            a_est = Gatherv_rows(a_est, comm)
            b_est = Gatherv_rows(b_est, comm)

        if rank == 0:
            if method == 'em':
                reshape = [coupling_lambdas.size,
                           tuning_lambdas.size,
                           Ks.size,
                           splits.size]
                mlls.shape = reshape
                B_est.shape = reshape + [M, N]
                Psi_est.shape = reshape + [-1]
                L_est.shape = reshape + [Ks.max(), N + 1]
            elif method == 'itsfa':
                raise NotImplementedError()
            elif method == 'tc':
                reshape = [coupling_lambdas.size,
                           tuning_lambdas.size,
                           splits.size]
                mses.shape = reshape
            a_est.shape = reshape + [-1]
            b_est.shape = reshape + [-1]
            bics.shape = reshape

    if method == 'em':
        return mlls, bics, a_est, b_est, B_est, Psi_est, L_est
    elif method == 'itsfa':
        return mses, bics, a_est, b_est, B_est
    elif method == 'tc':
        return mses, bics, a_est, b_est


def cv_solver_full(
    method, selection, M, N, K, D, coupling_distribution,
    coupling_sparsities, coupling_locs, coupling_scale, coupling_rngs,
    tuning_distribution, tuning_sparsities, tuning_locs, tuning_scale,
    tuning_rngs, corr_clusters, corr_back, dataset_rngs, coupling_lambdas,
    tuning_lambdas, Ks=np.array([1]), cv=5, solver='ow_lbfgs',
    initialization='fits', max_iter=1000, tol=1e-4, refit=True, fitter_rng=None,
    comm=None, cv_verbose=False, fitter_verbose=False, mstep_verbose=False,
    lightweight=False
):
    """Performs a cross-validated, sparse EM fit on data generated from a TM.

    This function parallelizes across TM hyperparameters, model instantiations,
    dataset instantiations, cross-validation fold, and cross-validation
    hyperparameters.

    Parameters
    ----------
    M : int
        The number of tuning parameters.
    N : int
        The number of coupling parameters.
    K : int
        The number of latent factors.
    D : int
        The dataset size.
    coupling_distribution : string
        The distribution from which to draw the coupling parameters.
    coupling_sparsities : np.ndarray
        An array of coupling sparsities over which TMs are instantiated.
    coupling_locs : np.ndarray
        An array of coupling means over which TMs are instantiated.
    coupling_scale : float
        The scale parameter with which to instantiate TM coupling parameters.
        This does not vary across model hyperparameters.
    coupling_rngs : np.ndarray
        An array of ints with which to seed the TM coupling parameters. These
        guarantee that the model can be reproduced.
    tuning_distribution : string
        The distribution from which to draw the tuning parameters.
    tuning_sparsities : np.ndarray
        An array of tuning sparsities over which TMs are instantiated.
    tuning_locs : np.ndarray
        An array of tuning means over which TMs are instantiated.
    tuning_scale : float
        The scale parameter with which to instantiate TM tuning parameters.
        This does not vary across model hyperparameters.
    tuning_rngs : np.ndarray
        An array of ints with which to seed the TM tuning parameters. These
        guarantee that the model can be reproduced.
    coupling_lambdas : np.ndarray
        The coupling sparsity penalties to apply to each optimization.
    tuning_lambdas : np.ndarray
        The tuning sparsity penalties to apply to each optimization.
    Ks : np.ndarray
        The latent factors to cross-validate over for fitting.
    cv : int, or cross-validation object
        The number of cross-validation folds, or a cross-validation object.
    solver : string
        The sparse solver to use. Defaults to orthant-wise LBFGS.
    initialization : string
        The type of procedure to use for initializing the parameters.
    max_iter : int
        The maximum number of EM iterations.
    tol : float
        The convergence criteria for coordinate descent.
    refit : bool
        Whether to refit the optimization without the sparse solver. If using
        oracle selection, this option is ignored.
    fitter_rng : {None, int, array_like[ints], SeedSequence, BitGenerator,
                  Generator}
        The random seed or generator for the fitting algorithm.
    comm : MPI communicator
        The MPI communicator. If None, assumes that MPI is not used.
    cv_verbose : bool
        Verbosity flag for the CV folds.
    fitter_verbose : bool
        Verbosity flag for the solver.
    mstep_verbose : bool
        Only for EM inference. Verbosity flag for the solver during the M-step.
    lightweight : bool
        If True, only a small portion of results will be returned.

    Returns
    -------
    mlls : np.ndarray
        The marginal log-likelihood of the trained model on the held out data.
        Only for EM option.
    mses : np.ndarray
        The mean-squared error on the held-out data, using only tuning and
        coupling parameters. Only for TC and ITSFA options.
    bics : np.ndarray
        The BIC on the training data.
    a : np.ndarray
        The coupling parameters.
    a_est : np.ndarray
        The estimated coupling parameters.
    b : np.ndarray
        The tuning parameters.
    b_est : np.ndarray
        The estimated tuning parameters.
    B : np.ndarray
        The non-target tuning parameters.
    B_est : np.ndarray
        The estimated non-target tuning parameters. Only for EM and ITSFA.
    Psi : np.ndarray
        The private variances.
    Psi_est : np.ndarray
        The estimated private variances. Only for EM.
    L : np.ndarray
        The latent factors.
    L_est : np.ndarray
        The estimated latent factors. Only for EM.
    """
    # Handle MPI communicators
    rank = 0
    size = 1
    if comm is not None:
        from mpi_utils.ndarray import Gatherv_rows
        rank = comm.rank
        size = comm.size

    # Number of models
    model_idxs = np.arange(tuning_rngs.size)
    # Get CV object
    cv = check_cv(cv=cv)
    n_splits = cv.get_n_splits()
    # Assign tasks
    splits = np.arange(n_splits)

    # Set up hyperparameters
    if method == 'em':
        if selection == 'sparse':
            hyperparameters = cartesian(
                (tuning_sparsities,
                 tuning_locs,
                 coupling_sparsities,
                 coupling_locs,
                 model_idxs,
                 corr_clusters,
                 dataset_rngs,
                 splits,
                 coupling_lambdas,
                 tuning_lambdas,
                 Ks)
            )
        elif selection == 'oracle':
            hyperparameters = cartesian(
                (tuning_sparsities,
                 tuning_locs,
                 coupling_sparsities,
                 coupling_locs,
                 model_idxs,
                 corr_clusters,
                 dataset_rngs,
                 splits,
                 Ks)
            )
    elif method == 'itsfa':
        if selection == 'sparse':
            raise NotImplementedError()
        elif selection == 'oracle':
            hyperparameters = cartesian(
                (tuning_sparsities,
                 tuning_locs,
                 coupling_sparsities,
                 coupling_locs,
                 model_idxs,
                 corr_clusters,
                 dataset_rngs,
                 splits,
                 Ks)
            )
    elif method == 'tc':
        if selection == 'sparse':
            hyperparameters = cartesian(
                (tuning_sparsities,
                 tuning_locs,
                 coupling_sparsities,
                 coupling_locs,
                 model_idxs,
                 corr_clusters,
                 dataset_rngs,
                 splits,
                 coupling_lambdas,
                 tuning_lambdas)
            )
        elif selection == 'oracle':
            hyperparameters = cartesian(
                (tuning_sparsities,
                 tuning_locs,
                 coupling_sparsities,
                 coupling_locs,
                 model_idxs,
                 corr_clusters,
                 dataset_rngs,
                 splits)
            )
    tasks = np.array_split(hyperparameters, size)[rank]
    n_tasks = len(tasks)

    # Create storage arrays
    if method == 'em':
        mlls = np.zeros(n_tasks)
        bics = np.zeros(n_tasks)
        a = np.zeros((n_tasks, N))
        a_est = np.zeros((n_tasks, N))
        b = np.zeros((n_tasks, M))
        b_est = np.zeros((n_tasks, M))
        B = np.zeros((n_tasks, M, N))
        B_est = np.zeros((n_tasks, M, N))
        Psi = np.zeros((n_tasks, N + 1))
        Psi_est = np.zeros((n_tasks, N + 1))
        L = np.zeros((n_tasks, K, N + 1))
        L_est = np.zeros((n_tasks, Ks.max(), N + 1))
    elif method == 'itsfa':
        mses = np.zeros(n_tasks)
        bics = np.zeros(n_tasks)
        a = np.zeros((n_tasks, N))
        a_est = np.zeros((n_tasks, N))
        b = np.zeros((n_tasks, M))
        b_est = np.zeros((n_tasks, M))
        B = np.zeros((n_tasks, M, N))
        B_est = np.zeros((n_tasks, M, N))
        Psi = np.zeros((n_tasks, N + 1))
        L = np.zeros((n_tasks, K, N + 1))
    elif method == 'tc':
        mses = np.zeros(n_tasks)
        bics = np.zeros(n_tasks)
        a = np.zeros((n_tasks, N))
        a_est = np.zeros((n_tasks, N))
        b = np.zeros((n_tasks, M))
        b_est = np.zeros((n_tasks, M))
        if not lightweight:
            B = np.zeros((n_tasks, M, N))
            Psi = np.zeros((n_tasks, N + 1))
            L = np.zeros((n_tasks, K, N + 1))

    # Iterate over tasks for this rank
    for task_idx, values in enumerate(tasks):
        if method == 'em':
            if selection == 'sparse':
                (tuning_sparsity, tuning_loc, coupling_sparsity, coupling_loc,
                 model_idx, corr_cluster, dataset_rng, split_idx, c_coupling,
                 c_tuning, K_cv) = values
            elif selection == 'oracle':
                (tuning_sparsity, tuning_loc, coupling_sparsity, coupling_loc,
                 model_idx, corr_cluster, dataset_rng, split_idx, K_cv) = values
        elif method == 'itsfa':
            if selection == 'sparse':
                raise NotImplementedError()
            elif selection == 'oracle':
                (tuning_sparsity, tuning_loc, coupling_sparsity, coupling_loc,
                 model_idx, corr_cluster, dataset_rng, split_idx, K_cv) = values
        elif method == 'tc':
            if selection == 'sparse':
                (tuning_sparsity, tuning_loc, coupling_sparsity, coupling_loc,
                 model_idx, corr_cluster, dataset_rng, split_idx, c_coupling,
                 c_tuning) = values
            elif selection == 'oracle':
                (tuning_sparsity, tuning_loc, coupling_sparsity, coupling_loc,
                 model_idx, corr_cluster, dataset_rng, split_idx) = values

        if cv_verbose:
            print(f'Rank {rank}, Task {task_idx}')

        # Generate triangular model
        tm = TriangularModel(
            model='linear',
            parameter_design='direct_response',
            M=M, N=N, K=K,
            corr_cluster=corr_cluster,
            corr_back=corr_back,
            tuning_distribution=tuning_distribution,
            tuning_sparsity=tuning_sparsity,
            tuning_loc=tuning_loc,
            tuning_scale=tuning_scale,
            tuning_rng=tuning_rngs[int(model_idx)],
            coupling_distribution=coupling_distribution,
            coupling_sparsity=coupling_sparsity,
            coupling_loc=coupling_loc,
            coupling_scale=coupling_scale,
            coupling_rng=coupling_rngs[int(model_idx)],
            stim_distribution='uniform'
        )
        # Store true parameters
        a[task_idx] = tm.a.ravel()
        b[task_idx] = tm.b.ravel()
        if not lightweight:
            B[task_idx] = tm.B
            Psi[task_idx] = tm.Psi.ravel()
            L[task_idx, :int(K), :] = tm.L

        # Generate data using seed
        X, Y, y = tm.generate_samples(n_samples=D, rng=int(dataset_rng))
        # Pull out the indices for the current fold
        train_idx, test_idx = list(cv.split(X))[int(split_idx)]
        X_train = X[train_idx]
        Y_train = Y[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]
        y_test = y[test_idx]

        # Run the sparse fitter
        if method == 'em':
            if selection == 'oracle':
                fitter = EMSolver(
                    X=X_train,
                    Y=Y_train,
                    y=y_train,
                    K=int(K_cv),
                    a_mask=tm.a != 0,
                    b_mask=tm.b != 0,
                    B_mask=tm.B != 0,
                    solver=solver,
                    initialization=initialization,
                    max_iter=max_iter,
                    tol=tol,
                    rng=fitter_rng,
                    fa_rng=2332).fit_em(
                        verbose=fitter_verbose,
                        mstep_verbose=mstep_verbose,
                        refit=False
                    )
            elif selection == 'sparse':
                fitter = EMSolver(
                    X=X_train,
                    Y=Y_train,
                    y=y_train,
                    K=int(K_cv),
                    solver=solver,
                    initialization=initialization,
                    max_iter=max_iter,
                    tol=tol,
                    c_tuning=c_tuning,
                    c_coupling=c_coupling,
                    rng=fitter_rng,
                    fa_rng=2332).fit_em(
                        verbose=fitter_verbose,
                        mstep_verbose=mstep_verbose,
                        refit=refit
                    )

            # Store parameter fits
            a_est[task_idx] = fitter.a.ravel()
            b_est[task_idx] = fitter.b.ravel()
            B_est[task_idx] = fitter.B
            Psi_est[task_idx] = fitter.Psi_tr_to_Psi()
            L_est[task_idx, :int(K), :] = fitter.L
            # Score the resulting fit
            mlls[task_idx] = fitter.marginal_log_likelihood(
                X=X_test,
                Y=Y_test,
                y=y_test
            )
            # Calculate BIC
            bics[task_idx] = fitter.bic()

        elif method == 'itsfa':
            if selection == 'sparse':
                raise NotImplementedError()
            elif selection == 'oracle':
                fitter = ITSFASolver(
                    X=X_train,
                    Y=Y_train,
                    y=y_train,
                    K=K_cv,
                    B=None,
                    a_mask=tm.a.ravel() != 0,
                    b_mask=tm.b.ravel() != 0,
                    max_iter=max_iter,
                    tol=tol,
                    fa_max_iter=10000,
                    fa_tol=1e-4
                ).fit_itsfa()

            # Store parameter fits
            a_est[task_idx] = fitter.a.ravel()
            b_est[task_idx] = fitter.b.ravel()
            # Score the resulting fit
            mses[task_idx] = fitter.mse(X=X_test, Y=Y_test, y=y_test)
            # Calculate BIC
            bics[task_idx] = fitter.bic()

        elif method == 'tc':
            if selection == 'sparse':
                # Run the sparse fitter
                fitter = TCSolver(
                    X=X_train,
                    Y=Y_train,
                    y=y_train,
                    solver=solver,
                    c_tuning=c_tuning,
                    c_coupling=c_coupling,
                    initialization=initialization,
                    max_iter=max_iter,
                    tol=tol,
                    rng=fitter_rng).fit_lasso(
                        refit=refit,
                        verbose=fitter_verbose
                    )
            elif selection == 'oracle':
                fitter = TCSolver(
                    X=X_train,
                    Y=Y_train,
                    y=y_train,
                    a_mask=tm.a.ravel() != 0,
                    b_mask=tm.b.ravel() != 0,
                    solver=solver,
                    initialization=initialization,
                    max_iter=max_iter,
                    tol=tol,
                    rng=fitter_rng).fit_ols()

            # Store parameter fits
            a_est[task_idx] = fitter.a.ravel()
            b_est[task_idx] = fitter.b.ravel()
            if not lightweight:
                B_est[task_idx] = fitter.B
            # Score the resulting fit
            mses[task_idx] = fitter.mse(X=X_test, Y=Y_test, y=y_test)
            # Calculate BIC
            bics[task_idx] = fitter.bic()

    if comm is not None:
        # Gather tasks across all storage arrays
        if method == 'em':
            mlls = Gatherv_rows(mlls, comm)
            bics = Gatherv_rows(bics, comm)
            a = Gatherv_rows(a, comm)
            a_est = Gatherv_rows(a_est, comm)
            b = Gatherv_rows(b, comm)
            b_est = Gatherv_rows(b_est, comm)
            B = Gatherv_rows(B, comm)
            B_est = Gatherv_rows(B_est, comm)
            Psi = Gatherv_rows(Psi, comm)
            Psi_est = Gatherv_rows(Psi_est, comm)
            L = Gatherv_rows(L, comm)
            L_est = Gatherv_rows(L_est, comm)
        elif method == 'itsfa':
            raise NotImplementedError()
        elif method == 'tc':
            mses = Gatherv_rows(mses, comm)
            bics = Gatherv_rows(bics, comm)
            a = Gatherv_rows(a, comm)
            a_est = Gatherv_rows(a_est, comm)
            b = Gatherv_rows(b, comm)
            b_est = Gatherv_rows(b_est, comm)
            if not lightweight:
                B = Gatherv_rows(B, comm)
                Psi = Gatherv_rows(Psi, comm)
                L = Gatherv_rows(L, comm)

        if rank == 0:
            # Reshape arrays
            if method == 'em':
                if selection == 'sparse':
                    reshape = [
                        tuning_sparsities.size,
                        tuning_locs.size,
                        coupling_sparsities.size,
                        coupling_locs.size,
                        model_idxs.size,
                        corr_clusters.size,
                        dataset_rngs.size,
                        splits.size,
                        coupling_lambdas.size,
                        tuning_lambdas.size,
                        Ks.size
                    ]
                elif selection == 'oracle':
                    reshape = [
                        tuning_sparsities.size,
                        tuning_locs.size,
                        coupling_sparsities.size,
                        coupling_locs.size,
                        model_idxs.size,
                        corr_clusters.size,
                        dataset_rngs.size,
                        splits.size,
                        Ks.size
                    ]
            elif method == 'tc':
                if selection == 'sparse':
                    reshape = [
                        tuning_sparsities.size,
                        tuning_locs.size,
                        coupling_sparsities.size,
                        coupling_locs.size,
                        model_idxs.size,
                        corr_clusters.size,
                        dataset_rngs.size,
                        splits.size,
                        coupling_lambdas.size,
                        tuning_lambdas.size,
                    ]
                elif selection == 'oracle':
                    reshape = [
                        tuning_sparsities.size,
                        tuning_locs.size,
                        coupling_sparsities.size,
                        coupling_locs.size,
                        model_idxs.size,
                        corr_clusters.size,
                        dataset_rngs.size,
                        splits.size
                    ]

            if method == 'em':
                mlls.shape = reshape
                B_est.shape = reshape + [M, N]
                Psi_est.shape = reshape + [-1]
                L_est.shape = reshape + [Ks.max(), N + 1]
            elif method == 'tc':
                mses.shape = reshape

            bics.shape = reshape
            a.shape = reshape + [-1]
            a_est.shape = reshape + [-1]
            b.shape = reshape + [-1]
            b_est.shape = reshape + [-1]
            if not lightweight:
                B.shape = reshape + [M, N]
                Psi.shape = reshape + [-1]
                L.shape = reshape + [Ks.max(), N + 1]

    if method == 'em':
        return mlls, bics, a, a_est, b, b_est, B, B_est, Psi, Psi_est, L, L_est
    elif method == 'itsfa':
        return mses, bics, a, a_est, b, b_est, B, B_est, Psi, L
    elif method == 'tc':
        if lightweight:
            return mses, bics, a, a_est, b, b_est
        else:
            return mses, bics, a, a_est, b, b_est, B, Psi, L
    else:
        raise ValueError("Incorrect method specified.")
