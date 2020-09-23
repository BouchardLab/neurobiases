import numpy as np
import torch

from . import EMSolver
from .utils import inv_softplus
from sklearn.model_selection import check_cv
from sklearn.utils.extmath import cartesian


def marginal_log_likelihood_linear_tm(
    X, Y, y, a, b, B, L, Psi, a_mask=None, b_mask=None, B_mask=None
):
    """Calculates the marginal log-likelihood of a parameter set under data
    generated from the linear triangular model.

    Parameters
    ----------
    X : np.ndarray, shape (D, M)
        Design matrix for tuning features.
    Y : np.ndarray, shape (D, N)
        Design matrix for coupling features.
    y : np.ndarray, shape (D, 1)
        Neural response vector.
    a : np.ndarray, shape (N,)
        The coupling parameters.
    b : np.ndarray, shape (N,)
        The target tuning parameters.
    B : np.ndarray, shape (M, N)
        The non-target tuning parameters.
    L : np.ndarray, shape (K, N + 1)
        The latent factors
    Psi : np.ndarray, shape (N + 1,)
        The private variances.
    a_mask : np.ndarray, shape (N,)
        Mask for coupling features.
    b_mask : nd-array, shape (M,)
        Mask for tuning features.
    B_mask : nd-array, shape (N, M)
        Mask for non-target neuron tuning features.

    Returns
    -------
    ll : float
        The marginal log-likelihood according to the linear triangular model.
    """
    D, M = X.shape
    N = Y.shape[1]

    # dimensionality checks
    if a.ndim == 2:
        a = a.ravel()
    if b.ndim == 2:
        b = b.ravel()
    # check masks
    if a_mask is None:
        a_mask = np.ones(N)
    elif a_mask.ndim == 2:
        a_mask = a_mask.ravel()

    if b_mask is None:
        b_mask = np.ones(M)
    elif b_mask.ndim == 2:
        b_mask = np.ones(M)

    # apply masks
    a = a * a_mask
    b = b * b_mask

    # split up into target and non-target components
    l_t, L_nt = np.split(L, [1], axis=1)
    l_t = l_t.ravel()
    Psi_t, Psi_nt = np.split(Psi, [1])
    Psi_t = Psi_t.item()

    # mean and covariance matrices of the gaussian expression
    mu = np.zeros((D, N + 1))
    sigma = np.zeros((N + 1, N + 1))

    # calculate mean of marginal
    mu[:, 0] = X @ (b + B @ a)
    mu[:, 1:] = X @ B

    # combine data matrices
    Y_all = np.concatenate((y, Y), axis=1)

    # useful terms to store for later
    coupled_L = l_t + L_nt @ a
    cross_coupling = Psi_nt * a + L_nt.T @ coupled_L

    # fill covariance matrix
    sigma[0, 0] = Psi_t + Psi_nt @ a**2 + coupled_L @ coupled_L
    sigma[1:, 0] = cross_coupling
    sigma[0, 1:] = cross_coupling
    sigma[1:, 1:] = np.diag(Psi_nt) + L_nt.T @ L_nt

    # calculate log-likelihood
    residual = Y_all - mu
    ll = -D / 2. * np.linalg.slogdet(sigma)[1] \
        + -0.5 * np.sum(residual.T * np.linalg.solve(sigma, residual.T))
    return ll


def Psi_tr_to_Psi(Psi_tr, transform):
    """Takes transformed Psi back to Psi.

    Parameters
    ----------
    Psi_tr : np.ndarray
        Transfored Psi.

    Returns
    -------
    Psi : np.ndarray
        The original Psi.
    """
    if transform == 'softplus':
        if isinstance(Psi_tr, np.ndarray):
            Psi = np.logaddexp(0., Psi_tr)
        elif isinstance(Psi_tr, torch.Tensor):
            Psi = torch.logaddexp(torch.tensor(0, dtype=Psi_tr.dtype), Psi_tr)
        else:
            raise ValueError('Invalid type for Psi transform.')
    elif transform == 'exp':
        if isinstance(Psi_tr, np.ndarray):
            Psi = np.exp(Psi_tr)
        elif isinstance(Psi_tr, torch.Tensor):
            Psi = torch.log(Psi_tr)
        else:
            raise ValueError('Invalid type for Psi transform.')
    else:
        raise ValueError('Invalid Psi transform.')
    return Psi


def Psi_to_Psi_tr(Psi, transform):
    """Takes Psi to the transformed Psi.

    Parameters
    ----------
    Psi : np.ndarray
        The original Psi.

    Returns
    -------
    Psi_tr : np.ndarray
        Transfored Psi.
    """
    if transform == 'softplus':
        if isinstance(Psi, np.ndarray):
            Psi_tr = inv_softplus(Psi)
        elif isinstance(Psi, torch.Tensor):
            Psi_tr = Psi + torch.log(1 - torch.exp(-Psi))
        else:
            raise ValueError('Invalid type for Psi transform.')
    elif transform == 'exp':
        if isinstance(Psi, np.ndarray):
            Psi_tr = np.log(Psi)
        elif isinstance(Psi, torch.Tensor):
            Psi_tr = torch.log(Psi)
        else:
            raise ValueError('Invalid type for Psi transform.')
    else:
        raise ValueError('Invalid Psi transform.')
    return Psi_tr


def marginal_log_likelihood_linear_tm_wrapper(X, Y, y, tm):
    """Calculates the marginal log-likelihood of the parameters in a
    TriangularModel instance under data generated from the linear triangular
    model.

    Parameters
    ----------
    X : np.ndarray, shape (D, M)
        Design matrix for tuning features.
    Y : np.ndarray, shape (D, N)
        Design matrix for coupling features.
    y : np.ndarray, shape (D, 1)
        Neural response vector.
    tm : TriangularModel instance
        A TriangularModel instance.

    Returns
    -------
    ll : float
        The marginal log-likelihood according to the linear triangular model.
    """
    ll = marginal_log_likelihood_linear_tm(
        X=X, Y=Y, y=y, a=tm.a, b=tm.b, B=tm.B, L=tm.L, Psi=tm.Psi
    )
    return ll


def fista(f_df, params, lr, C0=0., C1=0., zero_start=-1, zero_end=-1,
          one_start=-1, one_end=-1, args=None, max_iter=250, tol=1e-8,
          verbose=False):
    if args is None:
        args = tuple()
    yt = params.copy()
    xtm = params.copy()
    t = 1.
    loss = None
    sl0 = slice(0, 0)
    sl1 = slice(0, 0)
    if C0 > 0.:
        sl0 = slice(zero_start, zero_end)
    if C1 > 0.:
        sl1 = slice(one_start, one_end)

    for ii in range(max_iter):
        lossp, grad = f_df(yt, *args)
        losst = lossp
        if C0 > 0.:
            losst = losst + C0 * np.sum(abs(yt[sl0]))
        if C1 > 0.:
            losst = losst + C1 * np.sum(abs(yt[sl1]))
        if loss is not None:
            if (loss - losst) / max(1., max(abs(loss), abs(losst))) < tol:
                break
        else:
            losso = losst
        loss = losst
        xt = yt - grad * lr
        if C0 > 0.:
            xt[sl0] = np.maximum(abs(xt[sl0]) - C0 * lr, 0.) * np.sign(xt[sl0])
        if C1 > 0.:
            xt[sl1] = np.maximum(abs(xt[sl1]) - C1 * lr, 0.) * np.sign(xt[sl1])
        t = 0.5 * (1. + np.sqrt(1. + 4 * t**2))
        yt = xt + (t - 1.) * (xt - xtm) / t
        xtm = xt
    if verbose:
        string = 'M step stopped on iteration {} of {} with loss {} and initial loss {}.'
        print(string.format(ii+1, max_iter, losst, losso))
    return yt


def cv_sparse_em_solver(
    X, Y, y, coupling_lambdas, tuning_lambdas, Ks, cv=5,
    solver='ow_lbfgs', max_iter=1000, tol=1e-4, random_state=None,
    comm=None, cv_verbose=False, em_verbose=False, mstep_verbose=False
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
    scores : np.ndarray
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
    # dimensions
    M = X.shape[1]
    N = Y.shape[1]

    # handle MPI communicators
    rank = 0
    size = 1
    if comm is not None:
        from mpi_utils.ndarray import Gatherv_rows
        rank = comm.rank
        size = comm.size

    # get cv objects
    cv = check_cv(cv=cv)
    n_splits = cv.get_n_splits()
    # assign tasks
    hyperparameters = cartesian((coupling_lambdas, tuning_lambdas, Ks))
    tasks = np.array_split(hyperparameters, size)[rank]
    n_tasks = len(tasks)
    # create storage
    scores = np.zeros((n_tasks, n_splits))
    a = np.zeros((n_tasks, n_splits, N))
    b = np.zeros((n_tasks, n_splits, M))
    B = np.zeros((n_tasks, n_splits, M, N))
    Psi = np.zeros((n_tasks, n_splits, N + 1))
    L = np.zeros((n_tasks, n_splits, Ks.max(), N + 1))

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # get training set
        X_train = X[train_idx]
        Y_train = Y[train_idx]
        y_train = y[train_idx]
        # get test set
        X_test = X[test_idx]
        Y_test = Y[test_idx]
        y_test = y[test_idx]

        # iterate over hyperparameters
        for task_idx, (c_coupling, c_tuning, K) in enumerate(tasks):
            if cv_verbose:
                print(f'Rank {rank}: fold = {split_idx + 1}',
                      f'coupling = {c_coupling}, tuning = {c_tuning}, K = {K}')
            # run the sparse fitter
            emfit = EMSolver.EMSolver(
                X=X_train, Y=Y_train, y=y_train,
                K=int(K),
                solver=solver,
                max_iter=max_iter,
                tol=tol,
                c_tuning=c_tuning,
                c_coupling=c_coupling,
                random_state=random_state).fit_em(
                    verbose=em_verbose,
                    mstep_verbose=mstep_verbose)
            # store parameter fits
            a[task_idx, split_idx] = emfit.a.ravel()
            b[task_idx, split_idx] = emfit.b.ravel()
            B[task_idx, split_idx] = emfit.B
            Psi[task_idx, split_idx] = emfit.Psi_tr_to_Psi()
            L[task_idx, split_idx, :int(K), :] = emfit.L
            # score the resulting fit
            scores[task_idx, split_idx] = emfit.marginal_log_likelihood(
                X=X_test, Y=Y_test, y=y_test
            )
    if comm is not None:
        scores = Gatherv_rows(scores, comm)
        a = Gatherv_rows(a, comm)
        b = Gatherv_rows(b, comm)
        B = Gatherv_rows(B, comm)
        Psi = Gatherv_rows(Psi, comm)
        L = Gatherv_rows(L, comm)
    return scores, a, b, B, Psi, L
