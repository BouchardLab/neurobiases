import numpy as np
import torch

from .utils import inv_softplus


def split_tparams(tparams, N, M, K):
    """Splits torch params up into the respective variables.

    Parameters
    ----------
    tparams : torch.tensor
        Torch tensor containing all the parameters concatenated together.
    N : int
        The number of coupling parameters.
    M : int
        The number of tuning parameters.
    K : int
        The number of latent factors.

    Returns
    -------
    a : torch.tensor, shape (N, 1)
        The coupling parameters.
    b : torch.tensor, shape (M, 1)
        The tuning parameters.
    B : torch.tensor, shape (M, N)
        The non-target tuning parameters.
    Psi_tr : torch.tensor, shape (N + 1,)
        The private variances.
    L : torch.tensor, shape (K, N+1)
        The latent factors.
    """
    a = tparams[:N].reshape(N, 1)
    b = tparams[N:(N + M)].reshape(M, 1)
    B = tparams[(N + M):(N + M + N * M)].reshape(M, N)
    Psi_tr = tparams[(N + M + N * M):(N + M + N * M + N + 1)].reshape(N + 1, 1)
    L = tparams[(N + M + N * M + N + 1):].reshape(K, N + 1)
    return a, b, B, Psi_tr, L


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

    # Dimensionality checks
    if a.ndim == 2:
        a = a.ravel()
    if b.ndim == 2:
        b = b.ravel()
    # Check masks
    if a_mask is None:
        a_mask = np.ones(N)
    elif a_mask.ndim == 2:
        a_mask = a_mask.ravel()

    if b_mask is None:
        b_mask = np.ones(M)
    elif b_mask.ndim == 2:
        b_mask = np.ones(M)

    # Apply masks
    a = a * a_mask
    b = b * b_mask
    # Split up into target and non-target components
    l_t, L_nt = np.split(L, [1], axis=1)
    l_t = l_t.ravel()
    Psi_t, Psi_nt = np.split(Psi, [1])
    Psi_t = Psi_t.item()
    # Mean and covariance matrices of the gaussian expression
    mu = np.zeros((D, N + 1))
    sigma = np.zeros((N + 1, N + 1))
    # Calculate mean of marginal
    mu[:, 0] = X @ (b + B @ a)
    mu[:, 1:] = X @ B
    # Combine data matrices
    Y_all = np.concatenate((y, Y), axis=1)
    # Useful terms to store for later
    coupled_L = l_t + L_nt @ a
    cross_coupling = Psi_nt * a + L_nt.T @ coupled_L
    # Fill covariance matrix
    sigma[0, 0] = Psi_t + Psi_nt @ a**2 + coupled_L @ coupled_L
    sigma[1:, 0] = cross_coupling
    sigma[0, 1:] = cross_coupling
    sigma[1:, 1:] = np.diag(Psi_nt) + L_nt.T @ L_nt
    # Calculate log-likelihood
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


def grad_mask(N, M, K, a_mask, b_mask, B_mask, train_B, train_Psi_tr_nt,
              train_Psi_tr, train_L_nt, train_L):
    """Creats a binary mask for optimization.

    Parameters
    ----------
    N : int
        Data dimensionality.
    M : int
        Stimulus dimensionality.
    K : int
        Latent dimensionality.
    a_mask : np.ndarray, shape (N, 1)
        Mask for coupling features.
    b_mask : nd-array, shape (M, 1)
        Mask for tuning features.
    B_mask : nd-array, shape (N, M)
        Mask for non-target neuron tuning features.
    train_B : bool
        If True, non-target tuning parameters will be trained.
    train_L_nt : bool
        If True, non-target latent factors will be trained.
    train_L : bool
        If True, latent factors will be trained. Takes precedence over
        train_L_nt.
    train_Psi_tr_nt : bool
        If True, non-target private variances will be trained.
    train_Psi_tr : bool
        If True, private variances will be trained. Takes precedence over
        train_Psi_tr_nt.

    Returns
    -------
    grad_mask : ndarray
        Binary mask for the gradient.
    """
    size = N + M + N * M + N + 1 + K * (N + 1)
    grad_mask = np.ones(size, dtype=bool)
    grad_mask[:N] = a_mask.ravel()
    grad_mask[N:(N + M)] = b_mask.ravel()
    # if we're training non-target tuning parameters, apply selection mask
    if train_B:
        grad_mask[(N + M):(N + M + N * M)] = B_mask.ravel()
    else:
        grad_mask[(N + M):(N + M + N * M)] = 0
    # mask out gradients for parameters not being trained
    if not train_Psi_tr_nt:
        grad_mask[(N + M + N * M + 1):(N + M + N * M + N + 1)] = 0
    if not train_Psi_tr:
        grad_mask[(N + M + N * M):(N + M + N * M + N + 1)] = 0
    if not train_L_nt:
        mask = np.zeros(K * (N + 1), dtype=bool)
        mask[0::(N + 1)] = 1
        grad_mask[(N + M + N * M + N + 1):] = mask
    if not train_L:
        grad_mask[(N + M + N * M + N + 1):] = 0
    return grad_mask
