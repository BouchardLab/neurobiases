import numpy as np
import torch

from .lbfgs import fmin_lbfgs
from neurobiases import plot
from neurobiases import em_utils as utils
from scipy.optimize import minimize
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression


class EMSolver():
    """Class to perform expectation-maximization or direct maximum-likelihood
    optimization on data obtained from the triangular model.

    Parameters
    ----------
    X : np.ndarray, shape (D, M)
        Design matrix for tuning features.
    Y : np.ndarray, shape (D, N)
        Design matrix for coupling features.
    y : np.ndarray, shape (D, 1)
        Neural response vector.
    K : int
        Number of latent factors.
    a_mask : np.ndarray, shape (N, 1)
        Mask for coupling features.
    b_mask : nd-array, shape (M, 1)
        Mask for tuning features.
    B_mask : nd-array, shape (N, M)
        Mask for non-target neuron tuning features.
    B : np.ndarray, shape (M, N)
        The non-target tuning parameters.
    L_nt : np.ndarray, shape (K, N)
        The non-target latent factors.
    L : np.ndarray, shape (K, N+1)
        All latent factors. This variable takes precedence over L_nt.
    Psi_nt : np.ndarray, shape (N,)
        The non-target private variances.
    Psi : np.ndarray, shape (N + 1,)
        The private variances. This variable takes precedence over Psi_nt.
    max_iter : int
        The maximum number of optimization iterations to perform.
    tol : float
        The tolerance with which the cease optimization.
    rng  : RandomState or int or None
        RandomState object, or int to create RandomState object.
    """
    def __init__(
        self, X, Y, y, K, a_mask=None, b_mask=None, B_mask=None,
        B=None, L_nt=None, L=None, Psi_nt=None, Psi=None, Psi_transform='softplus',
        solver='scipy_lbfgs', max_iter=1000, tol=1e-4, c_tuning=0., c_coupling=0.,
        penalize_B=False, initialization='zeros', rng=None, fa_rng=None
    ):
        # tuning and coupling design matrices
        self.X = X
        self.Y = Y
        # response vector
        self.y = y
        # number of latent factors
        self.K = K

        # dataset dimensions
        self.D, self.M = self.X.shape
        self.N = self.Y.shape[1]

        # optimization parameters
        self.Psi_transform = Psi_transform
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        if solver == 'scipy_lbfgs':
            self.c_coupling = 0
            self.c_tuning = 0
        else:
            self.c_coupling = c_coupling
            self.c_tuning = c_tuning
        self.penalize_B = penalize_B
        self.rng = np.random.default_rng(rng)
        self.fa_rng = fa_rng
        # initialize masks
        self.set_masks(a_mask=a_mask, b_mask=b_mask, B_mask=B_mask)
        # initialize parameter estimates
        self.initialization = initialization
        self._init_params(initialization)
        # initialize non-target tuning parameters
        self.freeze_B(B=B)
        # initialize variability parameters
        self.freeze_var_params(L_nt=L_nt, L=L, Psi_nt=Psi_nt, Psi=Psi)

    def _init_params(self, initialization='zeros'):
        """Initialize parameter estimates. Requires that X, Y, and y are
        already initialized."""
        if initialization == 'zeros':
            # initialize parameter estimates to be all zeros
            self.a = np.zeros((self.N, 1))
            self.b = np.zeros((self.M, 1))
            self.B = np.zeros((self.M, self.N))
            self.Psi_tr = np.zeros(self.N + 1)
            self.L = self.rng.normal(loc=0., scale=0.1, size=(self.K, self.N + 1))

        elif initialization == 'random':
            # Coupling parameters
            coupling_mask = self.a_mask.ravel().astype('bool')
            self.a = self.rng.normal(loc=0., scale=1., size=(self.N, 1))
            self.a[np.invert(coupling_mask), 0] = 0.
            # Tuning parameters
            tuning_mask = self.b_mask.ravel().astype('bool')
            self.b = self.rng.normal(loc=0., scale=1., size=(self.M, 1))
            self.b[np.invert(tuning_mask), 0] = 0.
            # Non-target tuning parameters
            self.B = np.zeros((self.M, self.N))
            for neuron in range(self.N):
                current_mask = self.B_mask[:, neuron].astype('bool')
                self.B[:, neuron][current_mask] = \
                    self.rng.normal(loc=0., scale=1., size=(current_mask.sum()))
            # Noise parameters
            self.Psi_tr = self.rng.normal(loc=0., scale=1., size=(self.N + 1))
            self.L = self.rng.normal(loc=0., scale=0.1, size=(self.K, self.N + 1))

        elif initialization == 'fits':
            # initialize parameter estimates via fits
            # coupling fit
            coupling_mask = self.a_mask.ravel().astype('bool')
            self.a = np.zeros((self.N, 1))
            coupling = LinearRegression(fit_intercept=False)
            coupling.fit(self.Y[:, coupling_mask], self.y.ravel())
            self.a[coupling_mask, 0] = coupling.coef_

            # tuning fit
            tuning_mask = self.b_mask.ravel().astype('bool')
            self.b = np.zeros((self.M, 1))
            tuning = LinearRegression(fit_intercept=False)
            tuning.fit(self.X[:, tuning_mask], self.y.ravel())
            self.b[tuning_mask, 0] = tuning.coef_

            # non-target tuning fit
            self.B = np.zeros((self.M, self.N))
            for neuron in range(self.N):
                current_mask = self.B_mask[:, neuron].astype('bool')
                tuning = LinearRegression(fit_intercept=False)
                tuning.fit(self.X[:, current_mask], self.Y[:, neuron])
                self.B[:, neuron][current_mask] = tuning.coef_

            # private and shared variability estimated from factor analysis
            Y_res = self.Y - np.dot(self.X, self.B)
            Z = np.concatenate((self.X[:, tuning_mask], self.Y[:, coupling_mask]), axis=1)
            tc_fit = LinearRegression(fit_intercept=False)
            tc_fit.fit(Z, self.y.ravel())
            y_res = self.y.ravel() - np.dot(Z, tc_fit.coef_)
            residuals = np.concatenate((y_res[..., np.newaxis], Y_res), axis=1)
            # run factor analysis on residuals
            fa = FactorAnalysis(n_components=self.K, random_state=self.fa_rng)
            fa.fit(residuals)
            self.L = fa.components_
            self.Psi_tr = self.Psi_to_Psi_tr(fa.noise_variance_)

        else:
            raise ValueError('Incorrect initialization specified.')

    def set_masks(self, a_mask=None, b_mask=None, B_mask=None):
        """Initialize masks. A value of None indicates that all features will
        be included in the mask.

        Parameters
        ----------
        a_mask : np.ndarray, shape (N, 1)
            Mask for coupling features.
        b_mask : nd-array, shape (M, 1)
            Mask for tuning features.
        B_mask : nd-array, shape (N, M)
            Mask for non-target neuron tuning features.
        """
        # coupling parameters mask
        if a_mask is None:
            self.a_mask = np.ones((self.N, 1))
        else:
            self.a_mask = a_mask.reshape((self.N, 1))
        # tuning parameters mask
        if b_mask is None:
            self.b_mask = np.ones((self.M, 1))
        else:
            self.b_mask = b_mask.reshape((self.M, 1))
        # non-target tuning parameters mask
        if B_mask is None:
            self.B_mask = np.ones((self.M, self.N))
        else:
            self.B_mask = B_mask.reshape((self.M, self.N))

    def set_params(self, a=None, b=None, B=None, Psi_tr=None, L=None):
        """Sets parameters equal to the provided parameter values.

        Parameters
        ----------
        a : np.ndarray, shape (N, 1)
            The coupling parameters.
        b : np.ndarray, shape (M, 1)
            The tuning parameters.
        B : np.ndarray, shape (M, N)
            The non-target tuning parameters.
        Psi_tr : np.ndarray, shape (N + 1,)
            The private variances.
        L : np.ndarray, shape (K, N+1)
            The latent factors.
        """
        if a is not None:
            self.a = np.copy(a.reshape((self.N, 1)))
        if b is not None:
            self.b = np.copy(b.reshape((self.M, 1)))
        if B is not None:
            self.B = np.copy(B.reshape(((self.M, self.N))))
        if Psi_tr is not None:
            self.Psi_tr = np.copy(Psi_tr.reshape(self.N + 1))
        if L is not None:
            self.L = np.copy(L.reshape((self.K, self.N + 1)))

    def freeze_B(self, B=None):
        """Sets all (or a subset of) the non-target tuning parameters, and
        freezes them so that they cannot be trained.

        Parameters
        ----------
        B : np.ndarray, shape (M, N)
            The non-target tuning parameters.
        """
        if B is not None:
            self.B_init = B
            self.B = B
            self.B_mask = np.ones((self.M, self.N))
            self.train_B = False
        else:
            self.train_B = True

    def freeze_var_params(self, L_nt=None, L=None, Psi_nt=None, Psi=None):
        """Sets all (or a subset of) the variance parameters, and freezes them
        so that they cannot be trained.

        Parameters
        ----------
        L_nt : np.ndarray, shape (K, N)
            The non-target latent factors.
        L : np.ndarray, shape (K, N+1)
            All latent factors. This variable takes precedence over L_nt.
        Psi_nt : np.ndarray, shape (N,)
            The non-target private variances.
        Psi : np.ndarray, shape (N + 1,)
            The private variances. This variable takes precedence over Psi_nt.
        """
        # initialize non-target latent factors
        if L_nt is not None:
            self.L_nt_init = L_nt
            self.L[:, 1:] = L_nt
            self.train_L_nt = False
        else:
            self.train_L_nt = True
        # initialize all latent factors
        if L is not None:
            self.L_init = L
            self.L = L
            self.train_L = False
        else:
            self.train_L = True
        # initialize all non-target private variances
        if Psi_nt is not None:
            self.Psi_tr_nt_init = self.Psi_to_Psi_tr(Psi_nt)
            self.Psi_tr[1:] = np.copy(self.Psi_tr_nt_init)
            self.train_Psi_tr_nt = False
        else:
            self.train_Psi_tr_nt = True
        # initialize all private variances
        if Psi is not None:
            self.Psi_tr_init = self.Psi_to_Psi_tr(Psi)
            self.Psi_tr = np.copy(self.Psi_tr_init)
            self.train_Psi_tr = False
        else:
            self.train_Psi_tr = True

    def reset_params(self):
        """Reset parameter estimates. If parameter estimates were trained,
        then they are reset to their initial values."""
        self.a = np.zeros((self.N, 1))
        self.b = np.zeros((self.M, 1))

        if self.train_B:
            self.B = np.zeros((self.M, self.N))
        else:
            self.B = self.B_init

        if self.train_L:
            self.L = self.L_init
        else:
            self.L = np.random.normal(size=(self.K, self.N + 1)) / 100

        if self.train_L_nt:
            self.L[:, 1:] = self.L_nt_init
        else:
            self.L[:, 1:] = np.random.normal(size=(self.K, self.N))

        if self.train_Psi_tr_nt:
            self.Psi_tr[1:] = np.copy(self.Psi_tr_nt_init)
        else:
            self.Psi_tr[1:] = np.zeros(self.N + 1)

        if self.train_Psi_tr:
            self.Psi_tr = np.copy(self.Psi_tr_init)
        else:
            self.Psi_tr = np.zeros(self.N + 1)

    def get_params(self, return_Psi=False):
        """Concatenates all parameters into a single vector.

        Returns
        -------
        params : np.ndarray, shape (N + M + N * M + N + 1 + K * (N + 1),)
            A vector containing all parameters concatenated together.
        """
        if return_Psi:
            Psi = self.Psi_tr_to_Psi(self.Psi_tr)
        else:
            Psi = self.Psi_tr
        params = np.concatenate((self.a.ravel(),
                                 self.b.ravel(),
                                 self.B.ravel(),
                                 Psi.ravel(),
                                 self.L.ravel()))
        return params

    def get_Psi(self):
        """Returns the private variances."""
        return self.Psi_tr_to_Psi(self.Psi_tr)

    def get_Psi_t(self):
        """Returns the target private variance."""
        Psi = self.get_Psi()
        return Psi[0]

    def get_Psi_nt(self):
        """Returns the non-target private variances."""
        Psi = self.get_Psi()
        return Psi[1:]

    def get_l_t(self):
        """Returns the target latent factor."""
        return self.L[:, 0]

    def get_L_nt(self):
        """Returns the non-target latent factors."""
        return self.L[:, 1:]

    def get_marginal_cov_blocks(self):
        Psi_t = self.get_Psi_t()
        Psi_nt = np.diag(self.get_Psi_nt())
        l_t = self.get_l_t()
        L_nt = self.get_L_nt()
        l_aug = l_t + L_nt @ self.a

        c00 = Psi_t + self.a.T @ Psi_nt @ self.a + l_aug @ l_aug
        c01 = self.a.T @ Psi_nt + l_aug.T @ L_nt
        c10 = c01.T
        c11 = Psi_nt + L_nt.T @ L_nt
        return c00, c01, c10, c11

    def get_marginal_cov(self):
        c00, c01, c10, c11 = self.get_marginal_cov_blocks()
        sigma = np.bmat(((c00, c01), (c10, c11)))
        return sigma

    def get_marginal_precision(self):
        c0, _, c, C = self.get_marginal_cov_blocks()
        Cc = np.linalg.solve(C, c)
        p00 = 1. / (c0 - c.T @ Cc)
        p10 = -p00 * Cc
        p01 = p10.T
        p11 = np.linalg.inv(C) + p00 * Cc @ Cc.T
        precision = np.bmat(((p00, p01), (p10, p11)))
        return precision

    def get_residual_matrix(self):
        """Returns the D x (N + 1) residual matrix for the neural activities."""
        R = np.concatenate(
            (self.y - self.X @ (self.b + self.B @ self.a),
             self.Y - self.X @ self.B),
            axis=1)
        return R

    def split_params(self, params):
        """Splits a parameter vector into the corresponding parameters.

        Parameters
        ----------
        params : np.ndarray, shape (N + M + N * M + N + 1 + K * (N + 1),)
            A vector containing all parameters concatenated together.

        Returns
        -------
        a : np.ndarray, shape (N, 1)
            The coupling parameters.
        b : np.ndarray, shape (M, 1)
            The tuning parameters.
        B : np.ndarray, shape (M, N)
            The non-target tuning parameters.
        Psi_tr : np.ndarray, shape (N + 1,)
            The private variances.
        L : np.ndarray, shape (K, N+1)
            The latent factors.
        """
        a = params[:self.N].reshape((self.N, 1))
        b = params[self.N:(self.N + self.M)].reshape((self.M, 1))
        B = params[(self.N + self.M):
                   (self.N + self.M + self.M * self.N)].reshape((self.M, self.N))
        Psi_tr = params[(self.N + self.M + self.M * self.N):
                        (self.N + self.M + self.M * self.N + self.N + 1)].reshape(
            self.N + 1)
        L = params[(self.N + self.M + self.N * self.M + self.N + 1):].reshape(
            (self.K, self.N + 1))
        return a, b, B, Psi_tr, L

    def copy(self):
        """Returns a copy of the current EMSolver object.

        Returns
        -------
        copy : EMSolver object
            A copy of the current object, with matching parameters and masks.
        """
        copy = EMSolver(X=self.X, Y=self.Y, y=self.y, K=self.K,
                        a_mask=self.a_mask, b_mask=self.b_mask, B_mask=self.B_mask)
        copy.set_params(a=self.a, b=self.b, B=self.B,
                        Psi_tr=self.Psi_tr, L=self.L)
        return copy

    def Psi_tr_to_Psi(self, Psi_tr=None):
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
        if Psi_tr is None:
            Psi_tr = self.Psi_tr
        return utils.Psi_tr_to_Psi(Psi_tr, self.Psi_transform)

    def Psi_to_Psi_tr(self, Psi):
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
        return utils.Psi_to_Psi_tr(Psi, self.Psi_transform)

    def bic(self, X=None, Y=None, y=None):
        """Calculates the Bayesian information criterion on the neural data.

        Parameters
        ----------
        X : np.ndarray, shape (D, M)
            Design matrix for tuning features.
        Y : np.ndarray, shape (D, N)
            Design matrix for coupling features.
        y : np.ndarray, shape (D, 1)
            Neural response vector.

        Returns
        -------
        bic : float
            The Bayesian information criterion.
        """
        # if no data is provided, use the data in the object
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if y is None:
            y = self.y

        D = X.shape[0]
        Psi = self.Psi_tr_to_Psi(self.Psi_tr)
        # calculate marginal log-likelihood
        mll = self.marginal_log_likelihood(X=X, Y=Y, y=y)
        # number of parameters
        k = \
            np.count_nonzero(self.a) + \
            np.count_nonzero(self.b) + \
            np.count_nonzero(self.B) + \
            np.count_nonzero(self.L) + \
            Psi.size
        bic = -2 * mll + k * np.log(D)
        return bic

    def marginal_log_likelihood(self, X=None, Y=None, y=None):
        """Calculates the marginal likelihood of the neural activities given
        the stimulus and current parameter estimates. Can optionally accept
        new data for comparison across datasets. This function acts as an
        accessor to an external marginal likelihood function.

        Parameters
        ----------
        X : np.ndarray, shape (D, M)
            Design matrix for tuning features.
        Y : np.ndarray, shape (D, N)
            Design matrix for coupling features.
        y : np.ndarray, shape (D, 1)
            Neural response vector.

        Returns
        -------
        mll : float
            The marginal log-likelihood.
        """
        # if no data is provided, use the data in the object
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if y is None:
            y = self.y

        Psi = self.Psi_tr_to_Psi(self.Psi_tr)
        mll = utils.marginal_log_likelihood_linear_tm(
            X=X, Y=Y, y=y, a=self.a, b=self.b, B=self.B, L=self.L,
            Psi=Psi, a_mask=self.a_mask, b_mask=self.b_mask,
            B_mask=self.B_mask
        )
        return mll

    def marginal_likelihood_hessian(self, mask=False, wrt_Psi=True):
        """Calculates the hessian of the marginal likelihood."""
        params = self.get_params()
        # Convert params to Psi if needed
        if wrt_Psi:
            a, b, B, Psi_tr, L = utils.split_tparams(params, self.N, self.M, self.K)
            Psi = self.Psi_tr_to_Psi(Psi_tr)
            params = np.concatenate((a.ravel(),
                                     b.ravel(),
                                     B.ravel(),
                                     Psi.ravel(),
                                     L.ravel()))

        # Convert params to torch tensor
        tparams = torch.tensor(params, requires_grad=True)

        def mll(tparams):
            return self.f_mll(tparams, self.X, self.Y, self.y, self.K,
                              self.Psi_transform, wrt_Psi)

        hessian = torch.autograd.functional.hessian(mll, inputs=tparams)

        if mask:
            a_idx = np.argwhere(self.a_mask.ravel() == 0).ravel()
            b_idx = self.N + np.argwhere(self.b_mask.ravel() == 0).ravel()
            n_entries = hessian.shape[0]
            L_idx = np.arange(self.N + self.M + self.N * self.M + self.N + 1,
                              n_entries)
            L_idx = np.delete(L_idx, np.arange(0, L_idx.size, self.N + 1))
            idx = np.concatenate((a_idx, b_idx, L_idx))
            hessian = np.delete(np.delete(hessian, idx, axis=0), idx, axis=1)

        return hessian

    def e_step(self):
        """Performs an E-step in the EM algorithm for the triangular model.

        Returns
        -------
        mu : np.ndarray, shape (D, K)
            The expected value of the latent state across samples.
        zz : np.ndarray, shape (D, K, K)
            The expected second-order statistics of the latent state across
            samples.

        sigma : np.ndarray, shape (K, K)
            The covariance of the latent states.
        """
        # apply masks
        a = self.a * self.a_mask
        b = self.b * self.b_mask
        B = self.B * self.B_mask

        # private variances
        Psi = self.Psi_tr_to_Psi(self.Psi_tr)
        Psi_t, Psi_negt = np.split(Psi, [1])

        # interaction terms
        LPsi = self.L / Psi
        lpsi_t, Lpsi_nt = np.split(LPsi, [1], axis=1)

        # calculate covariance
        sigma_inv = np.eye(self.K) + np.dot(LPsi, self.L.T)
        sigma = np.linalg.inv(sigma_inv)

        # calculate mu
        y_residual = self.y - np.dot(self.X, b) - np.dot(self.Y, a)
        Y_residual = self.Y - np.dot(self.X, B)
        y_residual_scaled = np.dot(lpsi_t, y_residual.T)
        Y_residual_scaled = np.dot(Lpsi_nt, Y_residual.T)
        mu = np.dot(sigma, y_residual_scaled + Y_residual_scaled).T

        # calculate zz
        zz = sigma[np.newaxis] + mu[..., np.newaxis] * mu[:, np.newaxis]

        return mu, zz, sigma

    def m_step(self, mu, zz, sigma, verbose=False, fista_max_iter=250,
               fista_lr=1e-6, store_parameters=False):
        """Performs an M-step in the EM algorithm for the triangular model.

        Parameters
        ----------
        mu : np.ndarray, shape (D, K)
            The expected value of the latent state across samples.
        zz : np.ndarray, shape (D, K, K)
            The expected second-order statistics of the latent state across
            samples.
        sigma : np.ndarray, shape (K, K)
            The covariance of the latent states.
        verbose : bool
            If True, print callback statements.

        Returns
        -------
        params : np.ndarray, shape (N + M + N * M + N + 1 + K * (N + 1),)
            A vector containing all parameters concatenated together.
        """
        # Get default values of parameters
        params = self.get_params()
        # Storage variable set to None if unused
        if not store_parameters:
            storage = None

        # Use scipy's LBFGS solver (can't apply sparsity)
        if self.solver == 'scipy_lbfgs':
            # Create callback function for verbosity
            if verbose:
                # create callback function
                def callback(params):
                    print(
                        'Expected complete log-likelihood:',
                        self.expected_complete_ll(params, self.X, self.Y, self.y,
                                                  mu, zz, sigma,
                                                  c_coupling=0.,
                                                  c_tuning=0.,
                                                  a_mask=self.a_mask,
                                                  b_mask=self.b_mask,
                                                  B_mask=self.B_mask,
                                                  transform_tuning=False,
                                                  Psi_transform=self.Psi_transform)
                    )
            else:
                callback = None
            optimize = minimize(
                self.f_df_em, x0=params,
                method='L-BFGS-B',
                args=(self.X, self.Y, self.y,
                      self.a_mask, self.b_mask, self.B_mask,
                      self.train_B,
                      self.train_L_nt,
                      self.train_L,
                      self.train_Psi_tr_nt,
                      self.train_Psi_tr,
                      mu, zz, sigma,
                      1.,
                      self.penalize_B,
                      self.Psi_transform),
                callback=callback,
                jac=True)
            # extract optimized parameters
            params = optimize.x

        # Use orthant-wise lbfgs solver (for sparse situations)
        elif self.solver == 'ow_lbfgs':
            # create callable for owlbfgs
            if store_parameters:
                def progress(x, g, fx, xnorm, gnorm, step, k, num_eval, *args):
                    # Expand arrays if we have run out of storage
                    if storage['n_iterations'][-1] == 1:
                        storage['n_iterations'] = np.concatenate(
                            (storage['n_iterations'],
                             np.zeros(10000, dtype=np.int64)))
                        storage['ll'] = np.concatenate(
                            (storage['ll'],
                             np.zeros(10000, dtype=np.int64))
                        )
                        storage['a'] = np.concatenate(
                            (storage['a'],
                             np.zeros((10000, self.N))),
                            axis=0)
                        storage['b'] = np.concatenate(
                            (storage['b'],
                             np.zeros((10000, self.M))),
                            axis=0)
                        storage['Psi'] = np.concatenate(
                            (storage['Psi'],
                             np.zeros((10000, self.N + 1))),
                            axis=0)
                    # Include most recent parameter estimates in arrays
                    storage['n_iterations'][k-1] = 1
                    storage['ll'][k-1] = utils.marginal_log_likelihood_linear_tm(
                        X=self.X,
                        Y=self.Y,
                        y=self.y,
                        a=x[:self.N],
                        b=x[self.N:self.N+self.M],
                        B=x[(self.N + self.M):(self.N + self.M + self.M * self.N)].reshape(
                            (self.M, self.N)
                        ),
                        L=x[(self.N + self.M + self.N * self.M + self.N + 1):].reshape(
                            (self.K, self.N + 1)
                        ),
                        Psi=self.Psi_tr_to_Psi(
                            x[(self.N + self.M + self.M * self.N):
                              (self.N + self.M + self.M * self.N + self.N + 1)]
                        ),
                        a_mask=self.a_mask,
                        b_mask=self.b_mask,
                        B_mask=self.B_mask
                    )
                    storage['a'][k-1] = x[:self.N]
                    storage['b'][k-1] = x[self.N:self.N+self.M]
                    Psi_idx = self.N + self.M + self.N * self.M
                    Psi_tr = x[Psi_idx:Psi_idx + self.N + 1]
                    storage['Psi'][k-1] = self.Psi_tr_to_Psi(Psi_tr)
            elif verbose:
                # create callback function
                def progress(x, g, fx, xnorm, gnorm, step, k, num_eval, *args):
                    # x is in the transformed space
                    print(
                        'Expected complete log-likelihood:',
                        self.expected_complete_ll(x, self.X, self.Y, self.y,
                                                  mu, zz, sigma,
                                                  c_coupling=self.c_coupling,
                                                  c_tuning=self.c_tuning,
                                                  a_mask=self.a_mask,
                                                  b_mask=self.b_mask,
                                                  B_mask=self.B_mask,
                                                  transform_tuning=True,
                                                  penalize_B=self.penalize_B,
                                                  Psi_transform=self.Psi_transform)
                    )
            else:
                progress = None
            # penalize both coupling and tuning
            if self.c_coupling != 0 and self.c_tuning != 0:
                tuning_to_coupling_ratio = float(self.c_tuning) / self.c_coupling
                orthantwise_start = 0
                if self.penalize_B:
                    orthantwise_end = self.N + self.M + self.N * self.M
                else:
                    orthantwise_end = self.N + self.M
                c = self.c_coupling
                # transform tuning parameters
                params[self.N:orthantwise_end] *= tuning_to_coupling_ratio
            # penalize only coupling
            elif self.c_tuning == 0 and self.c_coupling != 0:
                tuning_to_coupling_ratio = 1.
                orthantwise_start = 0
                orthantwise_end = self.N
                c = self.c_coupling
            # penalize only tuning
            elif self.c_coupling == 0 and self.c_tuning != 0:
                tuning_to_coupling_ratio = 1.
                orthantwise_start = self.N
                if self.penalize_B:
                    orthantwise_end = self.N + self.M + self.N * self.M
                else:
                    orthantwise_end = self.N + self.M
                c = self.c_tuning
            # penalize neither
            else:
                orthantwise_start = 0
                orthantwise_end = -1
                c = 0
                tuning_to_coupling_ratio = 1.

            # tuning params are transformed
            if store_parameters:
                storage = {}
                storage['ll'] = np.zeros(10000)
                storage['n_iterations'] = np.zeros(10000, dtype=np.int64)
                storage['a'] = np.zeros((10000, self.N))
                storage['b'] = np.zeros((10000, self.M))
                storage['Psi'] = np.zeros((10000, self.N + 1))
            else:
                storage = None
            params = fmin_lbfgs(
                self.f_df_em_owlbfgs, x0=params,
                args=(self.X, self.Y, self.y,
                      self.a_mask, self.b_mask, self.B_mask,
                      self.train_B,
                      self.train_L_nt,
                      self.train_L,
                      self.train_Psi_tr_nt,
                      self.train_Psi_tr,
                      mu, zz, sigma,
                      tuning_to_coupling_ratio,
                      self.penalize_B,
                      self.Psi_transform,
                      storage),
                progress=progress,
                orthantwise_c=c,
                orthantwise_start=orthantwise_start,
                orthantwise_end=orthantwise_end)

            a, b, B, Psi_tr, L = self.split_params(params)
            b = b / tuning_to_coupling_ratio
            if self.penalize_B:
                B = B / tuning_to_coupling_ratio
            # transform params back to original values
            params = np.concatenate((a.ravel(),
                                     b.ravel(),
                                     B.ravel(),
                                     Psi_tr.ravel(),
                                     L.ravel()))
        # apply fista (for the sparse setting)
        elif self.solver == 'fista':
            zero_start = -1
            zero_end = -1
            one_start = -1
            one_end = -1
            if self.c_coupling > 0.:
                zero_start = 0
                zero_end = self.N
            if self.c_tuning > 0.:
                one_start = self.N
                one_end = self.N + self.M + self.N * self.M
            args = (self.X, self.Y, self.y, self.a_mask, self.b_mask, self.B_mask,
                    self.train_B, self.train_L_nt, self.train_L, self.train_Psi_tr_nt,
                    self.train_Psi_tr, mu, zz, sigma, 1.)
            params = utils.fista(self.f_df_em,
                                 params,
                                 lr=fista_lr,
                                 max_iter=fista_max_iter,
                                 C0=self.c_coupling,
                                 C1=self.c_tuning,
                                 zero_start=zero_start,
                                 zero_end=zero_end,
                                 one_start=one_start,
                                 one_end=one_end,
                                 verbose=verbose,
                                 args=args)
        else:
            raise ValueError(f'Solver {self.solver} not available.')

        return params, storage

    def fit_em(
        self, refit=False, verbose=False, mstep_verbose=False, mll_curve=False,
        fista_max_iter=250, fista_lr=1e-6, store_parameters=False
    ):
        """Fit the triangular model parameters using the EM algorithm.

        Parameters
        ----------
        verbose : bool
            If True, print out EM iteration updates.
        mstep_verbose : bool
            If True, print out M-step iteration updates.
        mll_curve : bool
            If True, return a curve containing the marginal likelihoods at
            each step of optimization.

        Returns
        -------
        mlls : np.ndarray
            An array containing the marginal log-likelihood of the model fit
            at each EM iteration.
        """
        # initialize iteration count and change in likelihood
        iteration = 0
        del_ml = np.inf

        # initialize storage for marginal log-likelihoods
        mlls = np.zeros(self.max_iter + 1)
        base_mll = self.marginal_log_likelihood()
        mlls[0] = base_mll

        if verbose:
            print('Initial marginal likelihood: %f' % base_mll)

        if store_parameters:
            steps = []
            ll_path = []
            a_path = []
            b_path = []
            Psi_path = []

        # EM iteration loop: convergence if tolerance or maximum iterations
        # are reached
        while (del_ml > self.tol) and (iteration < self.max_iter):
            # run E-step
            mu, zz, sigma = self.e_step()
            # run M-step
            params, storage = self.m_step(
                mu, zz, sigma,
                verbose=mstep_verbose,
                fista_max_iter=fista_max_iter,
                fista_lr=fista_lr,
                store_parameters=store_parameters)
            if store_parameters:
                n_steps = np.sum(storage['n_iterations'])
                steps.append(n_steps)
                ll_path.append(storage['ll'][:n_steps])
                a_path.append(storage['a'][:n_steps])
                b_path.append(storage['b'][:n_steps])
                Psi_path.append(storage['Psi'][:n_steps])

            self.a, self.b, self.B, self.Psi_tr, self.L = self.split_params(params)
            iteration += 1
            # update marginal log-likelihood
            current_mll = self.marginal_log_likelihood()
            # calculate fraction change in marginal log-likelihood
            del_ml = np.abs(current_mll - base_mll) / np.abs(base_mll)
            base_mll = current_mll
            mlls[iteration] = current_mll

            if verbose:
                print(
                    f'Iteration {iteration}: '
                    f'del={del_ml:0.5E}, '
                    f'mll={mlls[iteration]:0.7E}')
        self.n_iterations = iteration + 1

        if store_parameters:
            self.steps = np.array(steps)
            self.ll_path = np.concatenate(ll_path, axis=0)
            self.a_path = np.concatenate(a_path, axis=0)
            self.b_path = np.concatenate(b_path, axis=0)
            self.Psi_path = np.concatenate(Psi_path, axis=0)

        if refit:
            a_mask = self.a.ravel() != 0
            b_mask = self.b.ravel() != 0
            B_mask = self.B != 0
            self.set_masks(a_mask=a_mask, b_mask=b_mask, B_mask=B_mask)
            if verbose:
                print('Refitting EM estimates with new masks.')
            return self.fit_em(
                refit=False, verbose=verbose, mstep_verbose=mstep_verbose,
                mll_curve=mll_curve, fista_max_iter=fista_max_iter,
                fista_lr=fista_lr)
        else:
            if mll_curve:
                return mlls[:iteration]
            else:
                return self

    def fit_ml(self, verbose=False):
        """Fit the parameters using maximum likelihood."""
        params = self.get_params()
        if verbose:
            def callback(params):
                print('Marginal likelihood: ',
                      self.mll(params, self.X, self.Y, self.y,
                               self.K, self.a_mask, self.b_mask,
                               self.B_mask, self.Psi_transform))
        else:
            callback = None

        optimize = minimize(
            self.f_df_ml, x0=params,
            method='L-BFGS-B',
            args=(self.X, self.Y, self.y, self.K,
                  self.a_mask, self.b_mask, self.B_mask,
                  self.train_B, self.train_L, self.train_L_nt,
                  self.train_Psi_tr_nt, self.train_Psi_tr, self.Psi_transform),
            callback=callback,
            jac=True)
        params = optimize.x
        self.a, self.b, self.B, self.Psi_tr, self.L = self.split_params(params)
        return self

    def identifiability_transform(self, delta):
        """Apply an identifiability transform to the current parameters.

        Parameters
        ----------
        delta : np.ndarray, shape (K,)
            Identifiability transform parameter.
        """
        # grab latent factors
        l_t = self.L[:, 0][..., np.newaxis]
        L_nt = self.L[:, 1:]
        # grab private variances
        Psi = self.Psi_tr_to_Psi(self.Psi_tr)
        Psi_t = Psi[0]
        Psi_nt = np.diag(Psi[1:])

        if delta.ndim == 1:
            delta = delta[..., np.newaxis]
        # perturbation for coupling terms
        Delta = -np.linalg.solve(Psi_nt + np.dot(L_nt.T, L_nt),
                                 np.dot(L_nt.T, delta).ravel())[..., np.newaxis]

        # create augmented variables
        delta_aug = delta + np.dot(L_nt, Delta)
        L_aug = l_t + np.dot(L_nt, self.a)

        # correction for target private variance
        correction = \
            - 2 * np.dot(Delta.T, np.dot(Psi_nt, self.a)).ravel() \
            - np.dot(Delta.T, np.dot(Psi_nt, Delta)).ravel() \
            - np.dot(delta_aug.T, delta_aug) \
            - 2 * np.dot(L_aug.T, delta_aug)

        self.Psi_tr[0] = self.Psi_to_Psi_tr(Psi_t + correction)
        self.L[:, 0] = self.L[:, 0] + delta.ravel()
        self.a = self.a + Delta
        self.b = self.b - np.dot(self.B, Delta)

    def apply_identifiability_constraint(
        self, constraint, a_mask=None, b_mask=None
    ):
        """Apply an identifiability transform to satisfy a specified constraint.

        Parameters
        ----------
        constraint : string
            The identifiability constraint.
        a_mask : np.ndarray, shape (N, 1)
            Mask for coupling features.
        b_mask : nd-array, shape (M, 1)
            Mask for tuning features.

        Returns
        -------
        delta : np.ndarray, shape (K, 1)
            The identifiability parameter that achieves the constraint.
        """
        # latent factors
        l_t = self.L[:, 0][..., np.newaxis]
        L_nt = self.L[:, 1:]
        # grab private variances
        Psi = self.Psi_tr_to_Psi(self.Psi_tr)
        Psi_t = Psi[0]
        Psi_nt = np.diag(Psi[1:])

        # apply new masks, if provided
        if a_mask is None:
            a_mask = np.ones(self.a.shape, dtype=self.a.dtype)
        else:
            a_mask = a_mask.astype(self.a.dtype)

        if b_mask is None:
            b_mask = np.ones(self.b.shape, dtype=self.b.dtype)
        else:
            b_mask = b_mask.astype(self.b.dtype)

        optimize = minimize(
            self.f_df_constraint, x0=np.zeros(self.K),
            method='L-BFGS-B',
            args=(constraint, l_t, L_nt, Psi_t, Psi_nt,
                  self.a, self.b, self.B, a_mask, b_mask),
            jac=True)

        delta = optimize.x[..., np.newaxis]
        self.identifiability_transform(delta)
        return delta

    def transform_oracle(self, a_true, b_true):
        """Transform the parameters to be as close to the provided true
        parameters as possible.

        Parameters
        ----------
        a_true : np.ndarray, shape (N, 1)
            The true coupling parameters.

        b_true : np.ndarray, shape (M, 1)
            The true tuning parameters.

        Returns
        -------
        delta : np.ndarray, shape (K, 1)
            The identifiability parameter that achieves the constraint.
        """
        # latent factors
        L_nt = self.L[:, 1:]
        # grab private variances
        Psi = self.Psi_tr_to_Psi(self.Psi_tr)
        Psi_nt = np.diag(Psi[1:])

        optimize = minimize(
            self.f_df_oracle, x0=np.ones(self.K),
            method='L-BFGS-B',
            args=(L_nt, Psi_nt, self.a, self.b, self.B, a_true, b_true),
            jac=True)

        delta = optimize.x[..., np.newaxis]
        self.identifiability_transform(delta)
        return delta

    def create_cov(self):
        """Calculate the covariance matrix of the noise terms."""
        Psi = self.Psi_tr_to_Psi(self.Psi_tr)
        cov = np.dot(self.L.T, self.L) + np.diag(Psi)
        return cov

    def create_corr(self):
        """Calculate the correlation matrix of the noise terms."""
        cov = self.create_cov()
        inv_var = (1. / np.diag(cov))**(0.5)
        corr = cov * np.outer(inv_var, inv_var)
        return corr

    def calculate_private_shared_ratio(self):
        """Calculate the ratio of private variance to shared variance, for all
        neurons."""
        return self.Psi_tr_to_Psi(self.Psi_tr) / np.diag(np.dot(self.L.T, self.L))

    def plot_tc_fits(self, tm, fax=None, color='black', edgecolor='white'):
        """Scatters estimated tuning and coupling fits against ground truth
        fits.

        Parameters
        ----------
        tm : TriangularModel object
            The TriangularModel with the ground truth parameters.
        fax : mpl.figure and mpl.axes
            The matplotlib axes objects. If None, new objects are created.
        color : string
            The color of the points.
        edgecolor : string
            The edgecolor of the points.

        Returns
        -------
        fig, axes : mpl.figure and mpl.axes
            The matplotlib axes objects, with fits plotted on the axes.
        """
        # get true/estimated fits
        a_hat = self.a.ravel()
        a_true = tm.a.ravel()
        b_hat = self.b.ravel()
        b_true = tm.b.ravel()
        # plot fit comparison
        fig, axes = plot.plot_tc_fits(
            a_hat=a_hat, a_true=a_true, b_hat=b_hat, b_true=b_true,
            fax=fax, color=color, edgecolor=edgecolor
        )
        return fig, axes

    def compare_fits(self, tm, fax=None, color='black', edgecolor='white'):
        """Scatters a comparison between estimated/true parameters, across
        all parameters in the triangular model.

        Parameters
        ----------
        tm : TriangularModel object
            The TriangularModel with the ground truth parameters.
        fax : mpl.figure and mpl.axes
            The matplotlib axes objects. If None, new objects are created.
        color : string
            The color of the points.
        edgecolor : string
            The edgecolor of the points.

        Returns
        -------
        fig, axes : mpl.figure and mpl.axes
            The matplotlib axes objects, with fits plotted on the axes.
        """
        a_hat = self.a.ravel()
        a_true = tm.a.ravel()
        b_hat = self.b.ravel()
        b_true = tm.b.ravel()
        B_hat = self.B
        B_true = tm.B
        L_hat = self.L
        L_true = tm.L
        Psi_hat = self.Psi_tr_to_Psi(self.Psi_tr)
        Psi_true = tm.Psi

        fig, axes = plot.plot_tm_fits(
            a_hat=a_hat, a_true=a_true, b_hat=b_hat, b_true=b_true, B_hat=B_hat,
            B_true=B_true, L_hat=L_hat, L_true=L_true, Psi_hat=Psi_hat,
            Psi_true=Psi_true, fax=fax, color=color, edgecolor=edgecolor
        )
        return fig, axes

    @staticmethod
    def f_mll(
        tparams, X, Y, y, K, Psi_transform='softplus', wrt_Psi=False
    ):
        """Helper function for parameter fitting with maximum likelihood.
        Calculates the log-likelihood of the neural activity and gradients
        with respect to all parameters.

        Parameters
        ----------
        X : np.ndarray, shape (D, M)
            Design matrix for tuning features.
        Y : np.ndarray, shape (D, N)
            Design matrix for coupling features.
        y : np.ndarray, shape (D, 1)
            Neural response vector.
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
        loss : float
            The marginal log-likelihood.
        grad : np.ndarray
            The gradient of the loss with respect to all parameters. Any
            variables whose training flag was set to False will have corresponding
            gradients set equal to zero.
        """
        # Extract dimensions
        D, M = X.shape
        N = Y.shape[1]
        a, b, B, Psi_tr, L = utils.split_tparams(tparams, N, M, K)

        # Split up terms into target/non-target components
        if wrt_Psi:
            Psi = Psi_tr
        else:
            Psi = utils.Psi_tr_to_Psi(Psi_tr, Psi_transform)
        Psi_t = Psi[0]
        Psi_nt = Psi[1:].reshape(N, 1)  # N x 1
        l_t = L[:, 0].reshape(K, 1)  # K x 1
        L_nt = L[:, 1:]  # K x N

        # Turn data into torch tensors
        X = torch.tensor(X)  # D x M
        Y = torch.tensor(Y)  # D x N
        y = torch.tensor(y)  # D x 1
        y_cat = torch.cat((y, Y), dim=1)

        # Useful terms for the marginal log-likelihood
        mu_t = torch.mm(X, b + torch.mm(B, a))  # D x 1
        mu_nt = torch.mm(X, B)  # D x N
        mu = torch.cat((mu_t, mu_nt), dim=1)
        coupled_L = l_t + torch.mm(L_nt, a)  # K x 1

        # Calculate marginal log-likelihood
        c0 = Psi_t + torch.mm(Psi_nt.t(), a**2) + torch.mm(coupled_L.t(), coupled_L)
        c = Psi_nt * a + torch.mm(L_nt.t(), coupled_L)  # N x 1
        C = torch.diag(torch.flatten(Psi_nt)) + torch.mm(L_nt.t(), L_nt)
        sigma = torch.cat((
            torch.cat((c0, c.t()), dim=1),
            torch.cat((c, C), dim=1)),
            dim=0)

        # Calculate loss and perform autograd
        residual = y_cat - mu
        ll = 0.5 * D * torch.logdet(sigma) \
            + 0.5 * torch.sum(residual.t() * torch.solve(residual.t(), sigma)[0])
        return ll

    @staticmethod
    def f_df_ml(
        params, X, Y, y, K, a_mask, b_mask, B_mask, train_B=True,
        train_L_nt=True, train_L=True, train_Psi_tr_nt=True,
        train_Psi_tr=True, Psi_transform='softplus', wrt_Psi=False
    ):
        """Helper function for parameter fitting with maximum likelihood.
        Calculates the log-likelihood of the neural activity and gradients
        with respect to all parameters.

        Parameters
        ----------
        X : np.ndarray, shape (D, M)
            Design matrix for tuning features.
        Y : np.ndarray, shape (D, N)
            Design matrix for coupling features.
        y : np.ndarray, shape (D, 1)
            Neural response vector.
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
        loss : float
            The marginal log-likelihood.
        grad : np.ndarray
            The gradient of the loss with respect to all parameters. Any
            variables whose training flag was set to False will have corresponding
            gradients set equal to zero.
        """
        # Extract dimensions
        D, M = X.shape
        N = Y.shape[1]

        # Option to take derivative w.r.t Psi rather than Psi_tr
        if wrt_Psi:
            a, b, B, Psi_tr, L = utils.split_tparams(params, N, M, K)
            Psi = utils.Psi_tr_to_Psi(Psi_tr, Psi_transform)
            params = np.concatenate((a.ravel(), b.ravel(), B.ravel(), Psi.ravel(), L.ravel()))

        # Turn parameters into torch tensors
        tparams = torch.tensor(params, requires_grad=True)
        a, b, B, Psi_tr, L = utils.split_tparams(tparams, N, M, K)
        a = a * torch.tensor(a_mask, dtype=a.dtype)
        b = b * torch.tensor(b_mask, dtype=b.dtype)
        B = B * torch.tensor(B_mask, dtype=B.dtype)

        # Split up terms into target/non-target components
        if wrt_Psi:
            Psi = Psi_tr
        else:
            Psi = utils.Psi_tr_to_Psi(Psi_tr, Psi_transform)
        Psi_t = Psi[0]
        Psi_nt = Psi[1:].reshape(N, 1)  # N x 1
        l_t = L[:, 0].reshape(K, 1)  # K x 1
        L_nt = L[:, 1:]  # K x N

        # Turn data into torch tensors
        X = torch.tensor(X)  # D x M
        Y = torch.tensor(Y)  # D x N
        y = torch.tensor(y)  # D x 1
        y_cat = torch.cat((y, Y), dim=1)

        # Useful terms for the marginal log-likelihood
        mu_t = torch.mm(X, b + torch.mm(B, a))  # D x 1
        mu_nt = torch.mm(X, B)  # D x N
        mu = torch.cat((mu_t, mu_nt), dim=1)
        coupled_L = l_t + torch.mm(L_nt, a)  # K x 1

        # Calculate marginal log-likelihood
        c0 = Psi_t + torch.mm(Psi_nt.t(), a**2) + torch.mm(coupled_L.t(), coupled_L)
        c = Psi_nt * a + torch.mm(L_nt.t(), coupled_L)  # N x 1
        C = torch.diag(torch.flatten(Psi_nt)) + torch.mm(L_nt.t(), L_nt)
        sigma = torch.cat((
            torch.cat((c0, c.t()), dim=1),
            torch.cat((c, C), dim=1)),
            dim=0)

        # Calculate loss and perform autograd
        residual = y_cat - mu
        ll = 0.5 * D * torch.logdet(sigma) \
            + 0.5 * torch.sum(residual.t() * torch.solve(residual.t(), sigma)[0])
        ll.backward()
        loss = ll.detach().numpy().item()
        grad = tparams.grad.detach().numpy()

        # Apply masks to the gradient
        grad[:N] *= a_mask.ravel()
        grad[N:(N + M)] *= b_mask.ravel()

        # If we're training non-target tuning parameters, apply selection mask
        if train_B:
            grad[(N + M):(N + M + N * M)] *= B_mask.ravel()
        else:
            grad[(N + M):(N + M + N * M)] = 0

        # Mask out gradients for parameters not being trained
        if not train_Psi_tr_nt:
            grad[(N + M + N * M + 1):(N + M + N * M + N + 1)] = 0
        if not train_Psi_tr:
            grad[(N + M + N * M):(N + M + N * M + N + 1)] = 0
        if not train_L:
            grad[(N + M + N * M + N + 1):] = 0
        if not train_L_nt:
            mask = np.zeros(grad[(N + M + N * M + N + 1):].size)
            mask[0::(N + 1)] = np.ones(K)
            grad[(N + M + N * M + N + 1):] *= mask

        return loss, grad

    @staticmethod
    def f_df_em(params, X, Y, y, a_mask, b_mask, B_mask, train_B, train_L_nt,
                train_L, train_Psi_tr_nt, train_Psi_tr, mu, zz, sigma,
                tuning_to_coupling_ratio, penalize_B=False, Psi_transform='softplus',
                storage=None):
        """Helper function for the M-step in the EM procedure. Calculates the
        expected complete log-likelihood and gradients with respect to all
        parameters.

        Parameters
        ----------
        X : np.ndarray, shape (D, M)
            Design matrix for tuning features.
        Y : np.ndarray, shape (D, N)
            Design matrix for coupling features.
        y : np.ndarray, shape (D, 1)
            Neural response vector.
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
        mu : np.ndarray, shape (D, K)
            The expected value of the latent state across samples.
        zz : np.ndarray, shape (D, K, K)
            The expected second-order statistics of the latent state across
            samples.
        sigma : np.ndarray, shape (K, K)
            The covariance of the latent states.

        Returns
        -------
        loss : float
            The value of the loss function, the expected complete log-likelihood.
        grad : np.ndarray
            The gradient of the loss with respect to all parameters. Any
            variables whose training flag was set to False will have corresponding
            gradients set equal to zero.
        """
        # extract dimensions
        D, M = X.shape
        N = Y.shape[1]
        K = mu.shape[1]
        # turn parameters into torch tensors
        tparams = torch.tensor(params, requires_grad=True)
        a, b, B, Psi_tr, L = utils.split_tparams(tparams, N, M, K)
        # apply rescaling
        b = b / tuning_to_coupling_ratio
        if penalize_B:
            B = B / tuning_to_coupling_ratio
        # split up terms into target/non-target components
        Psi = utils.Psi_tr_to_Psi(Psi_tr, Psi_transform)
        Psi_nt = Psi[1:]
        l_t = L[:, 0].reshape(K, 1)
        L_nt = L[:, 1:]

        # turn data and E-step variables into torch tensors
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        y = torch.tensor(y)
        mu = torch.tensor(mu)
        zz = torch.tensor(zz)
        sigma = torch.tensor(sigma)

        # useful terms for the expected complete log-likelihood
        y_residual = y - torch.mm(X, b) - torch.mm(Y, a)
        Y_residual = Y - torch.mm(X, B)
        muL = torch.mm(mu, L_nt)

        # calculate expected complete log-likelihood, term by term
        # see paper for derivation
        term1 = torch.sum(torch.log(Psi))
        term2 = torch.mean(y_residual**2 / Psi[0])
        term3 = torch.mean((-2. / Psi[0]) * y_residual * torch.mm(mu, l_t))
        term4 = torch.mean(
            torch.matmul(torch.transpose(torch.matmul(zz, l_t), 1, 2), l_t)) / Psi[0]
        term5 = torch.sum(Y_residual**2 / Psi_nt.t()) / D
        term6 = -2 * torch.sum(Y_residual * muL / Psi_nt.t()) / D
        term7a = torch.trace(torch.chain_matmul(L_nt, L_nt.t() / Psi_nt, sigma))
        term7b = torch.sum(muL**2 / Psi_nt.t()) / D
        # calculate loss and perform autograd
        loss = term1 + term2 + term3 + term4 + term5 + term6 + term7a + term7b
        loss.backward()
        # extract gradient
        grad = tparams.grad.detach().numpy()

        # apply masks to the gradient
        grad[:N] *= a_mask.ravel()
        grad[N:(N + M)] *= b_mask.ravel()

        # if we're training non-target tuning parameters, apply selection mask
        if train_B:
            grad[(N + M):(N + M + N * M)] *= B_mask.ravel()
        else:
            grad[(N + M):(N + M + N * M)] = 0

        # mask out gradients for parameters not being trained
        if not train_Psi_tr_nt:
            grad[(N + M + N * M + 1):(N + M + N * M + N + 1)] = 0
        if not train_Psi_tr:
            grad[(N + M + N * M):(N + M + N * M + N + 1)] = 0
        if not train_L_nt:
            mask = np.zeros(grad[(N + M + N * M + N + 1):].size)
            mask[0::(N + 1)] = np.ones(K)
            grad[(N + M + N * M + N + 1):] *= mask
        if not train_L:
            grad[(N + M + N * M + N + 1):] = 0

        return loss.detach().numpy(), grad

    @staticmethod
    def f_df_em_owlbfgs(params, grad, *args):
        """Wrapper for OW LBFGS"""
        loss, g = EMSolver.f_df_em(params, *args)
        grad[:] = g
        return loss

    @staticmethod
    def f_df_constraint(
        delta, constraint, l_t, L_nt, Psi_t, Psi_nt, a, b, B, a_mask, b_mask
    ):
        """Helper function for applying identifiability transform. Calculates
        the optimal identifiability parameter to satisfy a provided constraint.

        Parameters
        ----------
        L_nt : np.ndarray, shape (K, N)
            The non-target latent factors.

        L : np.ndarray, shape (K, N+1)
            All latent factors. This variable takes precedence over L_nt.

        """
        delta = torch.tensor(delta[..., np.newaxis], requires_grad=True)
        l_t = torch.tensor(l_t)
        L_nt = torch.tensor(L_nt)
        Psi_t = torch.tensor(Psi_t)
        Psi_nt = torch.tensor(Psi_nt)
        a = torch.tensor(a)
        b = torch.tensor(b)
        B = torch.tensor(B)

        a_mask = torch.tensor(a_mask, dtype=a.dtype)
        b_mask = torch.tensor(b_mask, dtype=b.dtype)
        # useful intermediate variables
        Delta = torch.solve(-1 * torch.mm(L_nt.t(), delta),
                            Psi_nt + torch.mm(L_nt.t(), L_nt))[0]
        L_aug = l_t + torch.mm(L_nt, a)
        delta_aug = delta + torch.mm(L_nt, Delta)

        l_t_mod = l_t + delta
        a_mod = a + Delta
        Psi_t_mod = Psi_t \
            - 2 * torch.chain_matmul(Delta.t(), Psi_nt, a) \
            - torch.chain_matmul(Delta.t(), Psi_nt, Delta) \
            - torch.mm(delta_aug.t(), delta_aug) \
            - 2 * torch.mm(L_aug.t(), delta_aug)
        b_mod = b - torch.mm(B, Delta)

        if constraint == 'coupling_norm':
            loss = torch.sum(a_mod**2)

        elif constraint == 'tuning_norm':
            loss = torch.sum(b_mod**2)

        elif constraint == 'coupling_and_tuning_norm':
            loss = torch.sum(a_mod**2) + torch.sum(b_mod**2)

        elif constraint == 'coupling_norm_w_mask':
            loss = torch.sum(a_mod**2)
            loss += torch.sum(((1 - a_mask) * a_mod)**2)
            loss += torch.sum(((1 - b_mask) * b_mod)**2)

        elif constraint == 'coupling_sum':
            loss = torch.sum(a_mod)**2

        elif constraint == 'target_private_variance':
            loss = Psi_t_mod * -1

        elif constraint == 'variance_ratio':
            l_t_mod_norm = torch.sum(l_t_mod**2)
            desired_ratio = torch.mean(
                torch.diag(Psi_nt) / torch.diag(torch.mm(L_nt.t(), L_nt))
            )
            loss = (Psi_t_mod - l_t_mod_norm * desired_ratio)**2

        elif constraint == 'covariance_ratio':
            L_mod = torch.cat((l_t_mod, L_nt), dim=1)
            LL_mod = torch.mm(L_mod.t(), L_mod)
            Psi_mod = torch.cat(
                (Psi_t_mod, torch.diag(Psi_nt)[..., np.newaxis]),
                dim=0)

            LL_Psi_ratio = LL_mod / Psi_mod
            LL_Psi_ratio_mask = \
                torch.ones(LL_mod.size()) - torch.eye(LL_mod.size()[0])

            # target ratio
            loss1 = (LL_Psi_ratio[0, 0] - torch.mean(LL_Psi_ratio[1:, 1:]))**2

            sums = torch.mean(LL_Psi_ratio * LL_Psi_ratio_mask.double(), dim=1)
            loss2 = (sums[0] - torch.mean(sums[1:]))**2

            loss = loss1 + loss2

        elif constraint == 'coupling_norm_plus_covariance_ratio':
            L_mod = torch.cat((l_t_mod, L_nt), dim=1)
            LL_mod = torch.mm(L_mod.t(), L_mod)
            Psi_mod = torch.cat(
                (Psi_t_mod, torch.diag(Psi_nt)[..., np.newaxis]),
                dim=0)

            LL_Psi_ratio = LL_mod / Psi_mod
            LL_Psi_ratio_mask = \
                torch.ones(LL_mod.size()) - torch.eye(LL_mod.size()[0])

            # target ratio
            loss1 = (LL_Psi_ratio[0, 0] - torch.mean(LL_Psi_ratio[1:, 1:]))**2

            sums = torch.mean(LL_Psi_ratio * LL_Psi_ratio_mask.double(), dim=1)
            loss2 = (sums[0] - torch.mean(sums[1:]))**2

            loss3 = torch.sum(a_mod**2)

            loss = loss1 + loss2 + loss3

        elif constraint == 'noise_correlation':
            L_mod = torch.cat((l_t_mod, L_nt), dim=1)
            LL_mod = torch.mm(L_mod.t(), L_mod)
            Psi_mod = torch.cat((torch.flatten(Psi_t_mod), torch.diag(Psi_nt)))
            cov = torch.diag(Psi_mod) + LL_mod

            inv_var = (1. / torch.diag(cov))**(0.5)
            corr = cov * torch.ger(inv_var, inv_var)
            mask = torch.ones_like(corr) - torch.diag(torch.ones_like(corr[0]))
            masked_corr = corr * mask
            means = torch.mean(masked_corr, dim=1)
            loss = (means[0] - torch.mean(means[1:]))**2

        elif constraint == 'noise_correlation_w_mask':
            L_mod = torch.cat((l_t_mod, L_nt), dim=1)
            LL_mod = torch.mm(L_mod.t(), L_mod)
            Psi_mod = torch.cat((torch.flatten(Psi_t_mod), torch.diag(Psi_nt)))
            cov = torch.diag(Psi_mod) + LL_mod

            inv_var = (1. / torch.diag(cov))**(0.5)
            corr = cov * torch.ger(inv_var, inv_var)
            mask = torch.ones_like(corr) - torch.diag(torch.ones_like(corr[0]))
            masked_corr = corr * mask
            means = torch.mean(masked_corr, dim=1)
            loss = (means[0] - torch.mean(means[1:]))**2
            mask_loss = torch.sum(((1 - a_mask) * a_mod)**2) \
                + torch.sum(((1 - b_mask) * b_mod)**2)
            loss += 0.5 * mask_loss

        loss.backward()
        grad = delta.grad.detach().numpy()
        return loss.detach().numpy(), grad

    @staticmethod
    def f_df_oracle(delta, L_nt, Psi_nt, a, b, B, a_true, b_true):
        delta = torch.tensor(delta[..., np.newaxis], requires_grad=True)
        L_nt = torch.tensor(L_nt)
        Psi_nt = torch.tensor(Psi_nt)
        a = torch.tensor(a)
        b = torch.tensor(b)
        B = torch.tensor(B)
        a_true = torch.tensor(a_true)
        b_true = torch.tensor(b_true)

        Delta = torch.solve(-1 * torch.mm(L_nt.t(), delta),
                            Psi_nt + torch.mm(L_nt.t(), L_nt))[0]

        a_mod = a + Delta
        b_mod = b - torch.mm(B, Delta)
        loss = torch.sum((a_true - a_mod)**2) + torch.sum((b_true - b_mod)**2)

        loss.backward()
        grad = delta.grad.detach().numpy()
        return loss.detach().numpy(), grad

    @staticmethod
    def mll(params, X, Y, y, K, a_mask, b_mask, B_mask, Psi_transform='softplus'):
        """Calculate the marginal log-likelihood of the provided data."""
        # storage for joint mean and covariance matrices
        D, M = X.shape
        N = Y.shape[1]

        mu = np.zeros((D, N + 1))
        sigma = np.zeros((N + 1, N + 1))

        a = params[:N].reshape((N, 1)) * a_mask
        b = params[N:(N + M)].reshape((M, 1)) * b_mask
        B = params[(N + M):(N + M + M * N)].reshape((M, N)) * B_mask
        Psi_tr = params[(N + M + M * N):
                        (N + M + M * N + N + 1)].reshape(N + 1)
        L = params[(N + M + N * M + N + 1):].reshape((K, N + 1))

        # mean of marginal
        mu[:, 0] = np.dot(X, b + np.dot(B, a)).squeeze()
        mu[:, 1:] = np.dot(X, B)

        # combine data matrices
        Y = np.concatenate((y, Y), axis=1)
        # unravel coupling terms for easy products
        a = a.ravel()
        # private variances
        Psi = utils.Psi_tr_to_Psi(Psi_tr, Psi_transform)
        Psi_t = Psi[0]
        Psi_nt = Psi[1:]
        # bases
        l_t = L[:, 0]
        L_nt = L[:, 1:]

        # useful terms to store for later
        coupled_L = l_t + np.dot(L_nt, a)
        cross_coupling = Psi_nt * a + np.dot(L_nt.T, coupled_L)

        # fill covariance matrix
        sigma[0, 0] = Psi_t + np.dot(Psi_nt, a**2) + np.dot(coupled_L, coupled_L)
        sigma[1:, 0] = cross_coupling
        sigma[0, 1:] = cross_coupling
        sigma[1:, 1:] = np.diag(Psi_nt) + np.dot(L_nt.T, L_nt)

        # calculate log-likelihood
        residual = Y - mu
        ll = -D / 2. * np.linalg.slogdet(sigma)[1] \
            + -0.5 * np.sum(residual.T * np.linalg.solve(sigma, residual.T))

        return ll

    @staticmethod
    def expected_complete_ll(
        params, X, Y, y, mu, zz, sigma, c_coupling=0., c_tuning=0., a_mask=None,
        b_mask=None, B_mask=None, transform_tuning=False, Psi_transform='softplus',
        penalize_B=False
    ):
        """Calculate the expected complete log-likelihood."""
        D, M = X.shape
        N = Y.shape[1]
        K = mu.shape[1]

        # extract parameters
        a = params[:N]
        b = params[N:(N + M)]
        B = params[(N + M):(N + M + M * N)].reshape((M, N))
        Psi_tr = params[(N + M + M * N):
                        (N + M + M * N + N + 1)]
        L = params[(N + M + N * M + N + 1):].reshape((K, N + 1))

        # split up terms into target/non-target components
        Psi = utils.Psi_tr_to_Psi(Psi_tr, Psi_transform)
        Psi_t, Psi_nt = np.split(Psi, [1])
        Psi_t = Psi_t.item()
        l_t, L_nt = np.split(L, [1], axis=1)
        l_t = l_t.ravel()

        # check masks
        if a_mask is None:
            a_mask = np.ones(N)
        elif a_mask.ndim == 2:
            a_mask = a_mask.ravel()

        if b_mask is None:
            b_mask = np.ones(M)
        elif b_mask.ndim == 2:
            b_mask = np.ones(M)

        if B_mask is None:
            B_mask = np.ones_like(B_mask)

        # apply masks
        a = a.ravel() * a_mask
        b = b.ravel() * b_mask
        B = B * B_mask

        # get original params if incoming are transformed
        if transform_tuning:
            tuning_to_coupling_ratio = c_tuning / c_coupling
            b = b / tuning_to_coupling_ratio
            if penalize_B:
                B = B / tuning_to_coupling_ratio

        y_residual = y.ravel() - X @ b - Y @ a
        Y_residual = Y - X @ B

        muL = mu @ L_nt

        term1 = np.sum(np.log(Psi))
        term2 = np.mean(y_residual**2 / Psi_t)
        term3 = np.mean((-2. / Psi_t) * y_residual * (mu @ l_t))
        term4 = (1. / Psi_t) * np.mean(
            np.matmul(np.matmul(zz, l_t), l_t)
        )
        term5 = np.sum(Y_residual**2 / Psi_nt.T) / D
        term6 = -2 * np.sum(Y_residual * muL / Psi_nt.T) / D
        term7a = np.trace(L_nt @ (L_nt.T / Psi_nt[..., np.newaxis]) @ sigma)
        term7b = np.sum(muL**2 / Psi_nt.T) / D

        loss = term1 + term2 + term3 + term4 + term5 + term6 + term7a + term7b
        # add sparsity penalty
        loss += c_coupling * np.linalg.norm(a, ord=1) + \
            c_tuning * (np.linalg.norm(b, ord=1) + np.linalg.norm(B, ord=1))
        return loss
