import numpy as np
import torch

from neurobiases import solver_utils as utils
from scipy.optimize import minimize
from sklearn.utils import check_random_state


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
    log_Psi_nt : np.ndarray, shape (N,)
        The non-target private variances.
    log_Psi : np.ndarray, shape (N + 1,)
        The private variances. This variable takes precedence over
        log_Psi_nt.
    max_iter : int
        The maximum number of optimization iterations to perform.
    tol : float
        The tolerance with which the cease optimization.
    random_state : RandomState or int or None
        RandomState object, or int to create RandomState object.
    """
    def __init__(
        self, X, Y, y, K, a_mask=None, b_mask=None, B_mask=None,
        B=None, L_nt=None, L=None, log_Psi_nt=None, log_Psi=None,
        max_iter=1000, tol=1e-4, random_state=None
    ):
        # tuning and coupling design matrices
        self.X = X
        self.Y = Y
        # response vector
        self.y = y
        # number of latent factors
        self.K = K

        # optimization parameters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)

        # initialize parameter estimates
        self._init_params()
        # initialize masks
        self.set_masks(a_mask=a_mask, b_mask=b_mask, B_mask=B_mask)
        # initialize non-target tuning parameters
        self.freeze_B(B=B)
        # initialize variability parameters
        self.freeze_var_params(L_nt=L_nt, L=L, log_Psi_nt=log_Psi_nt, log_Psi=log_Psi)

    def _init_params(self):
        """Initialize parameter estimates. Requires that X, Y, and y are
        already initialized."""
        # dataset dimensions
        self.D, self.M = self.X.shape
        self.N = self.Y.shape[1]

        # initialize parameter estimates to be all zeros
        # coupling parameters
        self.a = np.zeros((self.N, 1))
        # tuning parameters
        self.b = np.zeros((self.M, 1))
        # non-target tuning parameters
        self.B = np.zeros((self.M, self.N))
        # private variances
        self.log_Psi = np.zeros(self.N + 1)
        # latent factors are initialized to be small, random values
        self.L = self.random_state.normal(loc=0., scale=0.1, size=(self.K, self.N + 1))

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

    def set_params(self, a=None, b=None, B=None, log_Psi=None, L=None):
        """Sets parameters equal to the provided parameter values.

        Parameters
        ----------
        a : np.ndarray, shape (N, 1)
            The coupling parameters.
        b : np.ndarray, shape (M, 1)
            The tuning parameters.
        B : np.ndarray, shape (M, N)
            The non-target tuning parameters.
        log_Psi : np.ndarray, shape (N + 1,)
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
        if log_Psi is not None:
            self.log_Psi = np.copy(log_Psi.reshape(self.N + 1))
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

    def freeze_var_params(self, L_nt=None, L=None, log_Psi_nt=None, log_Psi=None):
        """Sets all (or a subset of) the variance parameters, and freezes them
        so that they cannot be trained.

        Parameters
        ----------
        L_nt : np.ndarray, shape (K, N)
            The non-target latent factors.
        L : np.ndarray, shape (K, N+1)
            All latent factors. This variable takes precedence over L_nt.
        log_Psi_nt : np.ndarray, shape (N,)
            The non-target private variances.
        log_Psi : np.ndarray, shape (N + 1,)
            The private variances. This variable takes precedence over
            log_Psi_nt.
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
        if log_Psi_nt is not None:
            self.log_Psi_nt_init = log_Psi_nt
            self.log_Psi[1:] = log_Psi_nt
            self.train_log_Psi_nt = False
        else:
            self.train_log_Psi_nt = True
        # initialize all private variances
        if log_Psi is not None:
            self.log_Psi_init = log_Psi
            self.log_Psi = log_Psi
            self.train_log_Psi = False
        else:
            self.train_log_Psi = True

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

        if self.train_log_Psi_nt:
            self.log_Psi[1:] = self.log_Psi_nt_init
        else:
            self.log_Psi[1:] = np.zeros(self.N + 1)

        if self.train_log_Psi:
            self.log_Psi = self.log_Psi_init
        else:
            self.log_Psi = np.zeros(self.N + 1)

    def get_params(self):
        """Concatenates all parameters into a single vector.

        Returns
        -------
        params : np.ndarray, shape (N + M + N * M + N + 1 + K * (N + 1),)
            A vector containing all parameters concatenated together.
        """
        params = np.concatenate((self.a.ravel(),
                                 self.b.ravel(),
                                 self.B.ravel(),
                                 self.log_Psi.ravel(),
                                 self.L.ravel()))
        return params

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
        log_Psi : np.ndarray, shape (N + 1,)
            The private variances.
        L : np.ndarray, shape (K, N+1)
            The latent factors.
        """
        a = params[:self.N].reshape((self.N, 1))
        b = params[self.N:(self.N + self.M)].reshape((self.M, 1))
        B = params[(self.N + self.M):
                   (self.N + self.M + self.M * self.N)].reshape((self.M, self.N))
        log_Psi = params[(self.N + self.M + self.M * self.N):
                         (self.N + self.M + self.M * self.N + self.N + 1)].reshape(
            self.N + 1)
        L = params[(self.N + self.M + self.N * self.M + self.N + 1):].reshape(
            (self.K, self.N + 1))
        return a, b, B, log_Psi, L

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
                        log_Psi=self.log_Psi, L=self.L)
        return copy

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

        mll = utils.marginal_log_likelihood_linear_tm(
            X=X, Y=Y, y=y, a=self.a, b=self.b, B=self.B, L=self.L,
            Psi=np.exp(self.log_Psi), a_mask=self.a_mask, b_mask=self.b_mask,
            B_mask=self.B_mask
        )
        return mll

    def marginal_likelihood_hessian(self):
        """Calculates the hessian of the marginal likelihood."""
        params = self.get_params()
        n_params = params.size
        tparams = torch.tensor(params, requires_grad=True)
        a, b, B, log_Psi, L = EMSolver.split_tparams(
            tparams, self.N, self.M, self.K)

        Psi = torch.exp(log_Psi)
        Psi_t = Psi[0]
        Psi_nt = Psi[1:].reshape(self.N, 1)  # N x 1
        l_t = L[:, 0].reshape(self.K, 1)  # K x 1
        L_nt = L[:, 1:]  # K x N

        # data
        X = torch.tensor(self.X)  # D x M
        Y = torch.tensor(self.Y)  # D x N
        y = torch.tensor(self.y)  # D x 1
        y_cat = torch.cat((y, Y), dim=1)

        mu_t = torch.mm(X, b + torch.mm(B, a))  # D x 1
        mu_nt = torch.mm(X, B)  # D x N
        mu = torch.cat((mu_t, mu_nt), dim=1)
        coupled_L = l_t + torch.mm(L_nt, a)  # K x 1

        c0 = Psi_t + torch.mm(Psi_nt.t(), a**2) + torch.mm(coupled_L.t(), coupled_L)
        c = Psi_nt * a + torch.mm(L_nt.t(), coupled_L)  # N x 1
        C = torch.diag(torch.flatten(Psi_nt)) + torch.mm(L_nt.t(), L_nt)
        sigma = torch.cat((
            torch.cat((c0, c.t()), dim=1),
            torch.cat((c, C), dim=1)),
            dim=0)
        sigma_det = torch.det(sigma)

        residual = y_cat - mu
        ll = self.D / 2. * torch.log(sigma_det) \
            + 0.5 * torch.sum(residual.t() * torch.solve(residual.t(), sigma)[0])

        grads = torch.autograd.grad(ll, tparams, create_graph=True)[0]
        hessian = np.zeros((n_params, n_params))

        for idx, grad in enumerate(grads):
            hessian[idx] = \
                torch.autograd.grad(grad, tparams, retain_graph=True)[0].detach().numpy()

        return grads.detach().numpy(), hessian

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
        Psi = np.exp(self.log_Psi)
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

    def m_step(self, mu, zz, sigma, verbose=False):
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
        params = self.get_params()
        if verbose:
            # create callback function
            def callback(params):
                print(
                    'Expected complete log-likelihood:',
                    self.expected_complete_ll(params, self.X, self.Y, self.y,
                                              mu, zz, sigma)
                )
        else:
            callback = None

        # run m-step minimization
        optimize = minimize(
            self.f_df_em, x0=params,
            method='L-BFGS-B',
            args=(self.X, self.Y, self.y, self.a_mask, self.b_mask, self.B_mask,
                  self.train_B, self.train_L, self.train_L_nt, self.train_log_Psi_nt,
                  self.train_log_Psi, mu, zz, sigma),
            callback=callback,
            jac=True)

        # extract optimized parameters
        params = optimize.x
        return params

    def fit_em(self, verbose=False, mstep_verbose=False, mll_curve=False):
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

        # EM iteration loop: convergence if tolerance or maximum iterations
        # are reached
        while (del_ml > self.tol) and (iteration < self.max_iter):
            # run E-step
            mu, zz, sigma = self.e_step()
            # run M-step
            params = self.m_step(mu, zz, sigma, verbose=mstep_verbose)
            self.a, self.b, self.B, self.log_Psi, self.L = self.split_params(params)

            iteration += 1
            # update marginal log-likelihood
            current_mll = self.marginal_log_likelihood()
            # calculate fraction change in marginal log-likelihood
            del_ml = np.abs(current_mll - base_mll) / np.abs(base_mll)
            base_mll = current_mll
            mlls[iteration] = current_mll

            if verbose:
                print('Iteration %s, del=%0.9f, mll=%f' % (iteration, del_ml,
                                                           mlls[iteration]))

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
                               self.B_mask))
        else:
            callback = None

        optimize = minimize(
            self.f_df_ml, x0=params,
            method='L-BFGS-B',
            args=(self.X, self.Y, self.y, self.K,
                  self.a_mask, self.b_mask, self.B_mask,
                  self.train_B, self.train_L, self.train_L_nt,
                  self.train_log_Psi_nt, self.train_log_Psi),
            callback=callback,
            jac=True)
        params = optimize.x
        self.a, self.b, self.B, self.log_Psi, self.L = self.split_params(params)
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
        Psi = np.exp(self.log_Psi)
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

        self.log_Psi[0] = np.log(Psi_t + correction)
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
        # private variance
        Psi = np.exp(self.log_Psi)
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
        # private variance
        Psi = np.exp(self.log_Psi)
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
        cov = np.dot(self.L.T, self.L) + np.diag(np.exp(self.log_Psi))
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
        return np.exp(self.log_Psi) / np.diag(np.dot(self.L.T, self.L))

    def compare_tc_fits(self, lsem, fax=None, color='gray'):
        """Plot a comparison between the fitted and true parameters."""
        import matplotlib.pyplot as plt

        if fax is None:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        else:
            fig, axes = fax

        a = lsem.A
        b = lsem.Bi

        # plot tuning parameters
        axes[0].scatter(b.ravel(), self.b.ravel(), color=color, s=50)
        axes[0].set_xlim([1.5 * np.min([b, self.b]), 1.25 * np.max([b, self.b])])
        axes[0].set_ylim(axes[0].get_xlim())
        axes[0].plot(axes[0].get_xlim(), axes[0].get_ylim(), color='k')

        # plot coupling parameters
        axes[1].scatter(a.ravel(), self.a.ravel(), color=color, s=50)
        axes[1].set_xlim([1.5 * np.min([a, self.a]), 1.5 * np.max([a, self.a])])
        axes[1].set_ylim(axes[1].get_xlim())
        axes[1].plot(axes[1].get_xlim(), axes[1].get_ylim(), color='k')

        for ax in axes.ravel():
            ax.set_aspect('equal')
            ax.set_xlabel(r'\textbf{True}', fontsize=18)
            ax.set_ylabel(r'\textbf{Estimated}', fontsize=18)

        axes[0].set_title(r'\textbf{Tuning Parameters}', fontsize=20)
        axes[1].set_title(r'\textbf{Coupling Parameters}', fontsize=20)

        plt.tight_layout()

        return fig, axes

    def compare_fits(self, lsem, fax=None, color='gray'):
        """Plot a comparison between the fitted and true parameters."""
        import matplotlib.pyplot as plt

        if fax is None:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        else:
            fig, axes = fax

        a = lsem.A
        b = lsem.Bi
        B = lsem.Bj
        log_Psi = np.log(np.diag(lsem._Psi))
        LL = np.dot(lsem._L, lsem._L.T)[np.tril_indices(self.N + 1)]
        LL_hat = np.dot(self.L.T, self.L)[np.tril_indices(self.N + 1)]

        # plot tuning parameters
        axes[0, 0].scatter(b.ravel(), self.b.ravel(), color=color, s=50)
        axes[0, 0].set_xlim([1.5 * np.min([b, self.b]), 1.25 * np.max([b, self.b])])
        axes[0, 0].set_ylim(axes[0, 0].get_xlim())
        axes[0, 0].plot(axes[0, 0].get_xlim(), axes[0, 0].get_ylim(), color='k')

        # plot coupling parameters
        axes[0, 1].scatter(a.ravel(), self.a.ravel(), color=color, s=50)
        axes[0, 1].set_xlim([1.5 * np.min([a, self.a]), 1.5 * np.max([a, self.a])])
        axes[0, 1].set_ylim(axes[0, 1].get_xlim())
        axes[0, 1].plot(axes[0, 1].get_xlim(), axes[0, 1].get_ylim(), color='k')

        # plot non-target tuning parameters
        axes[1, 0].scatter(B.ravel(), self.B.ravel(), color=color)
        axes[1, 0].set_xlim([0.75 * np.min([B, self.B]), 1.25 * np.max([B, self.B])])
        axes[1, 0].set_ylim(axes[1, 0].get_xlim())
        axes[1, 0].plot(axes[1, 0].get_xlim(), axes[1, 0].get_ylim(), color='k')

        # plot private variances
        axes[1, 1].scatter(log_Psi, self.log_Psi.ravel(), color=color)
        axes[1, 1].set_xlim([0.75 * np.min([log_Psi, self.log_Psi.ravel()]),
                             1.25 * np.max([log_Psi, self.log_Psi.ravel()])])
        axes[1, 1].set_ylim(axes[1, 1].get_xlim())
        axes[1, 1].plot(axes[1, 1].get_xlim(), axes[1, 1].get_ylim(), color='k')

        # plot shared variability
        axes[0, 2].scatter(LL, LL_hat, color=color)
        axes[0, 2].set_xlim([np.min([LL, LL_hat]),
                             1.1 * np.max([LL, LL_hat])])
        axes[0, 2].set_ylim(axes[0, 2].get_xlim())
        axes[0, 2].plot(axes[0, 2].get_xlim(), axes[0, 2].get_xlim(), color='k')

        cov = np.diag(np.exp(log_Psi)) + np.dot(lsem._L, lsem._L.T)
        inv_var = (1. / np.diag(cov))**(0.5)
        corr = (cov * np.outer(inv_var, inv_var))[np.tril_indices(self.N + 1, k=-1)]
        corr_hat = self.create_corr()[np.tril_indices(self.N + 1, k=-1)]
        axes[1, 2].scatter(corr, corr_hat, color=color)
        axes[1, 2].set_xlim([-1, 1])
        axes[1, 2].set_ylim([-1, 1])
        axes[1, 2].plot(axes[1, 2].get_xlim(), axes[1, 2].get_xlim(), color='k')

        for ax in axes.ravel():
            ax.set_aspect('equal')
            ax.set_xlabel(r'\textbf{True}', fontsize=18)
            ax.set_ylabel(r'\textbf{Estimated}', fontsize=18)

        axes[0, 0].set_title(r'\textbf{Tuning Parameters}', fontsize=20)
        axes[0, 1].set_title(r'\textbf{Coupling Parameters}', fontsize=20)
        axes[1, 0].set_title(r'\textbf{Non-target Tuning}', fontsize=20)
        axes[1, 1].set_title(r'\textbf{Private Variances}', fontsize=20)
        axes[0, 2].set_title(r'\textbf{Shared (Co)-variances}', fontsize=20)
        axes[1, 2].set_title(r'\textbf{Shared Correlations}', fontsize=20)

        plt.tight_layout()

        return fig, axes

    @staticmethod
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

        log_Psi : torch.tensor, shape (N + 1,)
            The private variances.

        L : torch.tensor, shape (K, N+1)
            The latent factors.
        """
        a = tparams[:N].reshape(N, 1)
        b = tparams[N:(N + M)].reshape(M, 1)
        B = tparams[(N + M):(N + M + N * M)].reshape(M, N)
        log_Psi = tparams[(N + M + N * M):(N + M + N * M + N + 1)].reshape(N + 1, 1)
        L = tparams[(N + M + N * M + N + 1):].reshape(K, N + 1)

        return a, b, B, log_Psi, L

    @staticmethod
    def f_df_em(params, X, Y, y, a_mask, b_mask, B_mask, train_B, train_L_nt,
                train_L, train_log_Psi_nt, train_log_Psi, mu, zz, sigma):
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

        train_log_Psi_nt : bool
            If True, non-target private variances will be trained.

        train_log_Psi : bool
            If True, private variances will be trained. Takes precedence over
            train_log_Psi_nt.

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
        a, b, B, log_Psi, L = EMSolver.split_tparams(tparams, N, M, K)
        # split up terms into target/non-target components
        Psi = torch.exp(log_Psi)
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
        term1 = torch.sum(log_Psi)
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
        if not train_log_Psi_nt:
            grad[(N + M + N * M + 1):(N + M + N * M + N + 1)] = 0
        if not train_log_Psi:
            grad[(N + M + N * M):(N + M + N * M + N + 1)] = 0
        if not train_L_nt:
            mask = np.zeros(grad[(N + M + N * M + N + 1):].size)
            mask[0::(N + 1)] = np.ones(K)
            grad[(N + M + N * M + N + 1):] *= mask
        if not train_L:
            grad[(N + M + N * M + N + 1):] = 0

        return loss.detach().numpy(), grad

    @staticmethod
    def f_df_ml(
        params, X, Y, y, K, a_mask, b_mask, B_mask, train_B, train_L_nt, train_L,
        train_log_Psi_nt, train_log_Psi
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

        train_log_Psi_nt : bool
            If True, non-target private variances will be trained.

        train_log_Psi : bool
            If True, private variances will be trained. Takes precedence over
            train_log_Psi_nt.

        Returns
        -------
        loss : float
            The marginal log-likelihood.

        grad : np.ndarray
            The gradient of the loss with respect to all parameters. Any
            variables whose training flag was set to False will have corresponding
            gradients set equal to zero.
        """
        # extract dimensions
        D, M = X.shape
        N = Y.shape[1]
        # turn parameters into torch tensors
        tparams = torch.tensor(params, requires_grad=True)
        a, b, B, log_Psi, L = EMSolver.split_tparams(tparams, N, M, K)
        a = a * torch.tensor(a_mask, dtype=a.dtype)
        b = b * torch.tensor(b_mask, dtype=b.dtype)
        B = B * torch.tensor(B_mask, dtype=B.dtype)
        # split up terms into target/non-target components
        Psi = torch.exp(log_Psi)
        Psi_t = Psi[0]
        Psi_nt = Psi[1:].reshape(N, 1)  # N x 1
        l_t = L[:, 0].reshape(K, 1)  # K x 1
        L_nt = L[:, 1:]  # K x N

        # turn data into torch tensors
        X = torch.tensor(X)  # D x M
        Y = torch.tensor(Y)  # D x N
        y = torch.tensor(y)  # D x 1
        y_cat = torch.cat((y, Y), dim=1)

        # useful terms for the marginal log-likelihood
        mu_t = torch.mm(X, b + torch.mm(B, a))  # D x 1
        mu_nt = torch.mm(X, B)  # D x N
        mu = torch.cat((mu_t, mu_nt), dim=1)
        coupled_L = l_t + torch.mm(L_nt, a)  # K x 1

        # calculate marginal log-likelihood
        # see paper for derivation
        c0 = Psi_t + torch.mm(Psi_nt.t(), a**2) + torch.mm(coupled_L.t(), coupled_L)
        c = Psi_nt * a + torch.mm(L_nt.t(), coupled_L)  # N x 1
        C = torch.diag(torch.flatten(Psi_nt)) + torch.mm(L_nt.t(), L_nt)
        sigma = torch.cat((
            torch.cat((c0, c.t()), dim=1),
            torch.cat((c, C), dim=1)),
            dim=0)
        sigma_det = torch.det(sigma)

        # calculate loss and perform autograd
        residual = y_cat - mu
        ll = D / 2. * torch.log(sigma_det) \
            + 0.5 * torch.sum(residual.t() * torch.solve(residual.t(), sigma)[0])
        ll.backward()
        loss = np.asscalar(ll.detach().numpy())
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
        if not train_log_Psi_nt:
            grad[(N + M + N * M + 1):(N + M + N * M + N + 1)] = 0
        if not train_log_Psi:
            grad[(N + M + N * M):(N + M + N * M + N + 1)] = 0
        if not train_L:
            grad[(N + M + N * M + N + 1):] = 0
        if not train_L_nt:
            mask = np.zeros(grad[(N + M + N * M + N + 1):].size)
            mask[0::(N + 1)] = np.ones(K)
            grad[(N + M + N * M + N + 1):] *= mask

        return loss, grad

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
    def mll(params, X, Y, y, K, a_mask, b_mask, B_mask):
        """Calculate the marginal log-likelihood of the provided data."""
        # storage for joint mean and covariance matrices
        D, M = X.shape
        N = Y.shape[1]

        mu = np.zeros((D, N + 1))
        sigma = np.zeros((N + 1, N + 1))

        a = params[:N].reshape((N, 1)) * a_mask
        b = params[N:(N + M)].reshape((M, 1)) * b_mask
        B = params[(N + M):(N + M + M * N)].reshape((M, N)) * B_mask
        log_Psi = params[(N + M + M * N):
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
        Psi = np.exp(log_Psi)
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
    def expected_complete_ll(params, X, Y, y, mu, zz, sigma):
        """Calculate the expected complete log-likelihood."""
        D, M = X.shape
        N = Y.shape[1]
        K = mu.shape[1]

        # extract parameters
        tparams = torch.tensor(params, requires_grad=False)
        a = tparams[:N].reshape(N, 1)
        b = tparams[N:(N + M)].reshape(M, 1)
        B = tparams[(N + M):(N + M + N * M)].reshape(M, N)
        log_Psi = tparams[(N + M + N * M):(N + M + N * M + N + 1)].reshape(N + 1, 1)
        L = tparams[(N + M + N * M + N + 1):].reshape(K, N + 1)

        Psi = torch.exp(log_Psi)
        Psi_nt = Psi[1:]
        l_t = L[:, 0].reshape(K, 1)
        L_nt = L[:, 1:]

        # data
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        y = torch.tensor(y)
        mu = torch.tensor(mu)
        zz = torch.tensor(zz)
        sigma = torch.tensor(sigma)

        y_residual = y - torch.mm(X, b) - torch.mm(Y, a)
        Y_residual = Y - torch.mm(X, B)

        muL = torch.mm(mu, L_nt)

        term1 = torch.sum(log_Psi)
        term2 = torch.mean(y_residual**2 / Psi[0])
        term3 = torch.mean((-2. / Psi[0]) * y_residual * torch.mm(mu, l_t))
        term4 = torch.mean(
            torch.matmul(torch.transpose(torch.matmul(zz, l_t), 1, 2), l_t)) / Psi[0]
        term5 = torch.sum(Y_residual**2 / Psi_nt.t()) / D
        term6 = -2 * torch.sum(y_residual * muL / Psi_nt.t()) / D
        term7a = torch.trace(torch.chain_matmul(L_nt, L_nt.t() / Psi_nt, sigma))
        term7b = torch.sum(muL**2 / Psi_nt.t()) / D

        loss = term1 + term2 + term3 + term4 + term5 + term6 + term7a + term7b

        return loss.detach().numpy()
