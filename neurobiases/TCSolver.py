import numpy as np

from .lbfgs import fmin_lbfgs
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, Lasso


class TCSolver():
    """Class to perform a tuning + coupling fit to data generated from
    the triangular model.

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
    """
    def __init__(
        self, X, Y, y, a_mask=None, b_mask=None, solver='ow_lbfgs', c_tuning=0.,
        c_coupling=0., initialization='random', max_iter=1000, tol=1e-4,
        rng=None
    ):
        # Tuning and coupling design matrices
        self.X = X
        self.Y = Y
        # Response vector
        self.y = y
        # Random number generator
        self.rng = np.random.default_rng(rng)
        # Initialize parameter estimates
        self.initialization = initialization
        self._init_params()
        # Other settings
        self.max_iter = max_iter
        self.tol = tol
        # Initialize masks
        self.set_masks(a_mask=a_mask, b_mask=b_mask)
        # Optimization parameters
        self.solver = solver
        self.c_tuning = c_tuning
        self.c_coupling = c_coupling

    def _init_params(self):
        """Initialize parameter estimates. Requires that X, Y, and y are
        already initialized."""
        # Dataset dimensions
        self.D, self.M = self.X.shape
        self.N = self.Y.shape[1]

        # Initialize parameter estimates
        if self.initialization == 'zeros':
            self.a = np.zeros(self.N)
            self.b = np.zeros(self.M)
        elif self.initialization == 'random':
            self.a = self.rng.normal(loc=0, scale=1, size=self.N)
            self.b = self.rng.normal(loc=0, scale=1, size=self.M)
        else:
            raise ValueError('Incorrect initialization input.')

    def get_params(self):
        """Gets the parameter values in one vector."""
        params = np.concatenate((self.a, self.b))
        return params

    def split_params(self, params):
        """Splits a parameter vector into the coupling and tuning parameters."""
        a, b = np.split(params, [self.N])
        return a, b

    def set_masks(self, a_mask=None, b_mask=None):
        """Initialize masks. A value of None indicates that all features will
        be included in the mask.

        Parameters
        ----------
        a_mask : np.ndarray, shape (N, 1)
            Mask for coupling features.
        b_mask : nd-array, shape (M, 1)
            Mask for tuning features.
        """
        # Coupling mask
        if a_mask is None:
            self.a_mask = np.ones(self.N).astype(bool)
        else:
            self.a_mask = a_mask.ravel()

        # Tuning mask
        if b_mask is None:
            self.b_mask = np.ones(self.M).astype(bool)
        else:
            self.b_mask = b_mask.ravel()

        self.n_nonzero_coupling = self.a_mask.sum()
        self.n_nonzero_tuning = self.b_mask.sum()

    def fit_ols(self):
        """Fit Ordinary Least Squares to the data.

        Returns
        -------
        a_hat : np.ndarray, shape (N,)
            The fitted coupling parameters.
        b_hat : nd-array, shape (M,)
            The fitted tuning parameters.
        """
        # Apply masks to datasets
        X = self.X[:, self.b_mask]
        Y = self.Y[:, self.a_mask]
        # Form total design matrix
        Z = np.concatenate((X, Y), axis=1)
        # Edge case when masks are empty
        if Z.shape[1] == 0:
            self.a = np.zeros(self.N)
            self.b = np.zeros(self.M)
            return self
        # Perform OLS fit
        ols = LinearRegression(fit_intercept=False)
        ols.fit(Z, self.y.ravel())
        # Extract fits into class variables
        b_est, a_est = np.split(ols.coef_, [self.n_nonzero_tuning])
        self.a[self.a_mask] = a_est
        self.b[self.b_mask] = b_est
        return self

    def fit_nnls(self):
        """Fit non-negative least squares to the data.

        Returns
        -------
        a_hat : np.ndarray, shape (N,)
            The fitted coupling parameters.
        b_hat : nd-array, shape (M,)
            The fitted tuning parameters.
        """
        # initialize storage
        a_hat = np.zeros(self.N)
        b_hat = np.zeros(self.M)
        # apply masks
        X = self.X[:, self.b_mask]
        Y = self.Y[:, self.a_mask]
        # form design matrix
        Z = np.concatenate((X, Y), axis=1)
        # fit OLS
        coefs, _ = nnls(Z, self.y.ravel())
        # extract fits into masked arrays
        b_hat[self.b_mask], a_hat[self.a_mask] = np.split(coefs, [self.n_nonzero_tuning])
        return a_hat, b_hat

    def fit_lasso(self, refit=False, verbose=False):
        """Fit a lasso regression to the data, using separate penalities on the
        tuning and coupling parameters.

        Parameters
        ----------
        verbose : bool
            If True, print callback statement at each iteration.

        Returns
        -------
        a_hat : np.ndarray, shape (N,)
            The fitted coupling parameters.
        b_hat : nd-array, shape (M,)
            The fitted tuning parameters.
        """
        params = self.get_params()
        # Use orthant-wise lbfgs solver
        if self.solver == 'cd':
            if self.c_coupling != 0 and self.c_tuning != 0:
                # Create scaling matrix for data
                lambdas = np.diag(1. / np.concatenate((
                    np.repeat(self.c_tuning, self.N),
                    np.repeat(self.c_coupling, self.M)
                )))
                # Form design matrix and rescale
                Z = np.concatenate((self.X, self.Y), axis=1)
                Zpr = Z @ lambdas
                # Apply Lasso
                solver = Lasso(
                    alpha=1.0,
                    fit_intercept=False,
                    normalize=False,
                    max_iter=self.max_iter,
                    tol=self.tol)
                solver.fit(Zpr, self.y.ravel())
                # Rescale coefficients back
                b, a = np.split(lambdas @ solver.coef_, [self.M])
            elif self.c_tuning == 0 and self.c_coupling != 0:
                raise NotImplementedError()
            elif self.c_coupling == 0 and self.c_tuning != 0:
                raise NotImplementedError()
            else:
                return self.fit_ols()

        elif self.solver == 'ow_lbfgs':
            # Create callable for verbosity
            if verbose:
                # create callback function
                def progress(x, g, fx, xnorm, gnorm, step, k, num_eval, *args):
                    # x is in the transformed space
                    print(
                        'Loss:',
                        self.f_df_tc(x, self.X, self.Y, self.y, self.a_mask,
                                     self.b_mask, 1)[0]
                    )
            else:
                progress = None

            # Penalize both coupling and tuning
            if self.c_coupling != 0 and self.c_tuning != 0:
                orthantwise_start = 0
                orthantwise_end = self.N + self.M
                tuning_to_coupling_ratio = float(self.c_tuning) / self.c_coupling
                c = self.c_coupling
                # Transform tuning parameters
                params[self.N:orthantwise_end] *= tuning_to_coupling_ratio
            # Penalize only coupling
            elif self.c_tuning == 0 and self.c_coupling != 0:
                orthantwise_start = 0
                orthantwise_end = self.N
                tuning_to_coupling_ratio = 1.
                c = self.c_coupling
            # Penalize only tuning
            elif self.c_coupling == 0 and self.c_tuning != 0:
                orthantwise_start = self.N
                orthantwise_end = self.N + self.M
                tuning_to_coupling_ratio = 1.
                c = self.c_tuning
            # Penalize neither
            else:
                orthantwise_start = 0
                orthantwise_end = -1
                c = 0
                tuning_to_coupling_ratio = 1.

            # Tuning parameters are transformed
            params = fmin_lbfgs(
                self.f_df_tc_owlbfgs, x0=params,
                args=(self.X, self.Y, self.y, self.a_mask, self.b_mask,
                      tuning_to_coupling_ratio),
                progress=progress,
                orthantwise_c=c,
                orthantwise_start=orthantwise_start,
                orthantwise_end=orthantwise_end)
            # Tuning parameters are transformed back
            a, b = self.split_params(params)
            b = b / tuning_to_coupling_ratio
        self.a = a
        self.b = b
        self.set_masks(a_mask=self.a != 0, b_mask=self.b != 0)
        # Perform a refitting using OLS, if necessary
        if refit:
            return self.fit_ols()
        else:
            return self

    def mse(self, X=None, Y=None, y=None):
        """Calculate the mean-squared error given the fitted parameters, either
        on the initialized dataset, or a new one.

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
        mse : float
            The mean-squared error.
        """
        # If no data is provided, use the data in the object
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if y is None:
            y = self.y
        # Calculate mean squared error
        mse = np.sum((y.ravel() - X @ self.b - Y @ self.a)**2)
        return mse

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
        # If no data is provided, use the data in the object
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if y is None:
            y = self.y

        D = X.shape[0]
        # Calculate mean squared error
        mse = -self.mse(X=X, Y=Y, y=y)
        # Add in penalty for model size
        k = np.count_nonzero(self.a) + np.count_nonzero(self.b)
        bic = -2 * mse + k * np.log(D)
        return bic

    @staticmethod
    def f_df_tc(params, X, Y, y, a_mask, b_mask, tuning_to_coupling_ratio):
        """Calculates loss and gradient for linear tuning and coupling model."""
        # Dataset dimensions
        N = Y.shape[1]
        # Transform tuning parameters
        a, b = np.split(params, [N])
        b = b / tuning_to_coupling_ratio
        # Calculate residual and loss
        residual = y.ravel() - X @ b - Y @ a
        loss = np.sum(residual**2)
        # Calculate gradients, applying mask if necessary
        b_grad = -2 * (X.T @ residual)
        b_grad[np.invert(b_mask)] = 0
        a_grad = -2 * (Y.T @ residual)
        a_grad[np.invert(a_mask)] = 0
        # Put together gradients into final gradient
        grad = np.concatenate((a_grad, b_grad))
        return loss, grad

    @staticmethod
    def f_df_tc_owlbfgs(params, grad, *args):
        """Wrapper for OW LBFGS"""
        loss, g = TCSolver.f_df_tc(params, *args)
        grad[:] = g
        return loss
