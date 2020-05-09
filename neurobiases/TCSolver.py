import numpy as np

from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression


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
    def __init__(self, X, Y, y, a_mask=None, b_mask=None):
        # tuning and coupling design matrices
        self.X = X
        self.Y = Y
        # response vector
        self.y = y
        # initialize parameter estimates
        self._init_params()
        # initialize masks
        self.set_masks(a_mask=a_mask, b_mask=b_mask)

    def _init_params(self):
        """Initialize parameter estimates. Requires that X, Y, and y are
        already initialized."""
        # dataset dimensions
        self.D, self.M = self.X.shape
        self.N = self.Y.shape[1]

        # initialize parameter estimates to be all zeros
        # coupling parameters
        self.a = np.zeros(self.N)
        # tuning parameters
        self.b = np.zeros(self.M)

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
        # coupling parameters mask
        if a_mask is None:
            self.a_mask = np.ones(self.N).astype(bool)
        else:
            self.a_mask = a_mask.ravel()

        # tuning parameters mask
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
        # initialize storage
        a_hat = np.zeros(self.N)
        b_hat = np.zeros(self.M)
        # apply masks
        X = self.X[:, self.b_mask]
        Y = self.Y[:, self.a_mask]
        # form design matrix
        Z = np.concatenate((X, Y), axis=1)
        # fit OLS
        ols = LinearRegression(fit_intercept=False)
        ols.fit(Z, self.y.ravel())
        # extract fits into masked arrays
        b_hat[self.b_mask], a_hat[self.a_mask] = np.split(ols.coef_, [self.n_nonzero_tuning])
        return a_hat, b_hat

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
        breakpoint()
        b_hat[self.b_mask], a_hat[self.a_mask] = np.split(coefs, [self.n_nonzero_tuning])
        return a_hat, b_hat
