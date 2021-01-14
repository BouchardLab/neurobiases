import numpy as np

from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class ITSFASolver():
    """Class to perform ITSFA, an inference method for data obtained from the
    triangular model.

    Currently does not support intercepts. Data must be zero-centered before
    passing into the solver.

    Parameters
    ----------
    X : np.ndarray, shape (D, M)
        Design matrix for tuning features.
    Y : np.ndarray, shape (D, N)
        Design matrix for coupling features.
    y : np.ndarray, shape (D, 1)
        Neural response vector.
    K : int or None
        The number of latent factors. If None, K will be chosen through
        cross-validation.
    B : np.ndarray, shape (M, N)
        The non-target tuning parameters. Optional.
    a_mask : np.ndarray, shape (N, 1)
        Mask for coupling features.
    b_mask : nd-array, shape (M, 1)
        Mask for tuning features.
    max_iter : int
        The maximum number of ITSFA iterations.
    tol : float
        The (parameter) convergence tolerance for ITSFA.
    fa_max_iter : int
        The maximum number of iterations for the factor analyses.
    fa_tol : float
        The convergence tolerance for the factor analyses.
    K_max : int
        The maximum number of latent factors to cross-validate over, if K is
        not provided.
    K_splits : int
        The number of cross-validation folds for determining the size of the
        latent state.
    r2_convergence : bool
        Whether to use convergence in explained variance as a stopping criteria.
    r2_convergence_tol : float
        The tolerance for R2 convergence.

    Attributes
    ----------
    a_iters : np.ndarray
        The estimated coupling parameters for each ITSFA iteration.
    b_iters : np.ndarray
        The estimated tuning parameters for each ITSFA iteration.
    a : np.ndarray
        The estimated coupling parameters.
    b : np.ndarray
        The estimated tuning parameters.
    """
    def __init__(
        self, X, Y, y, K=None, B=None, a_mask=None, b_mask=None, max_iter=30,
        tol=1e-3, fa_max_iter=2000, fa_tol=1e-4, K_max=20, K_splits=5,
        r2_convergence=True, r2_convergence_tol=1e-7
    ):
        # Triangular model data
        self.X = X
        self.Y = Y
        self.y = y
        self.K = K
        # Non-target tuning parameters: optional
        self.B = B
        # Fitting settings
        self.max_iter = max_iter
        self.tol = tol
        self.fa_max_iter = fa_max_iter
        self.fa_tol = fa_tol
        self.K_max = K_max
        self.K_splits = K_splits
        self.r2_convergence = r2_convergence
        self.r2_convergence_tol = r2_convergence_tol
        # Dataset dimensions
        self.D, self.M = self.X.shape
        self.N = self.Y.shape[1]
        # Initialize parameter estimates
        self._init_params()
        # Initialize masks
        self.set_masks(a_mask=a_mask, b_mask=b_mask)

    def _init_params(self):
        """Initialize parameter estimates."""
        self.a = np.zeros(self.N)
        self.b = np.zeros(self.M)

        if self.B is None:
            ols = LinearRegression(fit_intercept=False)
            ols.fit(self.X, self.Y)
            self.B = ols.coef_

    def set_masks(self, a_mask=None, b_mask=None):
        """Initialize masks for the tuning and coupling parameters.

        If no mask is provided, all parameters will be used.

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

    def fit_itsfa(self, verbose=False):
        """Performs inference on triangular model data using ITSFA.

        Parameters
        ----------
        verbose : bool
            If True, print out progress statements.
        """
        X = np.copy(self.X)
        Y = np.copy(self.Y)
        y = np.copy(self.y)
        if y.ndim > 1:
            y = y.ravel()

        a_mask = self.a_mask.ravel()
        b_mask = self.b_mask.ravel()
        X_sel = X[:, b_mask]

        # Storage arrays
        self.a_iters = np.zeros((self.max_iter + 1, self.N))
        self.b_iters = np.zeros((self.max_iter + 1, self.M))
        r2_scores = np.zeros(self.max_iter + 1)

        # Handle case where B is not provided
        Y_hat = np.dot(X, self.B)
        Y_noise = Y - Y_hat

        # Cross-validate the first stage factor analysis
        if self.K is None:
            kf = KFold(n_splits=self.K_splits, shuffle=True)
            scores = np.zeros((self.K_splits, self.K_max))

            # Iterate over folds and latent factors
            for split_idx, (train_idx, test_idx), in enumerate(kf.split(Y_noise)):
                for latent_idx, n_latent in enumerate(range(1, self.K_max + 1)):
                    # Extract train and test data
                    Y_noise_train = Y_noise[train_idx]
                    Y_noise_test = Y_noise[test_idx]
                    # Fit factor analysis
                    fa = FactorAnalysis(
                        n_components=n_latent,
                        tol=self.fa_tol,
                        max_iter=self.fa_max_iter)
                    fa.fit(Y_noise_train)
                    # Score on test set
                    scores[split_idx, latent_idx] = fa.score(Y_noise_test)
            K = np.argmax(np.mean(scores, axis=0)) + 1
        else:
            K = self.K

        # First stage factor analysis
        fa1 = FactorAnalysis(
            n_components=K,
            tol=self.fa_tol,
            max_iter=self.fa_max_iter
        ).fit(Y_noise)
        # Subtract out projection of shared variability into Yj space
        Y_prime = Y - np.dot(fa1.transform(Y_noise), fa1.components_)
        Y_prime_sel = Y_prime[:, a_mask]

        # First stage regression
        ols = LinearRegression(fit_intercept=False)
        Z = np.concatenate((X_sel, Y_prime_sel), axis=1)
        ols.fit(Z, y.ravel())
        # Extract first fit
        self.b_iters[0, b_mask], self.a_iters[0, a_mask] = \
            np.split(ols.coef_, [self.n_nonzero_tuning])

        # Obtain estimates and score
        y_hat = \
            np.dot(Y_prime_sel, self.a_iters[0, a_mask]) + \
            np.dot(X_sel, self.b_iters[0, b_mask])
        r2_scores[0] = r2_score(y.ravel(), y_hat)

        # Iterated second stage
        for idx in range(self.max_iter):
            if verbose:
                print(f'Iteration {idx}.')
            # Obtain target neuron residuals
            y_noise = y - y_hat
            # Gather all residuals
            total_noise = np.insert(Y_noise, 0, y_noise, axis=1)

            # Extract shared variability from all residuals, using the number
            # of latent dimensions obtained in the first stage
            fa2 = FactorAnalysis(
                n_components=K,
                tol=self.fa_tol,
                max_iter=self.fa_max_iter)
            fa2.fit(total_noise)
            shared_variability = fa2.transform(total_noise) @ fa2.components_
            # Obtain modified dataset
            y_prime = y - shared_variability[:, 0]

            # Perform second stage regression
            Z = np.concatenate((X_sel, Y_prime_sel), axis=1)
            ols = LinearRegression(fit_intercept=False)
            ols.fit(Z, y_prime)

            # Extract estimates
            self.b_iters[idx + 1, b_mask], self.a_iters[idx + 1, a_mask] \
                = np.split(ols.coef_, [self.n_nonzero_tuning])

            # Obtain estimates and score
            y_hat = \
                np.dot(Y_prime_sel, self.a_iters[idx + 1, a_mask]) + \
                np.dot(X_sel, self.b_iters[idx + 1, b_mask])
            r2_scores[idx + 1] = r2_score(y, y_hat)

            # Check for convergence in parameter estimates
            if self.n_nonzero_coupling > 0:
                delta_a = np.mean(
                    np.abs(self.a_iters[idx + 1, a_mask] - self.a_iters[idx, a_mask])
                )
            else:
                delta_a = 0
            delta_b = np.mean(
                np.abs(self.b_iters[idx + 1, b_mask] - self.b_iters[idx, b_mask])
            )

            if delta_b < self.tol and delta_a < self.tol:
                break

            # Check for convergence in explained variance
            if self.r2_convergence:
                delta_r2 = np.abs(r2_scores[idx + 1] - r2_scores[idx])
                if delta_r2 < self.r2_convergence_tol:
                    break

        self.n_iterations = idx + 1
        self.a_iters = self.a_iters[:self.n_iterations + 1]
        self.b_iters = self.b_iters[:self.n_iterations + 1]
        self.a = self.a_iters[self.n_iterations]
        self.b = self.b_iters[self.n_iterations]
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
