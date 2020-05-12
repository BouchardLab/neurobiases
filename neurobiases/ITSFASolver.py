import numpy as np

from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class ITSFASolver():
    def __init__(self, X, Y, y, B=None, a_mask=None, b_mask=None):
        # tuning and coupling design matrices
        self.X = X
        self.Y = Y
        # response vector
        self.y = y
        # non-target tuning
        self.B = B
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

    def fit_itsfa(
        self, K=None, max_iter=25, n_latent_max=20, n_fa_splits=5,
        tol=1e-3, r2_convergence=True, verbose=False
    ):
        """Calculates fits according to ITSFA."""
        X = np.copy(self.X)
        Y = np.copy(self.Y)
        y = np.copy(self.y)
        a_mask = self.a_mask.ravel()
        b_mask = self.b_mask.ravel()

        a_hat, b_hat, intercept = self.itsfa(
            X=X, Y=Y, y=y.ravel(),
            K=K,
            a_mask=a_mask,
            b_mask=b_mask,
            B_hat=self.B,
            B_hat_intercept=np.zeros(self.N))

        return a_hat, b_hat, intercept

    @staticmethod
    def itsfa(
        X, Y, y, a_mask, b_mask, B_hat, B_hat_intercept, K=None, max_iter=25,
        n_latent_max=20, n_fa_splits=5, tol=1e-3, r2_convergence=True,
    ):
        n_samples, M = X.shape
        N = Y.shape[1]
        X_sel = X[:, b_mask]
        n_nonzero_tuning = X_sel.shape[1]

        if y.ndim > 1:
            y = y.ravel()

        # storage arrays
        intercepts = np.zeros((max_iter + 1))
        a_hats = np.zeros((max_iter + 1, N))
        b_hats = np.zeros((max_iter + 1, M))
        shared_noise_y = np.zeros((max_iter, n_samples))
        r2_scores = np.zeros(max_iter + 1)

        # first stage
        # make predictions and extract residuals (noise)
        Y_hat = B_hat_intercept + np.dot(X, B_hat)
        Y_noise = Y - Y_hat

        # cross-validate the first stage factor analysis
        if K is None:
            kf = KFold(n_splits=n_fa_splits, shuffle=True)
            scores = np.zeros((n_fa_splits, n_latent_max))
            # iterate over folds and latent factors
            for split_idx, (train_idx, test_idx), in enumerate(kf.split(Y_noise)):
                for latent_idx, n_latent in enumerate(range(1, n_latent_max + 1)):
                    # extract train and test data
                    Y_noise_train = Y_noise[train_idx]
                    Y_noise_test = Y_noise[test_idx]
                    # fit factor analysis
                    fa = FactorAnalysis(
                        n_components=n_latent,
                        tol=1e-6,
                        max_iter=2000)
                    fa.fit(Y_noise_train)
                    # score on test set
                    scores[split_idx, latent_idx] = fa.score(Y_noise_test)

            n_latent = np.argmax(np.mean(scores, axis=0)) + 1
        else:
            n_latent = K

        # first stage factor analysis
        fa1 = FactorAnalysis(
            n_components=n_latent,
            tol=1e-4,
            max_iter=2000)
        fa1.fit(Y_noise)

        # subtract out projection of shared variability into Yj space
        Y_prime = Y - np.dot(fa1.transform(Y_noise), fa1.components_)
        Y_prime_sel = Y_prime[:, a_mask]

        # first stage ols
        ols = LinearRegression()
        Z = np.concatenate((X_sel, Y_prime_sel), axis=1)
        ols.fit(Z, y.ravel())
        # extract fits
        b_hats[0, b_mask], a_hats[0, a_mask] = np.split(ols.coef_, [n_nonzero_tuning])
        intercepts[0] = ols.intercept_

        # obtain estimates and score
        y_hat = intercepts[0] + \
            np.dot(Y_prime_sel, a_hats[0, a_mask]) + np.dot(X_sel, b_hats[0, b_mask])
        r2_scores[0] = r2_score(y.ravel(), y_hat)

        # second stage
        for idx in range(max_iter):
            # obtain target neuron residuals
            y_noise = y - y_hat
            # gather all residuals
            total_noise = np.insert(Y_noise, 0, y_noise, axis=1)

            # extract shared variability from all residuals, using the number
            # of latent dimensions obtained in the first stage
            fa2 = FactorAnalysis(
                n_components=n_latent,
                tol=1e-4,
                max_iter=2000)
            fa2.fit(total_noise)
            shared_variability = fa2.transform(total_noise) @ fa2.components_
            # store noise
            shared_noise_y[idx] = shared_variability[:, 0].reshape(n_samples)

            # obtain modified dataset
            y_prime = y - shared_variability[:, 0]

            # perform second stage OLS
            Z = np.concatenate((X_sel, Y_prime_sel), axis=1)
            ols = LinearRegression()
            ols.fit(Z, y_prime)

            # extract estimates
            b_hats[idx + 1, b_mask], a_hats[idx + 1, a_mask] \
                = np.split(ols.coef_, [n_nonzero_tuning])
            intercepts[idx + 1] = ols.intercept_

            # obtain estimates and score
            y_hat = intercepts[idx + 1] +\
                np.dot(Y_prime_sel, a_hats[idx + 1, a_mask]) + \
                np.dot(X_sel, b_hats[idx + 1, b_mask])
            r2_scores[idx + 1] = r2_score(y, y_hat)

            # check for convergence
            if np.count_nonzero(a_mask) > 0:
                delta_a = np.mean(np.abs(
                    a_hats[idx + 1, a_mask] - a_hats[idx, a_mask]
                ))
            else:
                delta_a = 0

            delta_b = np.mean(np.abs(
                b_hats[idx + 1, b_mask] -
                b_hats[idx, b_mask]
            ))

            # explained variance has converged
            if r2_convergence:
                if np.abs(r2_scores[idx + 1] - r2_scores[idx]) < 1e-7:
                    break

            # parameters have converged
            if delta_b < tol and delta_a < tol:
                break
        n_iters = idx + 1

        return a_hats[n_iters], b_hats[n_iters], intercepts[n_iters]
