import numpy as np
import pytest
import torch

from neurobiases import EMSolver, TriangularModel
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis


def test_marginal_likelihood():
    """Tests the marginal likelihood for the linear TM in a simple case."""
    tm1 = TriangularModel(parameter_design='direct_response', N=8, M=9, K=2)
    tm2 = TriangularModel(parameter_design='basis_functions', N=8, M=9, K=2)

    for tm in [tm1, tm2]:
        X, Y, y = tm.generate_samples(n_samples=1000)
        solver = EMSolver(X, Y, y, K=1, initialization='zeros')
        solver.set_params(L=np.zeros_like(solver.L))
        # Test the marginal likelihood
        empirical_likelihood = solver.marginal_log_likelihood()
        Psi = solver.Psi_tr_to_Psi()
        true_likelihood = -0.5 * (
            X.shape[0] * np.log(np.prod(Psi)) +
            (y.T @ y).item() / Psi[0] + np.trace(Y / Psi[1:] @ Y.T)
        )
        assert_allclose(true_likelihood, empirical_likelihood)


def test_f_df_ml():
    """Tests the function generating the loss and gradient for the marginal
    log-likelihood."""
    # Generate triangular model and data
    tm = TriangularModel(parameter_design='direct_response', N=8, M=9, K=2)
    X, Y, y = tm.generate_samples(n_samples=1000)
    # Create EMSolver object
    solver = EMSolver(X, Y, y, K=1, initialization='zeros')
    solver.set_params(L=np.zeros_like(solver.L))

    neg_mll, _ = EMSolver.f_df_ml(
        solver.get_params(), X, Y, y, solver.K,
        a_mask=solver.a_mask, b_mask=solver.b_mask, B_mask=solver.B_mask,
        train_B=solver.train_B, train_L=solver.train_L,
        train_L_nt=solver.train_L_nt, train_Psi_tr_nt=solver.train_Psi_tr_nt,
        train_Psi_tr=solver.train_Psi_tr, Psi_transform=solver.Psi_transform)
    true_neg_mll = -solver.marginal_log_likelihood()
    assert_allclose(true_neg_mll, neg_mll)

    # Check for random initialization
    solver = EMSolver(X, Y, y, K=1, initialization='random')
    neg_mll, _ = EMSolver.f_df_ml(
        solver.get_params(), X, Y, y, solver.K,
        a_mask=solver.a_mask, b_mask=solver.b_mask, B_mask=solver.B_mask,
        train_B=solver.train_B, train_L=solver.train_L,
        train_L_nt=solver.train_L_nt, train_Psi_tr_nt=solver.train_Psi_tr_nt,
        train_Psi_tr=solver.train_Psi_tr, Psi_transform=solver.Psi_transform)
    true_neg_mll = -solver.marginal_log_likelihood()
    assert_allclose(true_neg_mll, neg_mll)


def test_f_mll():
    """Tests the function generating the loss for the marginal
    log-likelihood."""
    # Generate triangular model and data
    tm = TriangularModel(parameter_design='direct_response', N=8, M=9, K=2)
    X, Y, y = tm.generate_samples(n_samples=1000)
    # Create EMSolver object
    solver = EMSolver(X, Y, y, K=1, initialization='zeros')
    solver.set_params(L=np.zeros_like(solver.L))

    neg_mll = EMSolver.f_mll(
        torch.tensor(solver.get_params()), X, Y, y, solver.K,
        Psi_transform=solver.Psi_transform)
    true_neg_mll = -solver.marginal_log_likelihood()
    assert_allclose(true_neg_mll, neg_mll)

    # Check for random initialization
    solver = EMSolver(X, Y, y, K=1, initialization='random')
    neg_mll = EMSolver.f_mll(
        torch.tensor(solver.get_params()), X, Y, y, solver.K,
        Psi_transform=solver.Psi_transform)
    true_neg_mll = -solver.marginal_log_likelihood()
    assert_allclose(true_neg_mll, neg_mll)


def test_bic():
    """Tests the BIC for the linear TM in a simple case."""
    tm1 = TriangularModel(parameter_design='direct_response', N=8, M=9, K=2)
    tm2 = TriangularModel(parameter_design='basis_functions', N=8, M=9, K=2)
    D = 1000

    for tm in [tm1, tm2]:
        X, Y, y = tm.generate_samples(n_samples=D)
        solver = EMSolver(X, Y, y, K=1)
        solver.set_params(L=np.zeros_like(solver.L))
        # test the marginal likelihood
        empirical_bic = solver.bic()
        Psi = solver.Psi_tr_to_Psi()
        true_likelihood = -0.5 * (
            D * np.log(np.prod(Psi)) +
            (y.T @ y).item() / Psi[0] + np.trace(Y / Psi[1:] @ Y.T)
        )
        true_bic = -2 * true_likelihood + Psi.size * np.log(D)
        assert_allclose(true_bic, empirical_bic)


def test_initialization():
    """Tests the initialization of the EM solver."""
    # Generate data
    tm = TriangularModel(parameter_design='direct_response', N=8, M=9, K=1)
    X, Y, y = tm.generate_samples(n_samples=1000)

    # Zero initialization
    solver = EMSolver(X, Y, y, K=1, initialization='zeros')
    # Check that all initializations (except latent factors) are zero
    assert not np.any(solver.a.ravel())
    assert not np.any(solver.b.ravel())
    assert not np.any(solver.B.ravel())
    assert not np.any(solver.Psi_tr.ravel())

    # Run fit using initialization
    solver = EMSolver(X, Y, y, K=1, initialization='fits')
    fitter = LinearRegression(fit_intercept=False)
    assert_allclose(solver.a.ravel(), fitter.fit(Y, y.ravel()).coef_)
    assert_allclose(solver.b.ravel(), fitter.fit(X, y.ravel()).coef_)
    assert_allclose(solver.B, fitter.fit(X, Y).coef_.T)
    # Shared variability: fit by factor analysis model
    Y_res = Y - fitter.predict(X)
    Z = np.concatenate((X, Y), axis=1)
    fitter.fit(Z, y.ravel())
    y_res = y.ravel() - fitter.predict(Z)
    residuals = np.concatenate((y_res[..., np.newaxis], Y_res), axis=1)
    fa = FactorAnalysis(n_components=1)
    fa.fit(residuals)
    # Check private variability using Factor Analysis
    assert_allclose(fa.noise_variance_, solver.Psi_tr_to_Psi())
    LL = fa.components_.T @ fa.components_
    LL_hat = solver.L.T @ solver.L
    # Check latent factors using Factor Analysis
    assert_allclose(LL, LL_hat)


def test_hessian():
    """Tests that the Hessian is calculated correctly."""
    # Generate triangular model and data
    tm = TriangularModel(parameter_design='direct_response', N=8, M=9, K=2)
    X, Y, y = tm.generate_samples(n_samples=1000)
    # Create EMSolver object
    solver = EMSolver(X, Y, y, K=1, initialization='zeros')
    # Calculate Hessian
    hessian = solver.marginal_likelihood_hessian()
    n_params = solver.get_params().size
    assert hessian.shape == (n_params, n_params)


def test_directional_derivative():
    """Tests that the directional derivative in the direction of the
    identifiability subspace is orthogonal to the gradient."""
    K = 3
    # Create triangular model and solver
    tm = TriangularModel(parameter_design='direct_response', N=8, M=9, K=K)
    X, Y, y = tm.generate_samples(n_samples=10000)
    solver = EMSolver(X, Y, y, K=K, initialization='random')
    # Create copy for identifiability transform
    copy = solver.copy()
    copy.identifiability_transform(delta=np.random.randn(K) / 100.)
    # Calculate identifiability subspace direction
    dx_params = copy.get_params(return_Psi=True) - solver.get_params(return_Psi=True)
    dx_params = dx_params / np.linalg.norm(dx_params)
    # Calculate gradient
    _, grad = solver.f_df_ml(
        solver.get_params(), X, Y, y, K,
        a_mask=solver.a_mask,
        b_mask=solver.b_mask,
        B_mask=solver.B_mask,
        train_B=solver.train_B,
        train_L_nt=solver.train_L_nt,
        train_L=solver.train_L,
        train_Psi_tr_nt=solver.train_Psi_tr_nt,
        train_Psi_tr=solver.train_Psi_tr,
        Psi_transform=solver.Psi_transform,
        wrt_Psi=True)
    grad = grad / np.linalg.norm(grad)
    # Check that the dot product is close to zero
    assert_allclose(0, np.dot(grad, dx_params), atol=1e-1)


@pytest.mark.slow
def test_refit():
    """Tests that the refit option works correctly."""
    # Generate triangular model and data
    tm = TriangularModel(parameter_design='direct_response', N=8, M=9, K=1)
    X, Y, y = tm.generate_samples(n_samples=1000)
    solver = EMSolver(X, Y, y, K=1, initialization='zeros',
                      solver='ow_lbfgs',
                      c_tuning=100,
                      c_coupling=0.,
                      max_iter=100)
    # Run fit, making sure to refit
    solver.fit_em(refit=True)
    assert np.count_nonzero(solver.b.ravel()) == 0
    assert np.count_nonzero(solver.a.ravel()) == solver.N


@pytest.mark.slow
def test_em_update():
    """Tests that the EM optimization always increases the marginal likelihood."""
    tm1 = TriangularModel(parameter_design='direct_response', N=8, M=9, K=1)
    tm2 = TriangularModel(parameter_design='basis_functions', N=8, M=9, K=1)

    # Iterate over versions of the triangular model
    for tm in [tm1, tm2]:
        X, Y, y = tm.generate_samples(n_samples=100)
        solver = EMSolver(X=X, Y=Y, y=y, K=1, max_iter=50, tol=0)
        mlls = solver.fit_em(mll_curve=True)
        assert mlls.size == solver.max_iter
        assert np.all(np.ediff1d(mlls) > 0)


@pytest.mark.slow
def test_em_mask():
    """Tests that, when provided a mask, EMSolver maintains the selection
    profiles during fitting."""
    # Generate triangular model and data
    tm = TriangularModel(parameter_design='basis_functions', N=8, M=9, K=2)
    X, Y, y = tm.generate_samples(n_samples=100)
    # Fit solver using masks obtained from triangular model
    a_mask, b_mask, B_mask = tm.get_masks()
    solver = EMSolver(X=X, Y=Y, y=y, K=1, max_iter=10, tol=0,
                      a_mask=a_mask, b_mask=b_mask, B_mask=B_mask)
    solver.fit_em()
    # Check that the selection profiles are conserved
    assert_array_equal(a_mask, solver.a.ravel() != 0)
    assert_array_equal(b_mask, solver.b.ravel() != 0)
    assert_array_equal(B_mask, solver.B != 0)
