"""
LBFGS and OWL-QN optimization algorithms

Python wrapper around liblbfgs.
"""

import warnings
from ._lowlevel import LBFGS

import numpy as np
from numpy import array, asarray, float64, zeros
from scipy.optimize.lbfgsb import LbfgsInvHessProduct
from scipy.optimize import _lbfgsb
from scipy.optimize.optimize import (OptimizeResult,
                                     _check_unknown_options, _prepare_scalar_function)
from scipy.optimize._constraints import old_bound_to_new


def fmin_lbfgs(f, x0, progress=None, args=(), orthantwise_c=0,
               orthantwise_start=0, orthantwise_end=-1, m=10,
               epsilon=1e-5, past=1, delta=2.220446049250313e-09, max_iterations=0,
               line_search="default", max_linesearch=20, min_step=1e-20,
               max_step=1e+20, ftol=1e-4, wolfe=0.9, gtol=0.9, xtol=1e-30):
    """Minimize a function using LBFGS or OWL-QN

     Parameters
    ----------
    f : callable(x, g, *args)
        Computes function to minimize and its gradient.
        Called with the current position x (a numpy.ndarray), a gradient
        vector g (a numpy.ndarray) to be filled in and *args.
        Must return the value at x and set the gradient vector g.

    x0 : array-like
        Initial values. A copy of this array is made prior to optimization.

    progress : callable(x, g, fx, xnorm, gnorm, step, k, num_eval, *args),
               optional
        If not None, called at each iteration after the call to f with the
        current values of x, g and f(x), the L2 norms of x and g, the line
        search step, the iteration number, the number of evaluations at
        this iteration and args (see below).
        If the return value from this callable is not 0 and not None,
        optimization is stopped and LBFGSError is raised.

    args : sequence
        Arbitrary list of arguments, passed on to f and progress as *args.

    orthantwise_c: float, optional (default=0)
        Coefficient for the L1 norm of variables.
        This parameter should be set to zero for standard minimization
        problems. Setting this parameter to a positive value activates
        Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
        minimizes the objective function F(x) combined with the L1 norm |x|
        of the variables, {F(x) + C |x|}. This parameter is the coefficient
        for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
        zero, the library modifies function and gradient evaluations from
        a client program suitably; a client program thus have only to return
        the function value F(x) and gradients G(x) as usual. The default value
        is zero.

        If orthantwise_c is set, then line_search cannot be the default
        and must be one of 'armijo', 'wolfe', or 'strongwolfe'.

    orthantwise_start: int, optional (default=0)
        Start index for computing L1 norm of the variables.
        This parameter is valid only for OWL-QN method
        (i.e., orthantwise_c != 0). This parameter b (0 <= b < N)
        specifies the index number from which the library computes the
        L1 norm of the variables x,
            |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
        In other words, variables x_1, ..., x_{b-1} are not used for
        computing the L1 norm. Setting b (0 < b < N), one can protect
        variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
        regression) from being regularized. The default value is zero.

    orthantwise_end: int, optional (default=-1)
        End index for computing L1 norm of the variables.
        This parameter is valid only for OWL-QN method
        (i.e., orthantwise_c != 0). This parameter e (0 < e <= N)
        specifies the index number at which the library stops computing the
        L1 norm of the variables x,

    m: int, optional, default=6
        The number of corrections to approximate the inverse hessian matrix.
        The L-BFGS routine stores the computation results of previous `m`
        iterations to approximate the inverse hessian matrix of the current
        iteration. This parameter controls the size of the limited memories
        (corrections). The default value is 6. Values less than 3 are
        not recommended. Large values will result in excessive computing time.

    epsilon: float, optional (default=1e-5)
        Epsilon for convergence test.
        This parameter determines the accuracy with which the solution is to
        be found. A minimization terminates when
            ||g|| < \ref epsilon * max(1, ||x||),
        where ||.|| denotes the Euclidean (L2) norm. The default value is
        1e-5.

    past: int, optional (default=0)
        Distance for delta-based convergence test.
        This parameter determines the distance, in iterations, to compute
        the rate of decrease of the objective function. If the value of this
        parameter is zero, the library does not perform the delta-based
        convergence test. The default value is 0.

    delta: float, optional (default=0.)
        Delta for convergence test.
        This parameter determines the minimum rate of decrease of the
        objective function. The library stops iterations when the
        following condition is met:
            (f' - f) / max(|f|, |f'|, 1) <= delta,
        where f' is the objective value of `past` iterations ago, and f is
        the objective value of the current iteration.
        The default value is 0.

    max_iterations: int, optional (default=0)
        The maximum number of iterations. Setting this parameter to zero
        continues an optimization process until a convergence or error. The
        default value is 0.

    line_search: str, optional (default="default")
        The line search algorithm.
        This parameter specifies a line search algorithm to be used by the
        L-BFGS routine. Possible values are:

        - 'default': same as 'morethuente'
        - 'morethuente': Method proposed by More and Thuente
        - 'armijo': backtracking with Armijo's conditions
        - 'wolfe': backtracking with Wolfe's conditions
        - 'strongwolfe': backtracking with strong Wolfe's conditions

    max_linesearch: int, optional (default=20)
        The maximum number of trials for the line search.
        This parameter controls the number of function and gradients evaluations
        per iteration for the line search routine. The default value is 20.

    min_step: float, optional (default=1e-20)
        The minimum step of the line search routine.
        The default value is 1e-20. This value need not be modified unless
        the exponents are too large for the machine being used, or unless the
        problem is extremely badly scaled (in which case the exponents should
        be increased).

    max_step: float, optional (default=1e20)
        The maximum step of the line search.
        The default value is 1e+20. This value need not be modified unless
        the exponents are too large for the machine being used, or unless the
        problem is extremely badly scaled (in which case the exponents should
        be increased).

    ftol: float, optional (default=1e-4)
        A parameter to control the accuracy of the line search routine.
        The default value is 1e-4. This parameter should be greater
        than zero and smaller than 0.5.

    wolfe: float, optional (default=0.9)
        A coefficient for the Wolfe condition. This parameter is valid only
        when the backtracking line-search algorithm is used with the Wolfe
        condition (`line_search='wolfe'` or `line_search='strongwolfe'`),
        The default value is 0.9. This parameter should be greater
        the `ftol` parameter and smaller than 1.0.

    gtol: float, optional (default=0.9)
        A parameter to control the accuracy of the line search routine.
        The default value is 0.9. If the function and gradient
        evaluations are inexpensive with respect to the cost of the
        iteration (which is sometimes the case when solving very large
        problems) it may be advantageous to set this parameter to a small
        value. A typical small value is 0.1. This parameter should be
        greater than the ftol parameter (1e-4) and smaller than
        1.0.


    xtol: float, optional (default=1e-30)
        The machine precision for floating-point values.
        This parameter must be a positive value set by a client program to
        estimate the machine precision. The line search routine will terminate
        with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
        of the interval of uncertainty is less than this parameter.


    """

    # Input validation to make sure defaults with OWL-QN are adapted correctly
    assert orthantwise_c >= 0, "Orthantwise_c cannot be negative"

    if orthantwise_c > 0 and line_search not in ['wolfe', 'default']:
        line_search = 'wolfe'
        warnings.warn("When using OWL-QN, 'wolfe' is the only valid "
                      + "line_search. line_search has been set to 'wolfe'.")
    elif orthantwise_c > 0 and line_search == 'default':
        line_search = 'wolfe'

    opt = LBFGS()
    opt.orthantwise_c = orthantwise_c
    opt.orthantwise_start = orthantwise_start
    opt.orthantwise_end = orthantwise_end
    opt.m = m
    opt.epsilon = epsilon
    opt.past = past
    opt.delta = delta
    opt.max_iterations = max_iterations
    opt.linesearch = line_search
    opt.max_linesearch = max_linesearch
    opt.min_step = min_step
    opt.max_step = max_step
    opt.ftol = ftol
    opt.wolfe = wolfe
    opt.gtol = gtol
    opt.xtol = xtol

    return opt.minimize(f, x0, progress=progress, args=args)


def _minimize_lbfgsb(fun, x0, args=(), jac=None, bounds=None,
                     disp=None, maxcor=10, ftol=2.2204460492503131e-09,
                     gtol=1e-5, eps=1e-8, maxfun=15000, maxiter=15000,
                     iprint=-1, callback=None, maxls=20,
                     finite_diff_rel_step=None, **unknown_options):
    """
    Minimize a scalar function of one or more variables using the L-BFGS-B
    algorithm.
    Options
    -------
    disp : None or int
        If `disp is None` (the default), then the supplied version of `iprint`
        is used. If `disp is not None`, then it overrides the supplied version
        of `iprint` with the behaviour you outlined.
    maxcor : int
        The maximum number of variable metric corrections used to
        define the limited memory matrix. (The limited memory BFGS
        method does not store the full hessian but uses this many terms
        in an approximation to it.)
    ftol : float
        The iteration stops when ``(f^k -
        f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``pg_i`` is the i-th component of the
        projected gradient.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    maxfun : int
        Maximum number of function evaluations.
    maxiter : int
        Maximum number of iterations.
    iprint : int, optional
        Controls the frequency of output. ``iprint < 0`` means no output;
        ``iprint = 0``    print only one line at the last iteration;
        ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
        ``iprint = 99``   print details of every iteration except n-vectors;
        ``iprint = 100``  print also the changes of active set and final x;
        ``iprint > 100``  print details of every iteration including x and g.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    Notes
    -----
    The option `ftol` is exposed via the `scipy.optimize.minimize` interface,
    but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The
    relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.
    I.e., `factr` multiplies the default machine floating-point precision to
    arrive at `ftol`.
    """
    _check_unknown_options(unknown_options)
    m = maxcor
    pgtol = gtol
    factr = ftol / np.finfo(float).eps

    x0 = asarray(x0).ravel()
    n, = x0.shape

    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')

    # unbounded variables must use None, not +-inf, for optimizer to work properly
    bounds = [(None if lo == -np.inf else lo, None if u == np.inf else u) for lo, u in bounds]
    # LBFGSB is sent 'old-style' bounds, 'new-style' bounds are required by
    # approx_derivative and ScalarFunction
    new_bounds = old_bound_to_new(bounds)

    # check bounds
    if (new_bounds[0] > new_bounds[1]).any():
        raise ValueError("LBFGSB - one of the lower bounds is greater than an upper bound.")

    # initial vector must lie within the bounds. Otherwise ScalarFunction and
    # approx_derivative will cause problems
    x0 = np.clip(x0, new_bounds[0], new_bounds[1])

    if disp is not None:
        if disp == 0:
            iprint = -1
        else:
            iprint = disp

    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  bounds=new_bounds,
                                  finite_diff_rel_step=finite_diff_rel_step)

    func_and_grad = sf.fun_and_grad

    fortran_int = _lbfgsb.types.intvar.dtype

    nbd = zeros(n, fortran_int)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)
    bounds_map = {(None, None): 0,
                  (1, None): 1,
                  (1, 1): 2,
                  (None, 1): 3}
    for i in range(0, n):
        lo, u = bounds[i]
        if lo is not None:
            low_bnd[i] = lo
            lo = 1
        if u is not None:
            upper_bnd[i] = u
            u = 1
        nbd[i] = bounds_map[lo, u]

    if not maxls > 0:
        raise ValueError('maxls must be positive.')

    x = array(x0, float64)
    f = array(0.0, float64)
    g = zeros((n,), float64)
    wa = zeros(2*m*n + 5*n + 11*m*m + 8*m, float64)
    iwa = zeros(3*n, fortran_int)
    task = zeros(1, 'S60')
    csave = zeros(1, 'S60')
    lsave = zeros(4, fortran_int)
    isave = zeros(44, fortran_int)
    dsave = zeros(29, float64)

    task[:] = 'START'

    n_iterations = 0

    def make_result(f, g, x, task, wa, sf, maxfun, n_iterations, maxiter, isave, maxcor):
        task_str = task.tobytes().strip(b'\x00').strip()
        if task_str.startswith(b'CONV'):
            warnflag = 0
        elif sf.nfev > maxfun or n_iterations >= maxiter:
            warnflag = 1
        else:
            warnflag = 2

        # These two portions of the workspace are described in the mainlb
        # subroutine in lbfgsb.f. See line 363.
        s = wa[0: m*n].reshape(m, n)
        y = wa[m*n: 2*m*n].reshape(m, n)

        # See lbfgsb.f line 160 for this portion of the workspace.
        # isave(31) = the total number of BFGS updates prior the current iteration;
        n_bfgs_updates = isave[30]

        n_corrs = min(n_bfgs_updates, maxcor)
        hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])

        task_str = task_str.decode()
        return OptimizeResult(fun=f, jac=g, nfev=sf.nfev,
                              njev=sf.ngev,
                              nit=n_iterations, status=warnflag, message=task_str,
                              x=x, success=(warnflag == 0), hess_inv=hess_inv)

    results = []
    while 1:
        # x, f, g, wa, iwa, task, csave, lsave, isave, dsave = \
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr,
                       pgtol, wa, iwa, task, iprint, csave, lsave,
                       isave, dsave, maxls)
        task_str = task.tobytes()
        if task_str.startswith(b'FG'):
            # The minimization routine wants f and g at the current x.
            # Note that interruptions due to maxfun are postponed
            # until the completion of the current minimization iteration.
            # Overwrite f and g:
            f, g = func_and_grad(x)
        elif task_str.startswith(b'NEW_X'):
            # new iteration
            n_iterations += 1
            if callback is not None:
                callback(np.copy(x))

            if n_iterations >= maxiter:
                task[:] = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            elif sf.nfev > maxfun:
                task[:] = ('STOP: TOTAL NO. of f AND g EVALUATIONS '
                           'EXCEEDS LIMIT')
            results.append(make_result(f, g, x, task, wa, sf, maxfun, n_iterations, maxiter, isave,
                                       maxcor))
        else:
            break

    return results
