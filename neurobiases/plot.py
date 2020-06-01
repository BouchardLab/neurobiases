import matplotlib.pyplot as plt
import numpy as np


def check_fax(fax=None, n_rows=1, n_cols=1, figsize=(10, 10)):
    """Checks an incoming set of axes, and creates new ones if needed.

    Parameters
    ----------
    fax : tuple of mpl.figure and mpl.axes, or None
        The figure and axes. If None, a new set will be created.
    figsize : tuple or None
        The figure size, if fax is None.

    Returns
    -------
    fig, ax : mpl.figure and mpl.axes
        The matplotlib axes objects.
    """
    # no axes provided
    if fax is None:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    else:
        fig, ax = fax
    return fig, ax


def get_cmap_color(cmap, val, min_val=0, max_val=1):
    """Gets the RGBA values for a color in a colormap according to a value in
    an ordered list.

    Parameters
    ----------
    cmap : string
        The colormap.
    val : float
        The value in the list.
    min_val : float
        The minimum value of the list.
    max_val : float
        The maximum value of the list.

    Returns
    -------
    color : tuple
        A tuple indicating the RGBA values of the color.
    """
    idx = int(255 * (val - min_val) / max_val)
    return plt.get_cmap(cmap)(idx)


def tighten_scatter_plot(ax, bounds, line_kwargs=None):
    """Takes an axis and makes the x and y limits equal, while plotting an
    identity line.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to tighten.
    bounds : tuple or array-like
        The bounds of the x and y axes.
    line_color : string
        The color of the identity line.

    Returns
    -------
    ax : matplotlib axis
        Tightened scatterplot.
    """
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    if line_kwargs is not None:
        ax.plot(bounds, bounds,
                color=line_kwargs.get('color', 'gray'),
                linewidth=line_kwargs.get('linewidth', 3),
                linestyle=line_kwargs.get('linestyle', '-'),
                zorder=line_kwargs.get('zorder', -1))
    ax.set_aspect('equal')
    return ax


def plot_tc_fits(
    a_hat, a_true, b_hat, b_true, fax=None, color='black', edgecolor='white'
):
    """Scatters estimated tuning and coupling fits against ground truth fits.

    Parameters
    ----------
    a_hat, a_true : np.ndarray, shape (N,)
        The estimated and true coupling parameters.
    b_hat, b_true : np.ndarray, shape (M,)
        The estimated and true tuning parameters.
    fig, ax : mpl.figure and mpl.axes
        The matplotlib axes objects.

    Returns
    -------
    fig, axes : mpl.figure and mpl.axes
        The matplotlib axes objects.
    """
    fig, axes = check_fax(fax, n_rows=1, n_cols=2, figsize=(10, 5))
    # plot coupling parameters
    max_a = np.max([np.abs(a_hat), np.abs(a_true)])
    axes[0].scatter(a_true, a_hat,
                    color=color,
                    edgecolor=edgecolor,
                    s=75)
    tighten_scatter_plot(axes[0], bounds=[-1.1 * max_a, 1.1 * max_a],
                         line_kwargs={'color': 'gray', 'linewidth': 1.5})
    # plot tuning parameters
    max_b = np.max([b_hat, b_true])
    min_b = min(0, np.min([b_hat, b_true]))
    axes[1].scatter(b_true, b_hat,
                    color=color,
                    edgecolor=edgecolor,
                    s=75)
    tighten_scatter_plot(axes[1], bounds=[min_b - 0.1 * np.abs(min_b), 1.1 * max_b],
                         line_kwargs={'color': 'gray', 'linewidth': 1.5})
    return fig, axes


def plot_tm_fits(
    a_hat, a_true, b_hat, b_true, B_hat, B_true, L_hat, L_true, Psi_hat, Psi_true,
    fax=None, color='black', edgecolor='white'
):
    """Scatters estimated tuning and coupling fits against ground truth fits.

    Parameters
    ----------
    a_hat, a_true : np.ndarray, shape (N,)
        The estimated and true coupling parameters.
    b_hat, b_true : np.ndarray, shape (M,)
        The estimated and true tuning parameters.
    fig, ax : mpl.figure and mpl.axes
        The matplotlib axes objects.

    Returns
    -------
    fig, axes : mpl.figure and mpl.axes
        The matplotlib axes objects.
    """
    fig, axes = check_fax(fax, n_rows=2, n_cols=3, figsize=(12, 8))

    # plot coupling parameters
    axes[0, 0].scatter(a_true, a_hat,
                       color=color,
                       edgecolor=edgecolor,
                       s=50)
    max_a = np.max([np.abs(a_hat), np.abs(a_true)])
    tighten_scatter_plot(axes[0, 0], bounds=[-1.1 * max_a, 1.1 * max_a],
                         line_kwargs={'color': 'gray', 'linewidth': 1.5})

    # plot tuning parameters
    axes[0, 1].scatter(b_true, b_hat,
                       color=color,
                       edgecolor=edgecolor,
                       s=50)
    max_b = np.max([b_hat, b_true])
    min_b = min(0, np.min([b_hat, b_true]))
    bounds_b = [min_b - 0.1 * np.abs(min_b), 1.1 * max_b]
    tighten_scatter_plot(axes[0, 1], bounds=bounds_b,
                         line_kwargs={'color': 'gray', 'linewidth': 1.5})

    # plot non-target tuning parameters
    axes[0, 2].scatter(B_true.ravel(), B_hat.ravel(),
                       color=color,
                       edgecolor=edgecolor,
                       s=50)
    max_B = np.max([B_hat, B_true])
    min_B = min(0, np.min([B_hat, B_true]))
    bounds_B = [min_B - 0.1 * np.abs(min_B), 1.1 * max_B]
    tighten_scatter_plot(axes[0, 2], bounds=bounds_B,
                         line_kwargs={'color': 'gray', 'linewidth': 1.5})

    # plot (log)-private variances
    log_Psi_true = np.log(Psi_true)
    log_Psi_hat = np.log(Psi_hat)
    bounds_Psi = [0.90 * np.min([log_Psi_hat, log_Psi_true]),
                  1.10 * np.max([log_Psi_hat, log_Psi_true])]
    axes[1, 0].scatter(log_Psi_true, log_Psi_hat,
                       color=color,
                       edgecolor=edgecolor,
                       s=50)
    tighten_scatter_plot(axes[1, 0], bounds=bounds_Psi,
                         line_kwargs={'color': 'gray', 'linewidth': 1.5})

    # plot shared co-variances
    tri_idxs = np.triu_indices(n=a_true.size + 1, k=0)
    LL_true = (L_true.T @ L_true)[tri_idxs]
    LL_hat = (L_hat.T @ L_hat)[tri_idxs]
    max_LL = np.max([LL_hat, LL_true])
    min_LL = min(0, np.min([LL_hat, LL_true]))
    bounds_LL = [min_LL - 0.1 * np.abs(min_LL), 1.1 * max_LL]
    axes[1, 1].scatter(LL_true, LL_hat,
                       color=color,
                       edgecolor=edgecolor,
                       s=50)
    tighten_scatter_plot(axes[1, 1], bounds=bounds_LL,
                         line_kwargs={'color': 'gray', 'linewidth': 1.5})

    for ax in axes.ravel():
        ax.set_xlabel(r'\textbf{True}', fontsize=18)
        ax.set_ylabel(r'\textbf{Estimated}', fontsize=18)

    axes[0, 0].set_title(r'\textbf{Tuning Parameters}', fontsize=20)
    axes[0, 1].set_title(r'\textbf{Coupling Parameters}', fontsize=20)
    axes[0, 2].set_title(r'\textbf{Non-target Tuning}', fontsize=20)
    axes[1, 0].set_title(r'\textbf{Private Variances}', fontsize=20)
    axes[1, 1].set_title(r'\textbf{Shared (Co)-variances}', fontsize=20)
    axes[1, 2].axis('off')
    return fig, axes
