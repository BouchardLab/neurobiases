import matplotlib.pyplot as plt
import numpy as np


def check_fax(fax=None, figsize=None):
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
        # no figure size provided: use default
        if figsize is None:
            figsize = (10, 10)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = fax
    return fig, ax


def get_cmap_color(cmap, val, min_val=0, max_val=1):
    idx = int(255 * (val - min_val) / max_val)
    return plt.get_cmap(cmap)(idx)
