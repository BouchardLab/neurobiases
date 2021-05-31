import h5py
import numpy as np

from .TriangularModel import TriangularModel


def generate_data_from_file(file_path, return_object=False):
    """Generates triangular model data from settings stored in an H5 file.

    Parameters
    ----------
    file_path : string
        Path to the triangular model settings.
    return_object : bool
        If True, returns the Triangular model object in addition to the data.

    Returns
    -------
    X, Y, y : np.ndarrays
        The tuning, coupling, and target data, respectively.
    tm : TriangularModel, optional
        The triangular model object that generated the data, if requested.
    """
    # Open up experiment settings
    with h5py.File(file_path, 'r') as params:
        # Triangular model hyperparameters
        N = params.attrs['N']
        M = params.attrs['M']
        K = params.attrs['K']
        D = params.attrs['D']
        corr_cluster = params.attrs['corr_cluster']
        corr_back = params.attrs['corr_back']
        coupling_distribution = params.attrs['coupling_distribution']
        coupling_sparsity = params.attrs['coupling_sparsity']
        coupling_loc = params.attrs['coupling_loc']
        coupling_scale = params.attrs['coupling_scale']
        coupling_rng = params.attrs['coupling_rng']
        tuning_distribution = params.attrs['tuning_distribution']
        tuning_sparsity = params.attrs['tuning_sparsity']
        tuning_loc = params.attrs['tuning_loc']
        tuning_scale = params.attrs['tuning_scale']
        tuning_rng = params.attrs['tuning_rng']
        # Random seeds
        coupling_rng = params.attrs['coupling_rng']
        tuning_rng = params.attrs['tuning_rng']
        dataset_rng = params.attrs['dataset_rng']

    # Generate triangular model
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        M=M,
        N=N,
        K=K,
        corr_cluster=corr_cluster,
        corr_back=corr_back,
        coupling_distribution=coupling_distribution,
        coupling_sparsity=coupling_sparsity,
        coupling_loc=coupling_loc,
        coupling_scale=coupling_scale,
        coupling_rng=coupling_rng,
        tuning_distribution=tuning_distribution,
        tuning_sparsity=tuning_sparsity,
        tuning_loc=tuning_loc,
        tuning_scale=tuning_scale,
        tuning_rng=tuning_rng,
        stim_distribution='uniform')
    # Generate data using seed
    X, Y, y = tm.generate_samples(n_samples=D, rng=int(dataset_rng))

    if return_object:
        return X, Y, y, tm
    else:
        return X, Y, y


def generate_means_exp(
    tag, params_folder, N=10, M=10, K=1, D=2000, n_models=3, n_datasets=10,
    n_coupling_locs=5, coupling_loc_min=-1., coupling_loc_max=1.,
    n_tuning_locs=5, tuning_loc_min=-1., tuning_loc_max=1., coupling_rng=None,
    tuning_rng=None, dataset_rng=None, fitter_rng=None,
    coupling_distribution='gaussian', coupling_sparsity=0.5, coupling_scale=0.25,
    tuning_distribution='gaussian', tuning_sparsity=0.5, tuning_scale=0.25,
    corr_cluster=0.25, corr_back=0.10
):
    """Generates triangular model data from settings stored in an H5 file.

    Parameters
    ----------
    N : int
        The number of coupling parameters.
    M : int
        The number of tuning parameters.
    K : int
        The number of latent factors.
    D : int
        The number of samples.
    n_models : int
        The number of models per hyperparameter.
    n_datasets : int
        The number of datasets per model.
    n_coupling_locs, n_tuning_locs : int
        The number of coupling / tuning hyperparameters.
    coupling_loc_min, tuning_loc_min : float
        The minimum hyperparameter for the coupling / tuning means.
    coupling_loc_max, tuning_loc_max : float
        The maximum hyperparameter for the coupling / tuning means.
    coupling_rng, tuning_rng, dataset_rng, fitter_rng : {None,
                                                         int,
                                                         array_like[ints],
                                                         SeedSequence,
                                                         BitGenerator,
                                                         Generator}
        The random number generator for coupling parameter seeds, tuning
        parameter seeds, dataset seeds, and fitter.
    coupling_distribution : string
        The distribution from which to draw the coupling parameters.
    coupling_sparsity : float
        The fraction of coupling parameters that are set to zero.
    coupling_scale : float
        Specifies the scale of the distribution from which the coupling
        parameters are drawn.
    tuning_distribution : string
        The distribution from which to draw the tuning parameters.
    tuning_sparsity : float
        The fraction of tuning parameters that are set to zero.
    tuning_scale : float
        Specifies the scale of the distribution from which the tuning
        parameters are drawn.
    corr_cluster : float
        The correlation between neurons clusters. Relevant for 'cluster' noise
        structure.
    corr_back : float
        The correlation between neurons not within cluster. Relevant for
        'cluster' noise structure.

    Returns
    -------
    X, Y, y : np.ndarrays
        The tuning, coupling, and target data, respectively.
    tm : TriangularModel, optional
        The triangular model object that generated the data, if requested.
    """
    # Model hyperparameters
    coupling_locs = np.linspace(coupling_loc_min,
                                coupling_loc_max,
                                n_coupling_locs)
    tuning_locs = np.linspace(tuning_loc_min,
                              tuning_loc_max,
                              n_tuning_locs)

    # Process coupling random states
    if coupling_rng is None:
        coupling_rng = np.random.default_rng()
    else:
        coupling_rng = np.random.default_rng(coupling_rng)
    coupling_rngs = coupling_rng.integers(
        low=0,
        high=2**32-1,
        size=n_models)
    # Process tuning random states
    if tuning_rng is None:
        tuning_rng = np.random.default_rng()
    else:
        tuning_rng = np.random.default_rng(tuning_rng)
    tuning_rngs = tuning_rng.integers(
        low=0,
        high=2**32-1,
        size=n_models)
    # Process dataset random states
    if dataset_rng is None:
        dataset_rng = np.random.default_rng()
    else:
        dataset_rng = np.random.default_rng(dataset_rng)
    dataset_rngs = dataset_rng.integers(
        low=0,
        high=2**32 - 1,
        size=n_datasets)
    # Process fitter random state
    if fitter_rng is None:
        fitter_rng = np.random.randint(low=0, high=2**32 - 1)
    else:
        fitter_rng = fitter_rng

    for ii, coupling_loc in enumerate(coupling_locs):
        for jj, tuning_loc in enumerate(tuning_locs):
            for kk, (coupling_rng, tuning_rng) in enumerate(zip(coupling_rngs, tuning_rngs)):
                for ll, dataset_rng in enumerate(dataset_rngs):
                    # Generate triangular model
                    tm = TriangularModel(
                        model='linear',
                        parameter_design='direct_response',
                        M=M,
                        N=N,
                        K=K,
                        corr_cluster=corr_cluster,
                        corr_back=corr_back,
                        coupling_distribution=coupling_distribution,
                        coupling_sparsity=coupling_sparsity,
                        coupling_loc=coupling_loc,
                        coupling_scale=coupling_scale,
                        coupling_rng=coupling_rng,
                        tuning_distribution=tuning_distribution,
                        tuning_sparsity=tuning_sparsity,
                        tuning_loc=tuning_loc,
                        tuning_scale=tuning_scale,
                        tuning_rng=tuning_rng,
                        stim_distribution='uniform')
                    # Save path for current experiment
                    save_path = f"{params_folder}/{tag}_{ii}_{jj}_{kk}_{ll}.h5"
                    # Load the model configuration
                    with h5py.File(save_path, 'w') as params:
                        # Triangular model parameters
                        params['a_true'] = np.copy(tm.a.ravel())
                        params['b_true'] = np.copy(tm.b.ravel())
                        params['B_true'] = np.copy(tm.B)
                        params['Psi_true'] = np.copy(tm.Psi)
                        params['L_true'] = np.copy(tm.L)
                        # Triangular model hyperparameters
                        params.attrs['N'] = N
                        params.attrs['M'] = M
                        params.attrs['K'] = K
                        params.attrs['D'] = D
                        params.attrs['coupling_distribution'] = coupling_distribution
                        params.attrs['coupling_sparsity'] = coupling_sparsity
                        params.attrs['coupling_loc'] = coupling_loc
                        params.attrs['coupling_scale'] = coupling_scale
                        params.attrs['coupling_rng'] = coupling_rng
                        params.attrs['tuning_distribution'] = tuning_distribution
                        params.attrs['tuning_sparsity'] = tuning_sparsity
                        params.attrs['tuning_loc'] = tuning_loc
                        params.attrs['tuning_scale'] = tuning_scale
                        params.attrs['tuning_rng'] = tuning_rng
                        params.attrs['corr_cluster'] = corr_cluster
                        params.attrs['corr_back'] = corr_back
                        # Random seeds
                        params.attrs['coupling_rng'] = coupling_rng
                        params.attrs['tuning_rng'] = tuning_rng
                        params.attrs['dataset_rng'] = dataset_rng
                        params.attrs['fitter_rng'] = fitter_rng
