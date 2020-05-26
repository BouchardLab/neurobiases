from neurobiases import TriangularModel


def generate_bf_cluster_model(N=10, M=12, K=3):
    """Generates a triangular model using basis functions and clustered noise
    correlations."""
    tuning_kwargs, coupling_kwargs, noise_kwargs, stim_kwargs = \
        TriangularModel.generate_kwargs(
            parameter_design='basis_functions', M=M, tuning_sparsity=0.4,
            tuning_noise_scale=0.20, tuning_bf_scale=1./100, N=N, K=K
        )
    tm = TriangularModel(
        model='linear',
        parameter_design='basis_functions',
        tuning_kwargs=tuning_kwargs,
        coupling_kwargs=coupling_kwargs,
        noise_kwargs=noise_kwargs)
    return tm


def generate_dr_cluster_model(N=6, M=12, K=3):
    """Generates a triangular model using direct response and clustered noise
    correlations."""
    tuning_kwargs, coupling_kwargs, noise_kwargs, stim_kwargs = \
        TriangularModel.generate_kwargs(
            parameter_design='direct_response', M=M, tuning_sparsity=0.4,
            tuning_noise_scale=0.20, N=N, K=K
        )
    tm = TriangularModel(
        model='linear',
        parameter_design='direct_response',
        tuning_kwargs=tuning_kwargs,
        coupling_kwargs=coupling_kwargs,
        stim_kwargs=stim_kwargs,
        noise_kwargs=noise_kwargs)
    return tm
