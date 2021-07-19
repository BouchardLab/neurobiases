import neurobiases.utils as utils
import neurobiases.plot as plot
import numpy as np
import warnings

from scipy.stats import truncexpon


class TriangularModel:
    """Creates and draws samples from a triangular model.

    Parameters
    ----------
    model : string
        Specifies either a 'linear' or 'poisson' model.
    parameter_design : string
        The structure to impose on the parameters: either 'random',
        'direct_response' or 'basis_functions'.
    N : int
        The number of coupling parameters.
    coupling_distribution : string
        The distribution from which to draw the coupling parameters.
    coupling_sparsity : float
        The fraction of coupling parameters that are set to zero.
    coupling_loc : float
        Specifies the location of the distribution from which the coupling
        parameters are drawn.
    coupling_scale : float
        Specifies the scale of the distribution from which the coupling
        parameters are drawn.
    coupling_rng : {None, int, array_like[ints], SeedSequence, BitGenerator,
                    Generator}
        The coupling random number generator.
    M : int
        The number of tuning parameters.
    tuning_distribution : string
        The distribution from which to draw the tuning parameters.
    tuning_sparsity : float
        The fraction of tuning parameters that are set to zero.
    tuning_noise_scale : float
        Specifies the scale of noise added to the tuning parameters, if
        desired.
    tuning_peak : float
        The maximum possible value of the tuning curve (relevant for Hann
        window).
    tuning_loc : float
        Specifies the location of the distribution from which the tuning
        parameters are drawn.
    tuning_scale : float
        Specifies the scale of the distribution from which the tuning
        parameters are drawn.
    tuning_low : float
        The minimum value of the tuning parameters. Relevant for a uniform
        tuning distribution.
    tuning_high : float
        The maximum value of the tuning parameters. Relevant for a uniform
        tuning distribution.
    tuning_bound_frac : float
        Specifies the fraction of tuning curve that can be truncated off the
        tuning plane.
    tuning_diff_decay : float
        Specifies the coupling probability decays with tuning difference.
    tuning_bf_scale : float
        The scale parameter for the tuning basis functions. Relevant for the
        'basis_functions' parameter design.
    target_pref_tuning : float
        The preferred tuning of the target neuron. Relevant for the
        'basis_functions' parameter design.
    tuning_add_noise : bool
        If True, adds noise to the tuning parameters.
    tuning_overall_scale : float
        Scaling factor for all tuning parameters.
    tuning_rng : {None, int, array_like[ints], SeedSequence, BitGenerator,
                  Generator}
        The tuning random number generator.
    K : int
        The number of latent factors.
    snr : float
        The signal-to-noise ratio.
    noise_structure : string
        The structure of the latent factors.
    corr_cluster : float
        The correlation between neurons clusters. Relevant for 'cluster' noise
        structure.
    corr_back : float
        The correlation between neurons not within cluster. Relevant for
        'cluster' noise structure.
    corr_max : float
        The maximum noise correlation. Relevant for 'falloff' noise structure.
    corr_min : float
        The minimum noise correlation. Relevant for 'falloff' noise structure.
    L_corr : float
        Exponential falloff term for noise correlations.
    stim_distribution : string
        The distribution from which to draw the stimulus values.
    stim_kwargs : dict
        Stimulus keyword arguments. If None, it is automatically populated using
        default values.
    """
    def __init__(
        self, model='linear', parameter_design='direct_response', N=20,
        coupling_distribution='symmetric_lognormal', coupling_sparsity=0.5,
        coupling_loc=-1, coupling_scale=0.5, coupling_rng=2332, M=20,
        tuning_distribution='noisy_hann_window', tuning_sparsity=0.50,
        tuning_noise_scale=None, tuning_peak=150, tuning_loc=0.,
        tuning_scale=1., tuning_low=0, tuning_high=1., tuning_bound_frac=0.25,
        tuning_diff_decay=1., tuning_bf_scale=None, target_pref_tuning=0.5,
        tuning_add_noise=True, tuning_overall_scale=1., tuning_rng=2332,
        K=1, snr=3, noise_structure='clusters', corr_cluster=0.2,
        corr_back=0., corr_max=0.3, corr_min=0.0, L_corr=1,
        stim_distribution='uniform', stim_kwargs=None, warn=True
    ):
        self.model = model
        self.parameter_design = parameter_design

        # Coupling keyword arguments
        self.N = N
        self.coupling_kwargs = {
            'N': N,
            'sparsity': coupling_sparsity,
            'distribution': coupling_distribution,
            'loc': coupling_loc,
            'scale': coupling_scale,
            'rng': np.random.default_rng(coupling_rng),
        }

        # Noise keyword arguments
        if noise_structure == 'clusters':
            self.noise_kwargs = {
                'K': K,
                'noise_structure': noise_structure,
                'snr': snr,
                'corr_back': corr_back,
                'corr_cluster': corr_cluster
            }
        elif noise_structure == 'falloff':
            self.noise_kwargs = {
                'K': K,
                'noise_structure': noise_structure,
                'snr': snr,
                'corr_max': corr_max,
                'corr_min': corr_min,
                'L_corr': L_corr
            }
        else:
            raise ValueError('Noise structure invalid.')

        # Tuning keyword arguments
        self.M = M
        self.tuning_kwargs = {
            'M': M,
            'sparsity': tuning_sparsity,
            'diff_decay': tuning_diff_decay,
            'rng': np.random.default_rng(tuning_rng)
        }
        if tuning_noise_scale is not None:
            self.tuning_kwargs['noise_scale'] = tuning_noise_scale

        if parameter_design == 'direct_response':
            self.tuning_kwargs['distribution'] = tuning_distribution
            if (tuning_distribution == 'hann_window'
                    or tuning_distribution == 'noisy_hann_window'):
                self.tuning_kwargs['peak'] = tuning_peak
            elif tuning_distribution == 'uniform':
                self.tuning_kwargs['low'] = tuning_low
                self.tuning_kwargs['high'] = tuning_high
            else:
                self.tuning_kwargs['loc'] = tuning_loc
                self.tuning_kwargs['scale'] = tuning_scale
            self.tuning_kwargs['bound_frac'] = tuning_bound_frac
        elif parameter_design == 'basis_functions':
            self.tuning_kwargs['target_pref_tuning'] = target_pref_tuning
            self.tuning_kwargs['add_noise'] = tuning_add_noise
            self.tuning_kwargs['overall_scale'] = tuning_overall_scale
            if tuning_bf_scale is not None:
                self.tuning_kwargs['bf_scale'] = tuning_bf_scale
            else:
                self.tuning_kwargs['bf_scale'] = 0.25 / self.M
        else:
            raise ValueError('Parameter design invalid.')

        # Stimulus keyword arguments
        self.K = K
        if stim_kwargs is None:
            if stim_distribution == 'uniform':
                self.stim_kwargs = {
                    'distribution': 'uniform',
                    'low': 0,
                    'high': 1
                }
            elif stim_distribution == 'gaussian':
                self.stim_kwargs = {
                    'distribution': 'gaussian',
                    'loc': 0,
                    'scale': 1
                }
            else:
                self.stim_kwargs = {
                    'distribution': 'uniform',
                    'low': 0,
                    'high': 1
                }
        else:
            self.stim_kwargs['distribution'] = stim_distribution

        # Create parameters according to preferred design
        self.a, self.b, self.B = self.generate_triangular_model()
        self.B_all = np.concatenate((self.B, self.b), axis=1)
        self.generate_noise_structure()
        # Check that the instantiated model is identifiable
        if not self.check_identifiability_conditions():
            if warn:
                warnings.warn(
                    "This model is not identifiable. If you intend to " +
                    "perform inference on this model, you should adjust the " +
                    "hyperparameters (e.g., sparsity) or random seed to " +
                    "guarantee identifiability.",
                    category=RuntimeWarning
                )

    def generate_triangular_model(self):
        """Generate model parameters in the triangular model.

        Returns
        -------
        a : np.ndarray, shape (N, 1).
            The coupling parameters.
        b : np.ndarray, shape (M, 1).
            The target tuning parameters.
        B : np.ndarray, shape (M, N)
            The non-target tuning parameters.
        """
        # Initialize parameter arrays
        a = np.zeros((self.N, 1))
        B_all = np.zeros((self.M, self.N + 1))
        # Calculate number of non-zero parameters
        n_nonzero_tuning = int((1 - self.tuning_kwargs['sparsity']) * self.M)
        n_nonzero_coupling = int((1 - self.coupling_kwargs['sparsity']) * self.N)
        # Get random states
        coupling_rng = self.coupling_kwargs['rng']
        tuning_rng = self.tuning_kwargs['rng']

        # Randomly assign selection profiles
        if self.parameter_design == 'random':
            # Draw the parameter values
            nonzero_a = self.draw_parameters(
                size=n_nonzero_coupling,
                **self.coupling_kwargs)
            nonzero_B_all = self.draw_parameters(
                size=(n_nonzero_tuning, self.N + 1),
                **self.tuning_kwargs)
            # Store the non-zero values
            a[:n_nonzero_coupling, 0] = nonzero_a
            B_all[:n_nonzero_tuning, :] = nonzero_B_all
            # Shuffle the parameters in place
            coupling_rng.shuffle(a)
            [coupling_rng(B_all[:, idx]) for idx in range(self.N + 1)]
            b, B = np.split(B_all, [1], axis=1)

        # Direct response: each parameter directly denotes the neural response
        elif self.parameter_design == 'direct_response':
            b = np.zeros((self.M, 1))
            B = np.zeros((self.M, self.N))

            # Draw the non-zero parameters for the target neuron
            nonzero_b = self.draw_parameters(
                size=n_nonzero_tuning,
                **self.tuning_kwargs)
            # The offset of the target tuning curve
            b_offset_idx = int(0.5 * (self.M - n_nonzero_tuning))
            # Set the non-zero parameters within the tuning window
            b[b_offset_idx:b_offset_idx + n_nonzero_tuning, 0] = nonzero_b

            # Offsets for the non-target neuron
            bound_frac = self.tuning_kwargs['bound_frac']
            lower_bound = -int(bound_frac * n_nonzero_tuning)
            upper_bound = self.M - int((1 - bound_frac) * n_nonzero_tuning)
            offsets = np.linspace(lower_bound, upper_bound, self.N).astype('int')

            # Iterate over neurons/offsets, assigning tuning curves
            for neuron_idx, offset in enumerate(offsets):
                tuning_curve = self.draw_parameters(
                    size=n_nonzero_tuning,
                    **self.tuning_kwargs)

                # Tuning curve ends up on the left side of the tuning plane
                if offset < 0:
                    new_offset = n_nonzero_tuning + offset
                    B[:new_offset, neuron_idx] = tuning_curve[-new_offset:]
                # Tuning curve ends up on the right side of the tuning plane
                elif offset >= self.M - n_nonzero_tuning + 1:
                    B[offset:, neuron_idx] = tuning_curve[:(self.M - offset)]
                # Tuning curve is in the middle of the plane
                else:
                    B[offset:offset + n_nonzero_tuning, neuron_idx] = tuning_curve

            B_all = np.concatenate((b, B), axis=1)
            # Calculate tuning preferences
            self.tuning_prefs = np.insert(np.round(np.linspace(
                int(n_nonzero_tuning / 2.),
                self.M - int(n_nonzero_tuning / 2.),
                self.N)), 0, int(self.M / 2.) - 1)
            a = self.generate_coupling_profile(self.tuning_prefs)

        elif self.parameter_design == 'basis_functions':
            # Get basis function centers and width
            self.bf_centers = np.linspace(0, 1, self.M)
            self.bf_scale = self.tuning_kwargs['bf_scale']

            # Get preferred tunings for each neuron
            target_tuning = self.tuning_kwargs['target_pref_tuning']
            non_target_tuning = np.linspace(0, 1, self.N)
            all_tunings = np.append(target_tuning, non_target_tuning)

            # Calculate differences between preferred tuning and bf locations
            tuning_diffs = np.abs(np.subtract.outer(self.bf_centers, all_tunings))
            # Calculate preferred tuning, with possible scaling
            B_all = 1 - tuning_diffs
            # Apply a sigmoid to flatten coefficients
            for idx, B in enumerate(B_all.T):
                B_all[:, idx] = utils.sigmoid(
                    B, phase=B.mean(), b=5./(B.max() - B.min())
                )
            # Add noise to tuning parameters if desired
            if self.tuning_kwargs['add_noise']:
                noise = tuning_rng.normal(
                    loc=0.,
                    scale=self.tuning_kwargs.get('noise_scale',
                                                 self.bf_scale / 5),
                    size=(self.M, self.N + 1))
                B_all += noise
            B_all = np.abs(B_all) * self.tuning_kwargs['overall_scale']

            # For each neuron, get selection profile
            for idx in range(self.N + 1):
                # Get tuning differences with bfs
                tuning_diff = tuning_diffs[:, idx]
                # Choose zero indices by chance, weighted by tuning diff
                nonzero_indices = np.argsort(tuning_diff)[:n_nonzero_tuning]
                # Set parameters to zero
                zero_indices = np.setdiff1d(np.arange(self.M), nonzero_indices)
                B_all[zero_indices, idx] = 0
            b, B = np.split(B_all, [1], axis=1)
            # Calculate preferred tuning for each neuron
            self.tuning_prefs = utils.calculate_pref_tuning(
                B=B_all, bf_centers=self.bf_centers, bf_scale=self.bf_scale,
                n_stimuli=10000, limits=(0, 1)
            )
            # Generate coupling parameters
            a = self.generate_coupling_profile(self.tuning_prefs)

        else:
            raise ValueError('Parameter design not available.')
        self.coupled_indices = np.argwhere(a.ravel()).ravel()
        self.non_coupled_indices = np.setdiff1d(np.arange(self.N), self.coupled_indices)
        return a, b, B

    def generate_noise_structure(self):
        """Generates the noise covariance structure for the triangular model."""
        noise_structure = self.noise_kwargs['noise_structure']
        snr = self.noise_kwargs['snr']

        # Noise correlations exist in specific clusters, with increasing
        # latent dimension increasing the number of clusters
        if noise_structure == 'clusters':
            corr_cluster = self.noise_kwargs['corr_cluster']
            corr_back = self.noise_kwargs['corr_back']

            # Calculate non-target signal and noise variances
            non_target_signal_variance = self.non_target_signal_variance()
            non_target_noise_variance = non_target_signal_variance / snr
            # Latent factors non-target neurons; one extra factor for now
            L_nt = np.zeros((self.K + 1, self.N))
            # One basis vector will provide the background noise correlation
            L_nt[-1] = np.sqrt(corr_back * non_target_noise_variance)
            # Split up the variances into the K clusters of correlated neurons
            variance_clusters = np.array_split(np.sqrt(non_target_noise_variance), self.K)
            idx_clusters = np.array_split(np.arange(self.N), self.K)
            # Iterate over the K clusters, each of which will set a latent factor
            for lf_idx, (variances, idxs) in enumerate(zip(variance_clusters, idx_clusters)):
                # Place the correct values in the latent factor
                L_nt[lf_idx, idxs] = variances * np.sqrt(corr_cluster - corr_back)
            # Store these for now so that we can calculate target signal var
            self.L_nt = np.copy(L_nt)
            self.Psi_nt = non_target_noise_variance - np.diag(L_nt.T @ L_nt)

            # Calculate latent factors and private variance for target neuron
            target_signal_variance = self.target_signal_variance()
            target_noise_variance = target_signal_variance / snr
            # Place the target neuron in the middle group
            target_group_idx = int(np.ceil(self.K / 2 - 1))
            # Calculate target latent factor
            l_t = np.zeros((self.K + 1, 1))
            l_t[target_group_idx] = \
                np.sqrt(target_noise_variance) * np.sqrt(corr_cluster - corr_back)
            # Add on the background noise correlation term
            l_t[-1] = np.sqrt(corr_back * target_noise_variance)

            # Combine latent factors and private variances
            self.L = np.concatenate((l_t, L_nt), axis=1)
            self.L = utils.symmetric_low_rank_approx(self.L.T @ self.L, self.K).T
            self.l_t, self.L_nt = np.split(self.L, [1], axis=1)
            # Calculate private variances
            total_noise_variance = np.insert(
                non_target_noise_variance, 0, target_noise_variance
            )
            self.Psi = total_noise_variance - np.diag(self.L.T @ self.L)
            self.Psi_t, self.Psi_nt = np.split(self.Psi, [1], axis=0)

        # Noise correlations correspond to differences in tuning, with
        # increasing latent dimension corresponding to finer differences
        elif noise_structure == 'falloff':
            corr_max = self.noise_kwargs['corr_max']
            corr_min = self.noise_kwargs['corr_min']
            no_corr = (corr_max == 0) and (corr_min == 0)
            # Get noise correlation structure based off tuning preferences
            noise_corr = utils.noise_correlation_matrix(
                tuning_prefs=self.tuning_prefs,
                corr_max=corr_max,
                corr_min=corr_min,
                L=self.noise_kwargs['L_corr'],
                circular_stim=None)
            # Separate non-target portion
            non_target_noise_corr = noise_corr[1:, 1:]

            # Calculate latent factors and private variances for non-target neurons
            non_target_signal_variance = self.non_target_signal_variance()
            non_target_noise_variance = non_target_signal_variance / snr
            non_target_noise_cov = utils.corr2cov(non_target_noise_corr,
                                                  non_target_noise_variance)
            if no_corr:
                self.L_nt = np.zeros((self.K, self.N))
            else:
                lamb, W = np.linalg.eigh(non_target_noise_cov)
                self.L_nt = np.diag(np.sqrt(lamb[-self.K:])) @ W[:, -self.K:].T
            self.Psi_nt = np.diag(non_target_noise_cov) - np.diag(self.L_nt.T @ self.L_nt)

            # Calculate latent factors and private variance for target neuron
            target_signal_variance = self.target_signal_variance()
            target_noise_variance = target_signal_variance / snr
            noise_cov = utils.corr2cov(noise_corr, target_noise_variance)
            lamb, W = np.linalg.eigh(noise_cov)
            L = np.diag(np.sqrt(lamb[-self.K:])) @ W[:, -self.K:].T
            if no_corr:
                self.l_t = np.zeros((self.K, 1))
            else:
                self.l_t = L[:, -1][..., np.newaxis]
            self.Psi_t = noise_cov[-1, -1] - (self.l_t.T @ self.l_t).item()

            # Combine latent factors and private variances
            self.L = np.concatenate((self.L_nt, self.l_t), axis=1)
            self.Psi = np.append(self.Psi_nt, self.Psi_t)

    def generate_samples(
        self, n_samples, bin_width=0.5, rng=None, return_noise=False
    ):
        """Generate samples from the triangular model.

        Parameters
        ----------
        n_samples : int
            The number of samples.
        bin_width : float
            Sets the bin width for the Poisson model.
        rng : {None, int, array_like[ints], SeedSequence, BitGenerator,
               Generator}
            The random number generator or seed for the data samples.

        Returns
        -------
        X : np.ndarray, shape (n_samples, M)
            The tuning features for each stimulus.
        Y : np.ndarray, shape (n_samples, N)
            The non-target neural activity matrix.
        y : np.ndarray, shape (n_samples, 1)
            The target neural activity responses.
        Z : np.ndarray, shape (n_samples, K)
            (Optional) The latent state values.
        """
        rng = np.random.default_rng(rng)
        # Draw values based off parameter design
        if self.parameter_design == 'direct_response':
            if self.stim_kwargs['distribution'] == 'uniform':
                X = rng.uniform(
                    low=self.stim_kwargs['low'],
                    high=self.stim_kwargs['high'],
                    size=(n_samples, self.M)
                )
            elif self.stim_kwargs['distribution'] == 'gaussian':
                X = rng.normal(
                    loc=self.stim_kwargs['loc'],
                    scale=self.stim_kwargs['scale'],
                    size=(n_samples, self.M)
                )
        # Basis functions first require drawing a stimulus value
        elif self.parameter_design == 'basis_functions':
            stimuli = rng.uniform(low=0, high=1, size=n_samples)
            X = utils.calculate_tuning_features(stimuli, self.bf_centers, self.bf_scale)

        # Draw latent activity
        Z = rng.normal(loc=0, scale=1.0, size=(n_samples, self.K))

        if self.model == 'linear':
            # Non-target private variability
            psi_nt = np.sqrt(self.Psi_nt) * rng.normal(
                loc=0,
                scale=1.0,
                size=(n_samples, self.N))
            # Non-target neural activity
            Y = np.dot(X, self.B) + np.dot(Z, self.L_nt) + psi_nt
            # Target private variability
            psi_t = np.sqrt(self.Psi_t) * rng.normal(
                loc=0,
                scale=1.0,
                size=(n_samples, 1))
            # Target neural activity
            y = np.dot(X, self.b) + np.dot(Y, self.a) + np.dot(Z, self.l_t) + psi_t

        elif self.model == 'poisson':
            # Non-target responses
            non_target_pre_exp = np.dot(X, self.B) + np.dot(Z, self.L)
            non_target_mu = np.exp(bin_width * non_target_pre_exp)
            Y = rng.poisson(lam=non_target_mu)
            # Target response
            target_pre_exp = np.dot(X, self.b) + np.dot(Y, self.a) + np.dot(Z, self.l_t)
            target_mu = np.exp(bin_width * target_pre_exp)
            y = rng.poisson(lam=target_mu)

        if return_noise:
            return X, Y, y, Z
        else:
            return X, Y, y

    def generate_coupling_profile(self, tuning_prefs):
        """Generates a coupling profile according to a distribution of preferred
        tunings for the neural population.

        Parameters
        ----------
        tuning_prefs : np.ndarray, shape (N+1,)
            The preferred tunings for the neurons in the population. The first
            index denotes the preferred tuning for the target neuron, the
            remaining N indices denote it for the remaining neurons.

        Returns
        -------
        a : np.ndarray, shape (N, 1)
            The coupling parameters.
        """
        coupling_rng = self.coupling_kwargs['rng']
        n_nonzero_coupling = int((1 - self.coupling_kwargs['sparsity']) * self.N)
        # Get preferred tunings
        target_tuning, non_target_tuning = np.split(tuning_prefs, [1], axis=0)
        # How quickly probability decays with tuning difference
        tuning_diff_decay = self.tuning_kwargs['diff_decay']
        # Calculate and normalize probability of selecting tuning parameter
        probs = np.exp(-np.abs(non_target_tuning - target_tuning) / tuning_diff_decay)
        probs /= probs.sum()
        # Determine coupled and non-coupled indices randomly
        coupled_indices = np.sort(coupling_rng.choice(
            a=np.arange(self.N),
            size=n_nonzero_coupling,
            replace=False,
            p=probs))
        # Draw coupling parameters
        a = np.zeros((self.N, 1))
        a[coupled_indices, 0] = self.draw_parameters(
            size=n_nonzero_coupling,
            **self.coupling_kwargs)
        return a

    def identifiability_transform(self, delta, update=True):
        """Performs an identifiability transform on the parameters in the
        model.

        Parameters
        ----------
        delta : np.ndarray, shape (K,) or (K, 1)
            The identifiability transform.
        update : bool
            If True, class parameters will be updated. Otherwise, the
            transformed parameters will be returned.
        """
        # Make sure delta has the correct number of dimensions
        if delta.ndim == 1:
            delta = delta[..., np.newaxis]

        # Grab latent factors
        l_t = np.copy(self.l_t)
        L_nt = np.copy(self.L_nt)
        # Grab private variances
        Psi_t = self.Psi_t
        Psi_nt = np.diag(self.Psi_nt)

        # Perturbation for coupling terms
        Delta = -np.linalg.solve(Psi_nt + np.dot(L_nt.T, L_nt),
                                 np.dot(L_nt.T, delta))
        # Create augmented variables
        delta_aug = delta + np.dot(L_nt, Delta)
        L_aug = l_t + np.dot(L_nt, self.a)

        # Correction for target private variance
        Psi_t_correction = \
            - 2 * np.dot(Delta.T, np.dot(Psi_nt, self.a)).ravel() \
            - np.dot(Delta.T, np.dot(Psi_nt, Delta)).ravel() \
            - np.dot(delta_aug.T, delta_aug) \
            - 2 * np.dot(L_aug.T, delta_aug)

        # Apply corrections
        a = self.a + Delta
        b = self.b - np.dot(self.B, Delta)
        # Private variability
        Psi_t = (Psi_t + Psi_t_correction).item()
        Psi = np.copy(self.Psi)
        Psi[0] = Psi_t
        # Latent factors
        l_t = (self.l_t.ravel() + delta.ravel())[..., np.newaxis]
        L = np.copy(self.L)
        L[:, 0] = l_t.ravel()

        if update:
            self.Psi_t = Psi_t
            self.Psi[0] = self.Psi_t
            self.l_t = np.copy(l_t)
            self.L = np.copy(L)
            self.a = np.copy(a)
            self.b = np.copy(b)
        else:
            return a, b, Psi, L

    def check_identifiability_conditions(self, a_mask=None, b_mask=None):
        """Checks the conditions for clamping identifiability.

        Parameters
        ----------
        a_mask : np.ndarray
            The selection profile for the coupling parameters. If None, the
            stored mask is used.
        b_mask : np.ndarray
            The selection profile for the tuning parameters. If None, the stored
            mask is used.

        Returns
        -------
        check : bool
            Whether the check passed.
        """
        if a_mask is None:
            a_mask = self.get_masks()[0]
        if b_mask is None:
            b_mask = self.get_masks()[1]
        return utils.check_identifiability_conditions(
            Psi_nt=self.Psi_nt, L_nt=self.L_nt, B=self.B, a_mask=a_mask, b_mask=b_mask
        )

    def non_target_signal_variance(self, limits=(0, 1)):
        """Calculates the variance of the non-target signal, i.e., variance
        coming directly from the tuning.

        Parameters
        ----------
        limits : tuple
            The limits of stimulus.

        Returns
        -------
        variance : np.ndarray, shape (N,)
            The variance of the signal in each non-target neuron.
        """
        if self.parameter_design == 'direct_response':
            self.stim_var = self.calculate_variance(**self.stim_kwargs)
            variance = self.stim_var * np.sum(self.B**2, axis=0)

        elif self.parameter_design == 'basis_functions':
            variance = np.array([utils.bf_sum_var(
                weights=self.B[:, neuron],
                centers=self.bf_centers,
                scale=self.bf_scale,
                limits=limits) for neuron in range(self.N)])
        return variance

    def non_target_variance(self, limits=(0, 1)):
        """Calculates the variance of the non-target neurons.

        Parameters
        ----------
        limits : tuple
            The limits of stimulus.

        Returns
        -------
        variance : np.ndarray, shape (N,)
            The variance of the non-target neural activity.
        """
        signal_variance = self.non_target_signal_variance(limits=limits)
        latent_variance = np.sum(self.L_nt**2, axis=0)
        private_variance = self.Psi_nt
        variance = signal_variance + latent_variance + private_variance
        return variance

    def target_signal_variance(self, limits=(0, 1)):
        """Calculates the variance of the target signal, i.e., variance
        coming directly from the tuning and coupling.

        Parameters
        ----------
        limits : tuple
            The limits of stimulus.

        Returns
        -------
        variance : float
            The variance of the signal in the target neuron.
        """
        tuning_weights = self.b + self.B @ self.a

        if self.parameter_design == 'direct_response':
            self.stim_var = self.calculate_variance(**self.stim_kwargs)
            tuning_variance = self.stim_var * np.sum(tuning_weights**2)
        elif self.parameter_design == 'basis_functions':
            tuning_weights = self.b + self.B @ self.a
            tuning_variance = utils.bf_sum_var(
                weights=tuning_weights.ravel(),
                centers=self.bf_centers,
                scale=self.bf_scale,
                limits=limits)

        latent_variance = np.sum((self.L_nt @ self.a)**2)
        private_variance = self.a.ravel()**2 @ self.Psi_nt
        variance = tuning_variance + latent_variance + private_variance
        return variance

    def target_variance(self, limits=(0, 1)):
        """Calculates the variance of the target signal, i.e., variance
        coming directly from the tuning.

        Parameters
        ----------
        limits : tuple
            The limits of stimulus.

        Returns
        -------
        variance : np.ndarray, shape (N,)
            The variance of the target neuron.
        """
        tuning_weights = self.b + self.B @ self.a

        if self.parameter_design == 'direct_response':
            self.stim_var = self.calculate_variance(**self.stim_kwargs)
            tuning_variance = self.stim_var * np.sum(tuning_weights**2)
        elif self.parameter_design == 'basis_functions':
            tuning_weights = self.b + self.B @ self.a
            tuning_variance = utils.bf_sum_var(
                weights=tuning_weights.ravel(),
                centers=self.bf_centers,
                scale=self.bf_scale,
                limits=limits)

        latent_variance = np.sum((self.l_t + self.L_nt @ self.a)**2)
        private_variance = self.a.ravel()**2 @ self.Psi_nt + self.Psi_t
        variance = tuning_variance + latent_variance + private_variance
        return variance

    def get_masks(self):
        """Get Boolean masks for the main triangular model parameters.

        Returns
        -------
        a_mask, b_mask, B_mask : np.ndarray of bools
            Masks for the coupling, target tuning, and non-target tuning
            parameters.
        """
        a_mask = self.a.ravel() != 0
        b_mask = self.b.ravel() != 0
        B_mask = self.B != 0
        return a_mask, b_mask, B_mask

    def get_noise_cov(self):
        """Gets the noise covariance matrix."""
        return self.L.T @ self.L + np.diag(self.Psi)

    def plot_tuning_curves(self, neuron='all', fax=None, linewidth=1):
        """Plots the tuning curve(s) of the neurons in the triangular model.

        Parameters
        ----------
        neuron : string or array-like, default 'all'
            The neurons to plot. If 'all', all neurons plotted. If 'non-target',
            only non-target tuning curves are plotted. If 'target', only target
            tuning curves are plotted. If array-like, contains the neurons
            to plot directly.
        fax : tuple of mpl.figure and mpl.axes, or None
            The figure and axes. If None, a new set will be created.

        linewidth : float
            The widths of the plotted tuning curves.

        Returns
        -------
        fax : tuple of mpl.figure and mpl.axes
            The figure and axes, with tuning curves plotted.
        """
        fig, ax = plot.check_fax(fax, figsize=(12, 6))

        if neuron == 'all':
            to_plot = np.arange(self.N + 1)
        elif neuron == 'target':
            to_plot = np.array([self.N])
        elif neuron == 'non-target':
            to_plot = np.arange(self.N)
        elif isinstance(neuron, (list, tuple, np.ndarray)):
            to_plot = neuron
        else:
            raise ValueError('Value type for neuron not supported.')

        if self.parameter_design == 'direct_response':
            stimuli = np.arange(self.M)
            for idx in to_plot:
                # Color of tuning curve depends on target, coupled, or non-coupled
                if idx in self.coupled_indices:
                    color = 'black'
                elif idx in self.non_coupled_indices:
                    color = 'gray'
                elif idx == self.N:
                    color = 'red'
                ax.plot(stimuli, self.B_all[:, idx],
                        color=color, marker='o', lw=3)

        elif self.parameter_design == 'basis_functions':
            # Get tuning curves for all neurons
            stimuli, tuning_curves = utils.calculate_tuning_curves(
                B=self.B_all,
                bf_centers=self.bf_centers,
                bf_scale=self.bf_scale)
            # Iterate over tuning curves to plot
            for idx in to_plot:
                # Color of tuning curve depends on target, coupled, or non-coupled
                if idx in self.coupled_indices:
                    color = 'black'
                elif idx in self.non_coupled_indices:
                    color = 'gray'
                elif idx == self.N:
                    color = 'red'
                ax.plot(stimuli,
                        tuning_curves[:, idx],
                        color=color,
                        linewidth=linewidth)
        return fig, ax

    @staticmethod
    def draw_parameters(distribution, size, rng=None, **kwargs):
        """Draws parameters from a distribution according to specified
        parameter values.

        Parameters
        ----------
        distribution : string
            The distribution to draw parameters from.
        size : tuple of ints
            The number of samples to draw, in a desired shape.
        rng: int, RandomState instance or None, optional, default None
            The seed of the pseudo random number generator that selects a
            random feature to update.
        kwargs : dict
            Remaining arguments containing the properties of the distribution
            from which to draw parameters.

        Returns
        -------
        parameters : array-like
            the parameters drawn from the provided distribution
        """
        rng = np.random.default_rng(rng)

        if distribution == 'gaussian':
            parameters = rng.normal(
                loc=kwargs['loc'],
                scale=kwargs['scale'],
                size=size)

        elif distribution == 'laplacian':
            parameters = rng.laplace(
                loc=kwargs['loc'],
                scale=kwargs['scale'],
                size=size)

        elif distribution == 'uniform':
            parameters = rng.uniform(
                low=kwargs['low'],
                high=kwargs['high'],
                size=size)

        # Distribution where probability density increases with magnitude
        # (symmetric about zero)
        elif distribution == 'shifted_exponential':
            samples = truncexpon.rvs(
                b=kwargs['high'],  # cutoff value
                scale=kwargs['scale'],
                size=size,
                seed=rng)
            # Randomly assign each parameter to be positive or negative
            signs = rng.choice([-1, 1], size=size)
            # Shift the samples and apply the sign mask
            floor = kwargs.get('floor', 0.001)
            parameters = signs * (floor + (kwargs['high'] - samples))

        elif distribution == 'lognormal':
            parameters = rng.lognormal(
                mean=kwargs['loc'],
                sigma=kwargs['scale'],
                size=size)

        elif distribution == 'symmetric_lognormal':
            # Each parameter is equally likely to be positive or negative
            signs = rng.choice([-1, 1], size=size)
            parameters = signs * rng.lognormal(
                mean=kwargs['loc'],
                sigma=kwargs['scale'],
                size=size)

        # The Hann window is a discrete squared sine over one period
        elif distribution == 'hann_window':
            indices = np.arange(1, size + 1)
            parameters = kwargs['peak'] * np.sin(np.pi * indices / (size + 1))**2

        # A Hann window, but with noise added to it
        elif distribution == 'noisy_hann_window':
            indices = np.arange(1, size + 1)
            parameters = kwargs['peak'] * np.sin(np.pi * indices / (size + 1))**2
            # add noise, with scale 1/10 that of the peak value
            noise = rng.normal(
                loc=0,
                scale=kwargs.get('noise_scale', kwargs['peak'] / 10.),
                size=size)
            parameters = np.abs(parameters + noise)

        else:
            raise ValueError('Distribution %s not available.' % distribution)

        return parameters

    @staticmethod
    def calculate_variance(distribution, **kwargs):
        """Calculates the variance of a specified univariate distribution.

        Parameters
        ----------
        distribution : string
            The distribution of which to calculate the variance.
        kwargs : dict
            Remaining arguments containing the properties of the distribution
            from which to draw parameters.

        Returns
        -------
        variance : float
            The variance of the distribution specified in props.
        """
        if distribution == 'gaussian':
            sigma = kwargs['scale']
            variance = sigma**2

        elif distribution == 'laplacian':
            b = kwargs['scale']
            variance = 2 * b**2

        elif distribution == 'uniform':
            high = kwargs['high']
            low = kwargs['low']
            variance = (high - low)**2 / 12.

        else:
            raise ValueError('Chosen distribution not available.')

        return variance


class SubsampledModel(TriangularModel):
    """Creates and draws samples from a subsampled triangular model with no shared variability.

    Parameters
    ----------
    model : string
        Specifies either a 'linear' or 'poisson' model.
    parameter_design : string
        The structure to impose on the parameters: either 'random',
        'direct_response' or 'basis_functions'.
    N : int
        The number of coupling parameters.
    coupling_distribution : string
        The distribution from which to draw the coupling parameters.
    coupling_sparsity : float
        The fraction of coupling parameters that are set to zero.
    coupling_loc : float
        Specifies the location of the distribution from which the coupling
        parameters are drawn.
    coupling_scale : float
        Specifies the scale of the distribution from which the coupling
        parameters are drawn.
    coupling_rng : {None, int, array_like[ints], SeedSequence, BitGenerator,
                    Generator}
        The coupling random number generator.
    M : int
        The number of tuning parameters.
    tuning_distribution : string
        The distribution from which to draw the tuning parameters.
    tuning_sparsity : float
        The fraction of tuning parameters that are set to zero.
    tuning_noise_scale : float
        Specifies the scale of noise added to the tuning parameters, if
        desired.
    tuning_peak : float
        The maximum possible value of the tuning curve (relevant for Hann
        window).
    tuning_loc : float
        Specifies the location of the distribution from which the tuning
        parameters are drawn.
    tuning_scale : float
        Specifies the scale of the distribution from which the tuning
        parameters are drawn.
    tuning_low : float
        The minimum value of the tuning parameters. Relevant for a uniform
        tuning distribution.
    tuning_high : float
        The maximum value of the tuning parameters. Relevant for a uniform
        tuning distribution.
    tuning_bound_frac : float
        Specifies the fraction of tuning curve that can be truncated off the
        tuning plane.
    tuning_diff_decay : float
        Specifies the coupling probability decays with tuning difference.
    tuning_bf_scale : float
        The scale parameter for the tuning basis functions. Relevant for the
        'basis_functions' parameter design.
    target_pref_tuning : float
        The preferred tuning of the target neuron. Relevant for the
        'basis_functions' parameter design.
    tuning_add_noise : bool
        If True, adds noise to the tuning parameters.
    tuning_overall_scale : float
        Scaling factor for all tuning parameters.
    tuning_rng : {None, int, array_like[ints], SeedSequence, BitGenerator,
                  Generator}
        The tuning random number generator.
    stim_distribution : string
        The distribution from which to draw the stimulus values.
    stim_kwargs : dict
        Stimulus keyword arguments. If None, it is automatically populated using
        default values.
    subsample_frac : float
        Fraction of coupled neurons to randomly keep
    """
    def __init__(
        self, model='linear', parameter_design='direct_response', N=20,
        coupling_distribution='symmetric_lognormal', coupling_sparsity=0.5,
        coupling_loc=-1, coupling_scale=0.5, coupling_rng=2332, M=20,
        tuning_distribution='noisy_hann_window', tuning_sparsity=0.50,
        tuning_noise_scale=None, tuning_peak=150, tuning_loc=0.,
        tuning_scale=1., tuning_low=0, tuning_high=1., tuning_bound_frac=0.25,
        tuning_diff_decay=1., tuning_bf_scale=None, target_pref_tuning=0.5,
        tuning_add_noise=True, tuning_overall_scale=1., tuning_rng=2332,
        snr=3.,
        stim_distribution='uniform', stim_kwargs=None, warn=True,
        subsample_frac=0.9, subsample_rng=2332
    ):
        super().__init__(
            model=model, parameter_design=parameter_design, N=N,
            coupling_distribution=coupling_distribution, coupling_sparsity=coupling_sparsity,
            coupling_loc=coupling_loc, coupling_scale=coupling_scale, coupling_rng=coupling_rng, M=M,
            tuning_distribution=tuning_distribution, tuning_sparsity=tuning_sparsity,
            tuning_noise_scale=tuning_noise_scale, tuning_peak=tuning_peak, tuning_loc=tuning_loc,
            tuning_scale=tuning_scale, tuning_low=tuning_low, tuning_high=tuning_high, tuning_bound_frac=tuning_bound_frac,
            tuning_diff_decay=tuning_diff_decay, tuning_bf_scale=tuning_bf_scale, target_pref_tuning=target_pref_tuning,
            tuning_add_noise=tuning_add_noise, tuning_overall_scale=tuning_overall_scale, tuning_rng=tuning_rng,
            stim_distribution=stim_distribution, stim_kwargs=stim_kwargs, warn=warn
        )
        self.subsample_kwargs = {
            'subsample_frac': subsample_frac,
            'rng': np.random.default_rng(subsample_rng),
            'N_subsampled': int(N * subsample_frac)
            }
        self.generate_noise_structure()
        self.subsample_model()

    def generate_noise_structure(self):
        """Generates the noise covariance structure for the triangular model."""
        super().generate_noise_structure()
        snr = self.noise_kwargs['snr']

        # Noise correlations exist in specific clusters, with increasing
        # latent dimension increasing the number of clusters
        non_target_signal_variance = self.non_target_signal_variance()
        non_target_noise_variance = non_target_signal_variance / snr
        target_signal_variance = self.target_signal_variance()
        target_noise_variance = target_signal_variance / snr
        self.Psi = np.insert(
            non_target_noise_variance, 0, target_noise_variance
        )
        self.Psi_t, self.Psi_nt = np.split(self.Psi, [1], axis=0)
        self.L *= 0.
        self.L_nt *= 0.
        self.l_t *= 0.

    def subsample_model(self):
        """Subsample coupled units."""
        rng = self.subsample_kwargs['rng']
        subsample_frac = self.subsample_kwargs['subsample_frac']
        N_subsampled = self.subsample_kwargs['N_subsampled']
        all_units = rng.permutation(self.N)
        self.subsampled_units = all_units[:N_subsampled]
        self.removed_units = all_units[N_subsampled:]

        self.a_remove = self.a[self.removed_units].copy()
        self.a_full = self.a
        self.a = self.a[self.subsampled_units].copy()

        self.B_remove = self.B[:, self.removed_units].copy()
        self.B_full = self.B
        self.B = self.B[:, self.subsampled_units].copy()

        self.Psi_nt_remove = self.Psi_nt[self.removed_units].copy()
        self.Psi_full = self.Psi
        self.Psi_nt_full = self.Psi_nt
        Psi_nt_subsample = self.Psi_nt[self.subsampled_units].copy()
        self.Psi = np.insert(
            Psi_nt_subsample, 0, self.Psi_t
        )
        _, self.Psi_nt = np.split(self.Psi, [1], axis=0)

    def generate_samples(
        self, n_samples, bin_width=0.5, rng=None, subsample=True
    ):
        """Generate samples from the triangular model.

        Parameters
        ----------
        n_samples : int
            The number of samples.
        bin_width : float
            Sets the bin width for the Poisson model.
        rng : {None, int, array_like[ints], SeedSequence, BitGenerator,
               Generator}
            The random number generator or seed for the data samples.

        Returns
        -------
        X : np.ndarray, shape (n_samples, M)
            The tuning features for each stimulus.
        Y : np.ndarray, shape (n_samples, N)
            The non-target neural activity matrix.
        y : np.ndarray, shape (n_samples, 1)
            The target neural activity responses.
        """
        N_subsampled = self.subsample_kwargs['N_subsampled']
        rng = np.random.default_rng(rng)
        # Draw values based off parameter design
        if self.parameter_design == 'direct_response':
            if self.stim_kwargs['distribution'] == 'uniform':
                X = rng.uniform(
                    low=self.stim_kwargs['low'],
                    high=self.stim_kwargs['high'],
                    size=(n_samples, self.M)
                )
            elif self.stim_kwargs['distribution'] == 'gaussian':
                X = rng.normal(
                    loc=self.stim_kwargs['loc'],
                    scale=self.stim_kwargs['scale'],
                    size=(n_samples, self.M)
                )
        # Basis functions first require drawing a stimulus value
        elif self.parameter_design == 'basis_functions':
            stimuli = rng.uniform(low=0, high=1, size=n_samples)
            X = utils.calculate_tuning_features(stimuli, self.bf_centers, self.bf_scale)

        # Draw latent activity

        if self.model == 'linear':
            # Non-target private variability
            psi_nt = np.sqrt(self.Psi_nt_full) * rng.normal(
                loc=0,
                scale=1.0,
                size=(n_samples, self.N))
            # Non-target neural activity
            Y = np.dot(X, self.B_full) + psi_nt
            # Target private variability
            psi_t = np.sqrt(self.Psi_t) * rng.normal(
                loc=0,
                scale=1.0,
                size=(n_samples, 1))
            # Target neural activity
            y = np.dot(X, self.b) + np.dot(Y, self.a_full) + psi_t

        elif self.model == 'poisson':
            # Non-target responses
            non_target_pre_exp = np.dot(X, self.B_full)
            non_target_mu = np.exp(bin_width * non_target_pre_exp)
            Y = rng.poisson(lam=non_target_mu)
            # Target response
            target_pre_exp = np.dot(X, self.b) + np.dot(Y, self.a_full)
            target_mu = np.exp(bin_width * target_pre_exp)
            y = rng.poisson(lam=target_mu)

        if subsample:
            Y = Y[:, self.subsampled_units]
        return X, Y, y
