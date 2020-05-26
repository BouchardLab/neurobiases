import h5py
import neurobiases.utils as utils
import neurobiases.plot as plot
import numpy as np

from scipy.stats import truncexpon
from sklearn.utils import check_random_state


class TriangularModel:
    """Creates and draws samples from a triangular model.

    Parameters
    ----------
    model : string
        Specifies either a 'linear' or 'poisson' model.
    parameter_design : string
        The structure to impose on the parameters: either 'random',
        'direct_response' or 'basis_functions'.
    coupling_kwargs : dict
        A dictionary detailing the properties of the coupling parameters.
        These may include:
            - coupling dimension (N)
            - coupling sparsity
            - coupling distribution
            - coupling distribution parameters
    tuning_kwargs : dict
        A dictionary detailing the properties of the tuning parameters.
        These may include:
            - tuning dimension (M)
            - tuning sparsity
            - tuning distribution
            - tuning distribution parameters (loc, scale, low, high, etc.)
    stim_kwargs : dict
        A dictionary detailing the properties of the stimulus parameters.
        These may include:
            - stimulus distribution
            - stimulus distribution parameters (loc, scale, etc.)
    noise_kwargs : dict
        A dictionary detailing the properties of the shared variability
        parameters.
        These may include:
            - The number of latent factors.
            - The maximum and minimum noise correlations.
    random_state: int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a
        random feature to update. If int, random_state is the seed used
        by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random
        number generator is the RandomState instance used by np.random.
    """
    def __init__(
        self, model='linear', parameter_path=None, parameter_design='basis_functions',
        coupling_kwargs=None, tuning_kwargs=None, stim_kwargs=None,
        noise_kwargs=None, random_state=None
    ):
        self.model = model
        self.parameter_design = parameter_design

        # check if parameters are provided in a file
        if parameter_path is None:
            self.coupling_kwargs = coupling_kwargs
            self.tuning_kwargs = tuning_kwargs
            self.stim_kwargs = stim_kwargs
            self.noise_kwargs = noise_kwargs
            self.N = self.coupling_kwargs.get('N', 100)
            self.M = self.tuning_kwargs.get('M', 10)
            self.K = self.noise_kwargs.get('K', 2)
            self.coupling_kwargs['random_state'] = check_random_state(
                coupling_kwargs.get('random_state', None)
            )
            self.tuning_kwargs['random_state'] = check_random_state(
                tuning_kwargs.get('random_state', None)
            )
            self.noise_kwargs['random_state'] = check_random_state(
                noise_kwargs.get('random_state', None)
            )
            self.random_state = check_random_state(random_state)
            # create parameters according to preferred design
            self.a, self.b, self.B = self.generate_triangular_model()
            self.B_all = np.concatenate((self.B, self.b), axis=1)
            self.generate_noise_structure()
        else:
            parameter_file = h5py.File(parameter_path, 'r')
            # coupling parameters and kwargs
            coupling = parameter_file['coupling']
            self.a = coupling['a'][:][..., np.newaxis]
            self.coupling_kwargs = utils.copy_attribute_dict(coupling.attrs)
            # tuning parameters and kwargs
            tuning = parameter_file['tuning']
            self.b = tuning['b'][:][..., np.newaxis]
            self.B = tuning['B'][:]
            self.B_all = np.concatenate((self.B, self.b), axis=1)
            self.tuning_kwargs = utils.copy_attribute_dict(tuning.attrs)

    def generate_triangular_model(self):
        """Generate model parameters in the triangular model according to a
        variety of design critera.

        Returns
        -------
        a : np.ndarray, shape (N, 1).
            The coupling parameters.
        b : np.ndarray, shape (M, 1).
            The target tuning parameters.
        B : np.ndarray, shape (M, N)
            The non-target tuning parameters.
        """
        # initialize parameter arrays
        a = np.zeros((self.N, 1))
        B_all = np.zeros((self.M, self.N + 1))
        # calculate number of non-zero parameters using sparsity
        n_nonzero_tuning = int((1 - self.tuning_kwargs['sparsity']) * self.M)
        n_nonzero_coupling = int((1 - self.coupling_kwargs['sparsity']) * self.N)
        # get random states
        coupling_random_state = self.coupling_kwargs['random_state']
        tuning_random_state = self.tuning_kwargs['random_state']

        # randomly assign selection profiles
        if self.parameter_design == 'random':
            # draw coupling parameters
            nonzero_a = self.draw_parameters(
                size=n_nonzero_coupling,
                **self.coupling_kwargs)
            # draw all tuning parameters jointly
            nonzero_B_all = self.draw_parameters(
                size=(n_nonzero_tuning, self.N + 1),
                **self.tuning_kwargs)

            # store the non-zero values
            a[:n_nonzero_coupling, 0] = nonzero_a
            B_all[:n_nonzero_tuning, :] = nonzero_B_all
            # shuffle the parameters in place
            coupling_random_state.shuffle(a)
            # for tuning, we'll shuffle rows separately
            [tuning_random_state(B_all[:, idx]) for idx in range(self.N + 1)]
            b, B = np.split(B_all, [1], axis=1)

        elif self.parameter_design == 'direct_response':
            b = np.zeros((self.M, 1))
            B = np.zeros((self.M, self.N))

            # draw the non-zero parameters for the target neuron
            nonzero_b = self.draw_parameters(
                size=n_nonzero_tuning,
                **self.tuning_kwargs)
            # the offset of the target tuning curve
            b_offset_idx = int(0.5 * (self.M - n_nonzero_tuning))
            # set the non-zero parameters within the tuning window
            b[b_offset_idx:b_offset_idx + n_nonzero_tuning, 0] = nonzero_b

            # offsets for the non-target neuron
            bound_frac = self.tuning_kwargs.get('bound_frac', 0.25)
            lower_bound = -int(bound_frac * n_nonzero_tuning)
            upper_bound = self.M - int((1 - bound_frac) * n_nonzero_tuning)
            offsets = np.linspace(lower_bound, upper_bound, self.N).astype('int')

            # iterate over neurons/offsets, assigning tuning curves
            for neuron_idx, offset in enumerate(offsets):
                tuning_curve = self.draw_parameters(
                    size=n_nonzero_tuning,
                    **self.tuning_kwargs)

                # tuning curve ends up on the left side of the tuning plane
                if offset < 0:
                    new_offset = n_nonzero_tuning + offset
                    B[:new_offset, neuron_idx] = tuning_curve[-new_offset:]
                # tuning curve ends up on the right side of the tuning plane
                elif offset >= self.M - n_nonzero_tuning + 1:
                    B[offset:, neuron_idx] = tuning_curve[:(self.M - offset)]
                # tuning curve is in the middle of the plane
                else:
                    B[offset:offset + n_nonzero_tuning, neuron_idx] = tuning_curve

            B_all = np.concatenate((b, B), axis=1)
            # calculate tuning preferences
            self.tuning_prefs = np.argmax(B_all, axis=0)
            a = self.generate_coupling_profile(self.tuning_prefs)

        elif self.parameter_design == 'basis_functions':
            # get basis function centers and width
            self.bf_centers = np.linspace(0, 1, self.M)
            self.bf_scale = self.tuning_kwargs.get('bf_scale', 0.25 / self.M)

            # get preferred tunings for each neuron
            target_tuning = self.tuning_kwargs.get('target_pref_tuning', 0.5)
            non_target_tuning = np.linspace(0, 1, self.N)
            all_tunings = np.append(target_tuning, non_target_tuning)

            # calculate differences between preferred tuning and bf locations
            tuning_diffs = np.abs(np.subtract.outer(self.bf_centers, all_tunings))
            # calculate preferred tuning, with possible scaling
            B_all = 1 - tuning_diffs
            # apply a sigmoid to flatten coefficients
            for idx, B in enumerate(B_all.T):
                B_all[:, idx] = utils.sigmoid(
                    B, phase=B.mean(), b=5./(B.max() - B.min())
                )
            # add noise to tuning parameters if desired
            if self.tuning_kwargs.get('add_noise', True):
                noise = tuning_random_state.normal(
                    loc=0.,
                    scale=self.tuning_kwargs.get('noise_scale', 0.25),
                    size=(self.M, self.N + 1))
                B_all += noise
            B_all = np.abs(B_all) * self.tuning_kwargs.get('scale', 1)

            # for each neuron, get selection profile
            for idx in range(self.N + 1):
                # get tuning differences with bfs
                tuning_diff = tuning_diffs[:, idx]
                # choose zero indices by chance, weighted by tuning diff
                nonzero_indices = np.argsort(tuning_diff)[:n_nonzero_tuning]
                # set parameters to zero
                zero_indices = np.setdiff1d(np.arange(self.M), nonzero_indices)
                B_all[zero_indices, idx] = 0
            b, B = np.split(B_all, [1], axis=1)
            # calculate preferred tuning for each neuron
            self.tuning_prefs = utils.calculate_pref_tuning(
                B=B_all, bf_centers=self.bf_centers, bf_scale=self.bf_scale,
                n_stimuli=10000, limits=(0, 1)
            )
            # generate coupling parameters
            a = self.generate_coupling_profile(self.tuning_prefs)

        self.coupled_indices = np.argwhere(a.ravel()).ravel()
        self.non_coupled_indices = np.setdiff1d(np.arange(self.N), self.coupled_indices)
        return a, b, B

    def generate_noise_structure(self):
        """Generates the noise covariance structure for the triangular model."""
        noise_structure = self.noise_kwargs['noise_structure']
        snr = self.noise_kwargs.get('snr', 3)

        # noise correlations exist in specific clusters, with increasing
        # latent dimension increasing the number of clusters
        if noise_structure == 'clusters':
            corr_cluster = self.noise_kwargs.get('corr_cluster', 0.1)
            corr_back = self.noise_kwargs.get('corr_back', 0.)

            # calculate non-target signal and noise variances
            non_target_signal_variance = self.non_target_signal_variance()
            non_target_noise_variance = non_target_signal_variance / snr
            # latent factors non-target neurons; one extra factor for now
            L_nt = np.zeros((self.K + 1, self.N))
            # one basis vector will provide the background noise correlation
            L_nt[-1] = np.sqrt(corr_back * non_target_noise_variance)
            # split up the variances into the K clusters of correlated neurons
            variance_clusters = np.array_split(np.sqrt(non_target_noise_variance), self.K)
            idx_clusters = np.array_split(np.arange(self.N), self.K)
            # iterate over the K clusters, each of which will set a latent factor
            for lf_idx, (variances, idxs) in enumerate(zip(variance_clusters, idx_clusters)):
                # place the correct values in the latent factor
                L_nt[lf_idx, idxs] = variances * np.sqrt(corr_cluster - corr_back)
            # store these for now so that we can calculate target signal var
            self.L_nt = np.copy(L_nt)
            self.Psi_nt = non_target_noise_variance - np.diag(L_nt.T @ L_nt)

            # calculate latent factors and private variance for target neuron
            target_signal_variance = self.target_signal_variance()
            target_noise_variance = target_signal_variance / snr
            # place the target neuron in the middle group
            target_group_idx = int(np.ceil(self.K / 2 - 1))
            # calculate target latent factor
            l_t = np.zeros((self.K + 1, 1))
            l_t[target_group_idx] = \
                np.sqrt(target_noise_variance) * np.sqrt(corr_cluster - corr_back)
            # add on the background noise correlation term
            l_t[-1] = np.sqrt(corr_back * target_noise_variance)

            # combine latent factors and private variances
            self.L = np.concatenate((l_t, L_nt), axis=1)
            self.L = utils.symmetric_low_rank_approx(self.L.T @ self.L, self.K).T
            self.l_t, self.L_nt = np.split(self.L, [1], axis=1)
            # calculate private variances
            total_noise_variance = np.insert(non_target_noise_variance, 0, target_noise_variance)
            self.Psi = total_noise_variance - np.diag(self.L.T @ self.L)
            self.Psi_t, self.Psi_nt = np.split(self.Psi, [1], axis=0)

        # noise correlations correspond to differences in tuning, with
        # increasing latent dimension corresponding to finer differences
        elif noise_structure == 'falloff':
            corr_max = self.noise_kwargs.get('corr_max', 0.2)
            corr_min = self.noise_kwargs.get('corr_min', -0.05)
            no_corr = (corr_max == 0) and (corr_min == 0)
            # get noise correlation structure based off tuning preferences
            noise_corr = utils.noise_correlation_matrix(
                tuning_prefs=self.tuning_prefs,
                corr_max=corr_max,
                corr_min=corr_min,
                L=self.noise_kwargs.get('L_corr', 1.),
                circular_stim=None)
            # separate non-target portion
            non_target_noise_corr = noise_corr[:-1, :-1]

            # calculate latent factors and private variances for non-target neurons
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

            # calculate latent factors and private variance for target neuron
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

            # combine latent factors and private variances
            self.L = np.concatenate((self.L_nt, self.l_t), axis=1)
            self.Psi = np.append(self.Psi_nt, self.Psi_t)

    def generate_samples(
        self, n_samples, bin_width=0.5, random_state=None, return_noise=False
    ):
        """Generate samples from the triangular model.

        Parameters
        ----------
        n_samples : int
            The number of samples.
        bin_width : float
            Sets the bin width for the Poisson model.
        random_state : RandomState or None
            The RandomState instance to draw samples with. If None, the
            RandomState instance of the class is used.

        Returns
        -------
        stimuli : np.ndarray, shape (n_samples,)
            The stimuli that generated the samples.
        X : np.ndarray, shape (n_samples, M)
            The tuning features for each stimulus.
        Y : np.ndarray, shape (n_samples, N)
            The non-target neural activity matrix.
        y : np.ndarray, shape (n_samples, 1)
            The target neural activity responses.
        """
        if random_state is None:
            random_state = self.random_state
        else:
            random_state = check_random_state(random_state)

        if self.parameter_design == 'direct_response':
            X = random_state.uniform(
                low=self.stim_kwargs['low'], high=self.stim_kwargs['high'],
                size=(n_samples, self.M)
            )
        elif self.parameter_design == 'basis_functions':
            # draw stimulus and tuning features
            stimuli = random_state.uniform(low=0, high=1, size=n_samples)
            X = utils.calculate_tuning_features(stimuli, self.bf_centers, self.bf_scale)

        # draw latent activity
        Z = random_state.normal(loc=0, scale=1.0, size=(n_samples, self.K))

        if self.model == 'linear':
            # non-target private variability
            psi_nt = np.sqrt(self.Psi_nt) * random_state.normal(
                loc=0,
                scale=1.0,
                size=(n_samples, self.N))
            # non-target neural activity
            Y = np.dot(X, self.B) + np.dot(Z, self.L_nt) + psi_nt
            # target private variability
            psi_t = np.sqrt(self.Psi_t) * random_state.normal(
                loc=0,
                scale=1.0,
                size=(n_samples, 1))
            # target neural activity
            y = np.dot(X, self.b) + np.dot(Y, self.a) + np.dot(Z, self.l_t) + psi_t

        elif self.model == 'poisson':
            # non-target responses
            non_target_pre_exp = np.dot(X, self.B) + np.dot(Z, self.L)
            non_target_mu = np.exp(bin_width * non_target_pre_exp)
            Y = random_state.poisson(lam=non_target_mu)
            # target response
            target_pre_exp = np.dot(X, self.b) + np.dot(Y, self.a) + np.dot(Z, self.l_t)
            target_mu = np.exp(bin_width * target_pre_exp)
            y = random_state.poisson(lam=target_mu)

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
        coupling_random_state = self.coupling_kwargs['random_state']
        n_nonzero_coupling = int((1 - self.coupling_kwargs['sparsity']) * self.N)
        # get preferred tunings
        target_tuning, non_target_tuning = np.split(tuning_prefs, [1], axis=0)
        # how quickly probability decays with tuning difference
        tuning_diff_decay = self.tuning_kwargs.get('tuning_diff_decay', 1)
        # calculate and normalize probability of selecting tuning parameter
        probs = np.exp(-np.abs(non_target_tuning - target_tuning) / tuning_diff_decay)
        probs /= probs.sum()
        # determine coupled and non-coupled indices randomly
        coupled_indices = np.sort(coupling_random_state.choice(
            a=np.arange(self.N),
            size=n_nonzero_coupling,
            replace=False,
            p=probs))
        # draw coupling parameters
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
            If true, class parameters will be updated. Otherwise, the
            transformed parameters will be returned.
        """
        # make sure delta has the correct number of dimensions
        if delta.ndim == 1:
            delta = delta[..., np.newaxis]

        # grab latent factors
        l_t = np.copy(self.l_t)
        L_nt = np.copy(self.L_nt)
        # grab private variances
        Psi_t = self.Psi_t
        Psi_nt = np.diag(self.Psi_nt)

        # perturbation for coupling terms
        Delta = -np.linalg.solve(Psi_nt + np.dot(L_nt.T, L_nt),
                                 np.dot(L_nt.T, delta))
        # create augmented variables
        delta_aug = delta + np.dot(L_nt, Delta)
        L_aug = l_t + np.dot(L_nt, self.a)

        # correction for target private variance
        Psi_t_correction = \
            - 2 * np.dot(Delta.T, np.dot(Psi_nt, self.a)).ravel() \
            - np.dot(Delta.T, np.dot(Psi_nt, Delta)).ravel() \
            - np.dot(delta_aug.T, delta_aug) \
            - 2 * np.dot(L_aug.T, delta_aug)

        # apply corrections
        a = self.a + Delta
        b = self.b - np.dot(self.B, Delta)
        # private variability
        Psi_t = (Psi_t + Psi_t_correction).item()
        Psi = np.copy(self.Psi)
        Psi[0] = Psi_t
        # latent factors
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

        # figure out which neurons need to be plotted
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
                # color of tuning curve depends on target, coupled, or non-coupled
                if idx in self.coupled_indices:
                    color = 'black'
                elif idx in self.non_coupled_indices:
                    color = 'gray'
                elif idx == self.N:
                    color = 'red'
                ax.plot(stimuli, self.B_all[:, idx],
                        color=color, marker='o', lw=3)

        elif self.parameter_design == 'basis_functions':
            # get tuning curves for all neurons
            stimuli, tuning_curves = utils.calculate_tuning_curves(
                B=self.B_all,
                bf_centers=self.bf_centers,
                bf_scale=self.bf_scale)
            # iterate over tuning curves to plot
            for idx in to_plot:
                # color of tuning curve depends on target, coupled, or non-coupled
                if idx in self.coupled_indices:
                    color = 'black'
                elif idx in self.non_coupled_indices:
                    color = 'gray'
                elif idx == self.N:
                    color = 'red'
                # plot tuning curve
                ax.plot(stimuli,
                        tuning_curves[:, idx],
                        color=color,
                        linewidth=linewidth)
        return fig, ax

    @staticmethod
    def draw_parameters(distribution, size, random_state=None, **kwargs):
        """Draws parameters from a distribution according to specified
        parameter values.

        Parameters
        ----------
        distribution : string
            The distribution to draw parameters from.
        size : tuple of ints
            The number of samples to draw, in a desired shape.
        random_state: int, RandomState instance or None, optional, default None
            The seed of the pseudo random number generator that selects a
            random feature to update. If int, random_state is the seed used by
            the random number generator; If RandomState instance, random_state
            is the random number generator; If None, the random number
            generator is the RandomState instance used by np.random.
        kwargs : dict
            Remaining arguments containing the properties of the distribution
            from which to draw parameters.

        Returns
        -------
        parameters : array-like
            the parameters drawn from the provided distribution
        """
        random_state = check_random_state(random_state)

        if distribution == 'gaussian':
            parameters = random_state.normal(
                loc=kwargs['loc'],
                scale=kwargs['scale'],
                size=size)

        elif distribution == 'laplacian':
            parameters = random_state.laplace(
                loc=kwargs['loc'],
                scale=kwargs['scale'],
                size=size)

        elif distribution == 'uniform':
            parameters = random_state.uniform(
                low=kwargs['low'],
                high=kwargs['high'],
                size=size)

        # distribution where probability density increases with magnitude
        # (symmetric about zero)
        elif distribution == 'shifted_exponential':
            samples = truncexpon.rvs(
                b=kwargs['high'],  # cutoff value
                scale=kwargs['scale'],
                size=size,
                random_state=random_state)
            # randomly assign each parameter to be positive or negative
            signs = random_state.choice([-1, 1], size=size)
            # shift the samples and apply the sign mask
            floor = kwargs.get('floor', 0.001)
            parameters = signs * (floor + (kwargs['high'] - samples))

        elif distribution == 'lognormal':
            parameters = random_state.lognormal(
                mean=kwargs['loc'],
                sigma=kwargs['scale'],
                size=size)

        elif distribution == 'symmetric_lognormal':
            # each parameter is equally likely to be positive or negative
            signs = random_state.choice([-1, 1], size=size)
            parameters = signs * random_state.lognormal(
                mean=kwargs['loc'],
                sigma=kwargs['scale'],
                size=size)

        # the Hann window is a discrete squared sine over one period
        elif distribution == 'hann_window':
            indices = np.arange(1, size + 1)
            parameters = kwargs['peak'] * np.sin(np.pi * indices / (size + 1))**2

        # a Hann window, but with noise added to it
        elif distribution == 'noisy_hann_window':
            indices = np.arange(1, size + 1)
            parameters = kwargs['peak'] * np.sin(np.pi * indices / (size + 1))**2
            # add noise, with scale 1/10 that of the peak value
            noise = random_state.normal(
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

    @staticmethod
    def generate_kwargs(
        parameter_design='direct_response', M=50, tuning_sparsity=0.75,
        tuning_noise_scale=None, tuning_distribution='noisy_hann_window',
        tuning_peak=150, tuning_bound_frac=0.25, tuning_diff_decay=1.,
        tuning_random_state=2332, tuning_bf_scale=None, target_pref_tuning=0.5,
        tuning_add_noise=True, tuning_overall_scale=1., N=20,
        coupling_sparsity=0.5, coupling_distribution='symmetric_lognormal',
        coupling_loc=-1, coupling_scale=0.5, coupling_random_state=2332, K=2,
        snr=3, noise_structure='clusters', corr_cluster=0.2, corr_back=0.,
        corr_max=0.3, corr_min=0.0, L_corr=1, stim_distribution='uniform',
        stim_low=0, stim_high=1
    ):
        """Generates a set of keyword argument dictionaries for the piecewise
        formulated triangular model."""
        tuning_kwargs = {
            'M': M,
            'sparsity': tuning_sparsity,
            'tuning_diff_decay': tuning_diff_decay,
            'random_state': tuning_random_state,
        }
        if tuning_noise_scale is not None:
            tuning_kwargs['noise_scale'] = tuning_noise_scale

        if parameter_design == 'direct_response':
            tuning_kwargs['distribution'] = tuning_distribution
            tuning_kwargs['peak'] = tuning_peak
            tuning_kwargs['bound_frac'] = tuning_bound_frac
        elif parameter_design == 'basis_functions':
            tuning_kwargs['target_pref_tuning'] = target_pref_tuning
            tuning_kwargs['add_noise'] = tuning_add_noise
            tuning_kwargs['overall_scale'] = tuning_overall_scale
            if tuning_bf_scale is not None:
                tuning_kwargs['bf_scale'] = tuning_bf_scale

        coupling_kwargs = {
            'N': N,
            'sparsity': coupling_sparsity,
            'distribution': coupling_distribution,
            'loc': coupling_loc,
            'scale': coupling_scale,
            'random_state': coupling_random_state,
        }
        if noise_structure == 'clusters':
            noise_kwargs = {
                'K': K,
                'noise_structure': noise_structure,
                'snr': snr,
                'corr_back': corr_back,
                'corr_cluster': corr_cluster
            }
        elif noise_structure == 'falloff':
            noise_kwargs = {
                'K': K,
                'noise_structure': noise_structure,
                'snr': snr,
                'corr_max': corr_max,
                'corr_min': corr_min,
                'L_corr': L_corr
            }
        stim_kwargs = {
            'distribution': stim_distribution,
            'low': stim_low,
            'high': stim_high
        }
        return tuning_kwargs, coupling_kwargs, noise_kwargs, stim_kwargs
