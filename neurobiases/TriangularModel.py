import h5py
import matplotlib.pyplot as plt
import neurobiases.utils as utils
import numpy as np

from scipy.stats import truncexpon
from sklearn.utils import check_random_state


class TriangularModel:
    """Creates and draws samples from a triangular model.

    Parameters
    ----------
    parameters : list of numpy arrays
        A list containing the parameter values describing coupling (A)
        and tuning (Bi, Bj).

    parameter_design : string
        The structure to impose on the parameters.

    coupling_props : dict
        A dictionary detailing the properties of the coupling parameters.
        These may include:
            - coupling dimension (N)
            - coupling sparsity
            - coupling prior
            - coupling distribution parameters

    tuning_props : dict
        A dictionary detailing the properties of the tuning parameters.
        These may include:
            - tuning dimension (M)
            - tuning sparsity
            - tuning prior
            - tuning distribution parameters (loc, scale, low, high, etc.)
            - tuning overlap proportions (f_coupling, f_non_coupling)

    stim_props : dict
        A dictionary detailing the properties of the stimulus parameters.
        These may include:
            - stimulus prior
            - stimulus distribution parameters (loc, scale, etc.)

    kappa : float, optional
        The inverse signal-to-noise ratio. If None, uses preset value of
        kappa.

    rho_c : float, optional
        The value of noise correlation in each block of neurons. If None,
        uses preset value of rho_c.

    rho_b : float, optional
        The value of noise correlation amongst all neurons not in the same
        block (i.e., 'background' correlation). If None, uses preset value
        of rho_b.

    K : int, optional
        The number of latent dimensions. If None, uses preset value of K.

    random_state: int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a
        random feature to update. If int, random_state is the seed used
        by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random
        number generator is the RandomState instance used by np.random.
    """
    def __init__(
        self, model='linear', parameter_path=None, parameter_design='basis_functions',
        n_latent=1, signal_to_noise=3.,
        coupling_kwargs=None, tuning_kwargs=None, stim_kwargs=None,
        random_state=None
    ):
        self.model = model
        self.parameter_design = parameter_design

        # check if parameters are provided in a file
        if parameter_path is None:
            self.coupling_kwargs = coupling_kwargs
            self.tuning_kwargs = tuning_kwargs
            self.stim_kwargs = stim_kwargs
            self.N = self.coupling_kwargs.get('N', 100)
            self.M = self.tuning_kwargs.get('M', 10)
            self.coupling_kwargs['random_state'] = check_random_state(
                coupling_kwargs.get('random_state', None)
            )
            self.tuning_kwargs['random_state'] = check_random_state(
                tuning_kwargs.get('random_state', None)
            )
            self.random_state = check_random_state(random_state)
            self.n_latent = n_latent
            self.signal_to_noise = signal_to_noise
            # create parameters according to preferred design
            self.a, self.b, self.B = self.generate_triangular_model()
            self.generate_noise_structure()
        else:
            parameter_file = h5py.File(parameter_path, 'r')
            # coupling parameters and kwargs
            coupling = parameter_file['coupling']
            self.a = coupling['a'][:][..., np.newaxis]
            self.coupling_kwargs = utils.read_attribute_dict(coupling.attrs)
            # tuning parameters and kwargs
            tuning = parameter_file['tuning']
            self.b = tuning['b'][:][..., np.newaxis]
            self.B = tuning['B'][:]
            self.tuning_kwargs = utils.read_attribute_dict(tuning.attrs)

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
            b = B_all[:, 0]
            B = B_all[:, 1:]

        elif self.parameter_design == 'basis_functions':
            # get where each basis function is centered
            self.bf_pref_tuning = np.linspace(0, 1, self.M)
            self.bf_scale = self.tuning_kwargs.get('bf_scale', 0.5 / self.M)

            # get preferred tunings for each neuron
            non_target_tuning = np.linspace(0, 1, self.N)
            target_tuning = self.tuning_kwargs.get('target_pref_tuning', 0.5)
            all_tunings = np.append(non_target_tuning, target_tuning)

            # differences between preferred tuning and bf locations
            tuning_diffs = np.abs(np.subtract.outer(self.bf_pref_tuning, all_tunings))
            # calculate preferred tuning, with possible scaling
            B_all = 1 - tuning_diffs
            # apply a sigmoid to the coefficients to flatten
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
            B, b = np.split(B_all, [self.N], axis=1)

            # decide which neurons are coupled according to their tuning
            # distance with the target neuron
            p = np.maximum(1 - np.abs(target_tuning - non_target_tuning), 0)
            p = p / p.sum()
            self.coupled_indices = np.sort(coupling_random_state.choice(
                a=np.arange(self.N),
                size=n_nonzero_coupling,
                replace=False,
                p=p))
            self.non_coupled_indices = np.setdiff1d(np.arange(self.N),
                                                    self.coupled_indices)
            # draw coupling parameters
            a = np.zeros((self.N, 1))
            a[self.coupled_indices, 0] = self.draw_parameters(
                size=n_nonzero_coupling,
                **self.coupling_kwargs)

        return a, b, B

    def generate_noise_structure(self):
        self.L = np.random.normal(loc=0, scale=1, size=(self.n_latent, self.N))**2
        self.l_t = np.random.normal(loc=0, scale=1., size=(self.n_latent, 1))**2
        stimuli = self.random_state.uniform(low=0, high=1, size=100000)
        X = utils.calculate_tuning_features(stimuli, self.bf_pref_tuning, self.bf_scale)
        variances = np.var(np.dot(X, self.B), axis=0)
        self.L *= np.sqrt(variances / self.signal_to_noise) / np.sqrt(np.sum(self.L**2, axis=0))
        self.l_t *= np.sqrt(variances.mean() / self.signal_to_noise) / np.sqrt(np.sum(self.l_t**2))

    def generate_samples(self, n_samples, bin_width=0.5, random_state=None):
        if random_state is None:
            random_state = self.random_state
        else:
            random_state = check_random_state(random_state)

        if self.parameter_design == 'basis_functions':
            stimuli = random_state.uniform(low=0, high=1, size=n_samples)
            X = utils.calculate_tuning_features(stimuli, self.bf_pref_tuning, self.bf_scale)
            Z = np.random.normal(loc=0, scale=1.0, size=(n_samples, self.n_latent))
            # non-target responses
            non_target_pre_exp = np.dot(X, self.B) + np.dot(Z, self.L)
            non_target_mu = np.exp(bin_width * non_target_pre_exp)
            Y = random_state.poisson(lam=non_target_mu)
            # target response
            target_pre_exp = np.dot(X, self.b) + np.dot(Y, self.a) + np.dot(Z, self.l_t)
            target_mu = np.exp(bin_width * target_pre_exp)
            y = random_state.poisson(lam=target_mu)
            return stimuli, X, Y, y

    def plot_tuning_curves(self, fax=None, linewidth=1):
        if fax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        else:
            fig, ax = fax

        if self.parameter_design == 'basis_functions':
            stimuli, non_target_tuning = utils.calculate_tuning_curves(
                B=self.B, bf_pref_tuning=self.bf_pref_tuning, bf_scale=self.bf_scale,
            )
            _, target_tuning = utils.calculate_tuning_curves(
                B=self.b, bf_pref_tuning=self.bf_pref_tuning, bf_scale=self.bf_scale
            )
            # plot target responses
            ax.plot(stimuli,
                    target_tuning,
                    color='r',
                    linewidth=linewidth)
            # plot coupled responses
            for neuron in self.coupled_indices:
                ax.plot(stimuli,
                        non_target_tuning[:, neuron],
                        color='black',
                        linewidth=linewidth)
            for neuron in self.non_coupled_indices:
                ax.plot(stimuli,
                        non_target_tuning[:, neuron],
                        color='gray',
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
                scale=kwargs['peak'] / 10.,
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
        # the key 'prior' specifies the type of distribution
        distribution = kwargs['prior']

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
