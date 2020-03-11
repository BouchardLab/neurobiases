import h5py
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
        coupling_kwargs=None, tuning_kwargs=None, stim_kwargs=None
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
            # create parameters according to preferred design
            self.a, self.b, self.B = self.generate_tm_parameters()
        else:
            parameter_file = h5py.File(parameter_path, 'r')
            coupling = parameter_file['coupling']
            self.a = coupling['a'][:]
            self.coupling_kwargs = {}
            for key, val in coupling.attrs.items():
                if val == '':
                    self.coupling_kwargs[key] = None
                else:
                    self.coupling_props[key] = val
            self.a = self.a.reshape(self.coupling_kwargs['N'], 1)

    def generate_triangular_model(self):
        """Generate model parameters in the triangular model according to a
        variety of design critera.

        Parameters
        ----------
        design : string
            The structure that the coupling and tuning parameters should
            possess.

        Returns
        -------
        A : nd-array, shape (N, 1).
            Coupling parameters.

        B : nd-array, shape (M, N + 1).
            Tuning parameters.
        """
        # initialize parameter arrays
        a = np.zeros((self.N, 1))
        B_all = np.zeros((self.M, self.N + 1))

        n_nonzero_tuning = int((1 - self.tuning_kwargs['sparsity']) * self.M)
        n_nonzero_coupling = int((1 - self.coupling_kwargs['sparsity']) * self.N)

        # obtain random states
        tuning_random_state = self.tuning_props.get('random_state', None)
        tuning_random_state = check_random_state(tuning_random_state)
        coupling_random_state = self.coupling_props.get('random_state', None)
        coupling_random_state = check_random_state(coupling_random_state)

        # randomly assign selection profiles
        if self.parameter_design == 'random':
            # draw coupling parameters
            nonzero_a = self.draw_parameters(
                distribution=self.coupling_kwargs['distribution'],
                size=n_nonzero_coupling,
                random_state=coupling_random_state,
                **self.coupling_kwargs)
            # draw all tuning parameters jointly
            nonzero_B = self.draw_parameters(
                distribution=self.tuning_kwargs['distribution'],
                size=(n_nonzero_tuning, self.N + 1),
                random_state=tuning_random_state,
                **self.tuning_kwargs)

            # store the non-zero values
            a[:n_nonzero_coupling, 0] = nonzero_a
            B_all[:n_nonzero_tuning, :] = nonzero_B

            # shuffle the parameters in place
            coupling_random_state.shuffle(a)
            # for tuning, we'll shuffle rows separately
            [tuning_random_state(B_all[:, idx]) for idx in range(self.N + 1)]
            b = B_all[:, 0]
            B = B_all[:, 1:]

        elif self.parameter_design == 'basis_functions':
            

        return a, b, B

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

        # distribution where probability increases with magnitude
        # symmetrized around zero
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
