from numpy import random as np_random

'''
pos - positional arguments
opt - optional arguments (except argument \"size\")
'''
DIST_ARGS_DICT = {
        'beta': {'pos': ['a', 'b'], 'opt': []},
        'binomial': {'pos': ['n', 'p'], 'opt': []},
        'chisquare': {'pos': ['df'], 'opt': []},
        'dirichlet': {'pos': ['alpha'], 'opt': []},
        'exponential': {'pos': [], 'opt': ['scale']},
        'f': {'pos': ['dfnum', 'dfden'], 'opt': []},
        'gamma': {'pos': ['shape'], 'opt': ['scale']},
        'geometric': {'pos': ['p'], 'opt': []},
        'gumbel': {'pos': [], 'opt': ['loc', 'scale']},
        'hypergeometric': {'pos': ['ngood', 'nbad', 'nsample'], 'opt': []},
        'laplace': {'pos': [], 'opt': ['loc', 'scale']},
        'logistic': {'pos': [], 'opt': ['loc', 'scale']},
        'lognormal': {'pos': [], 'opt': ['mean', 'sigma']},
        'logseries': {'pos': ['p'], 'opt': []}, 
        'multinomial': {'pos': ['n', 'pvals'], 'opt': []},
        'multivariate_normal': {'pos': ['mean', 'cov'], 'opt': []},
        'negative_binomial': {'pos': ['n', 'p'], 'opt': []},
        'noncentral_chisquare': {'pos': ['df', 'nonc'], 'opt': []},
        'noncentral_f': {'pos': ['dfnum', 'dfden', 'nonc'], 'opt': []},
        'normal': {'pos': [], 'opt': ['loc', 'scale']},
        'pareto': {'pos': ['a'], 'opt': []},
        'poisson': {'pos': [], 'opt': ['lam']},
        'power': {'pos': ['a'], 'opt': []},
        'rayleigh': {'pos': [], 'opt': ['scale']},
        'standard_cauchy': {'pos': [], 'opt': []},
        'standard_exponential': {'pos': [], 'opt': []},
        'standard_gamma': {'pos': ['shape'], 'opt': []},
        'standard_normal': {'pos': [], 'opt': []},
        'standard_t': {'pos': ['df'], 'opt': []},
        'triangular': {'pos': ['left', 'mode', 'right'], 'opt': []},
        'uniform': {'pos': [], 'opt': ['low', 'high']},
        'vonmises': {'pos': ['mu', 'kappa'], 'opt': []},
        'wald': {'pos': ['mean', 'scale'], 'opt': []},
        'weibull': {'pos': ['a'], 'opt': []},
        'zipf': {'pos': ['a'], 'opt': []},
}


class Distribution(object):
    def __init__(self, np_distribution, *args, **kwargs):

        # error checking ***********************************************************
        try:
            name = np_distribution.__name__
        except AttributeError:
            raise TypeError(
                    '{} is probably not a numpy distribution! See np.random.__doc__ for details.'.\
                    format(np_distribution))

        try:
            assert hasattr(np_random, name) and name in DIST_ARGS_DICT.keys()
        except AssertionError:
            raise NameError('Unknown numpy distribution: {}! See np.random.__doc__ for details.'.\
                    format(np_distribution))

        try:
            assert 'size' not in kwargs.keys()
        except AssertionError:
            raise KeyError('Argument \"size\" must not be provided during the initialization! ' +\
                    'Specify \"size\" when sampling random numbers.')

        for arg in kwargs.keys():
            try:
                assert arg in DIST_ARGS_DICT[name]['pos'] or arg in DIST_ARGS_DICT[name]['opt']
            except AssertionError:
                raise KeyError('np.random.{}(...) does not take keyword argument \"{}\"!'.\
                        format(name, arg))
        # error checking end *******************************************************

        self.name = name
        self.np_distribution = np_distribution
        self.args = args
        self.kwargs = kwargs

    def sample(self, size=None):
        self.kwargs['size'] = size
        return self.np_distribution(*self.args, **self.kwargs)
