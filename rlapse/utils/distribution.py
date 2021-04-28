from numpy import random as np_random

# all distributions
DISTRIBUTIONS_ALL = [l.split()[0] for l in np_random.__doc__.split('\n') \
        if 'distribution.' in l or 'distribution ' in l]

# See https://numpy.org/doc/1.16/reference/routines.random.html
# distributions that do not take positional arguments
DISTRIBUTIONS_0PARG = [
        'exponential',
        'gumbel',
        'laplace',
        'logistic',
        'lognormal',
        'normal',
        'poisson',
        'rayleigh',
        'standard_cauchy',
        'standard_exponential',
        'standard_normal',
        'uniform',
        ]

# distributions that take one positional argument
DISTRIBUTIONS_1PARG = [
        'chisquare',
        'dirichlet',
        'f',
        'logseries',
        'pareto',
        'power',
        'standard_gamma',
        'standard_t',
        'weibull',
        'zipf',
        ]

# distributions that take two positional arguments
DISTRIBUTIONS_2PARGS = [
        'beta',
        'binomial',
        'gamma',
        'geometric',
        'multinomial',
        'multivariate_normal',
        'negative_binomial',
        'noncentral_chisquare',
        'vonmises',
        'wald',
        ]

# distributions that take tree positional arguments
DISTRIBUTIONS_3PARGS = [
        'hypergeometric',
        'noncentral_f',
        'triangular',
        ]

assert sorted(DISTRIBUTIONS_ALL) == sorted(DISTRIBUTIONS_0PARG + DISTRIBUTIONS_1PARG +\
        DISTRIBUTIONS_2PARGS + DISTRIBUTIONS_3PARGS)


class Distribution(object):
    def __init__(self, np_distribution, *args, **kwargs):

        # error checking ***********************************************************
        try:
            name = np_distribution.__name__
        except AttributeError as e:
            print('{} is probably not a numpy distribution! See np.random.__doc__ for details.'.\
                    format(np_distribution))
            raise e

        try:
            assert hasattr(np_random, name) and name in DISTRIBUTIONS_ALL
        except AssertionError as e:
            print('Unknown numpy distribution: {}! See np.random.__doc__ for details.'.\
                    format(np_distribution))
            raise e

        if name in DISTRIBUTIONS_0PARG:
            try:
                assert len(args) == 0
            except AssertionError as e:
                print('{} distribution takes no positional arguments! Please use keyword arguments'.\
                        format(name))
                raise e
        elif name in DISTRIBUTIONS_1PARG:
            try:
                assert len(args) == 1
            except AssertionError as e:
                print('Please provide one positional argument to {} distribution!'.format(name))
                raise e
        elif name in DISTRIBUTIONS_2PARGS:
            try:
                assert len(args) == 0
            except AssertionError as e:
                print('Please provide two positional arguments to {} distribution!'.format(name))
                raise e
        else:
            try:
                assert len(args) == 0
            except AssertionError as e:
                print('Please provide three positional arguments to {} distribution!'.format(name))
                raise e

        try:
            assert 'size' not in kwargs.keys()
        except AssertionError as e:
            print('Argument \"size\" must not be provided during the initialization! ' +\
                    'Specify \"size\" when sampling random numbers.')
            raise e
        # error checking end *******************************************************

        self.name = name
        self.np_distribution = np_distribution
        self.args = args
        self.kwargs = kwargs

    def sample(self, size=None):
        self.kwargs['size'] = size
        return self.np_distribution(*self.args, **self.kwargs)
