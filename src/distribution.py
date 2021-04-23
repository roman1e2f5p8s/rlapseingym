from numpy import random as np_random

doc = np_random.__doc__.split('\n')
distributions = []
for line in doc:
    if 'distribution.' in line or 'distribution ' in line:
        distributions += [line.split()[0]]

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
            assert hasattr(np_random, name) and name in distributions
        except AssertionError as e:
            print('Unknown numpy distribution: {}! See np.random.__doc__ for details.'.\
                    format(np_distribution))
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
