from numpy import random as np_random

doc = np_random.__doc__.split('\n')
distributions = []
for line in doc:
    if 'distribution.' in line or 'distribution ' in line:
        distributions += [line.split()[0]]

class Distribution(object):
    def __init__(self, np_distribution, *args, **kwargs):
        name = np_distribution.__name__

        assert hasattr(np_random, name) and name in distributions
        assert 'size' not in kwargs.keys()

        self.name = name
        self.np_distribution = np_distribution
        self.args = args
        self.kwargs = kwargs

    def sample(self, size=None):
        self.kwargs['size'] = size
        return self.np_distribution(*self.args, **self.kwargs)
