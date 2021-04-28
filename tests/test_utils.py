import numpy as np

from rlapse.utils.distribution import DISTRIBUTIONS_0PARG
from rlapse.utils.distribution import Distribution


def test_distributions():
    for dist_name in DISTRIBUTIONS_0PARG:
        np_distribution = getattr(np.random, dist_name)
        d = Distribution(np_distribution)
        d.sample(size=10)
