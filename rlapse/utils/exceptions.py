class LessThanTwo(Exception):
    '''
    To prevent passing number < 2
    '''
    pass


class BadArrayShape(Exception):
    '''
    To prevent arrays of wrong sizes
    '''
    pass


class NotProbabilityN(Exception):
    '''
    To prevent generating negative values for probabilities
    '''
    pass


class NotProbabilityP(Exception):
    '''
    To prevent generating values > 1 for probabilities
    '''
    pass


class NotRowStochastic(Exception):
    '''
    To prevent generating non-stochastic rows
    '''
    pass
