'''
Contains the implementation of logarithm of likelihood and CDF

Created on June 10, 2020

@authors: anonymous authors

Contact: anon.email@domain.com
'''

import numpy as np
from scipy.special import gammainc


def ln_l0(m, n_prime):
    '''
    Computes logarithm of the maximum likelihood of model M0 (i.e. open-loop assumption)
    Parameters:
        - m -- array of transition counts m(c',c,a), np.ndarray((A,S,S), dtype=int)
        - n_prime -- array of transition counts n'(c), np.ndarray(S, dtype=int)
    Returns:
        - lnl0 -- logarithm of the maximum likelihood l0, float
    '''

    A, S, _ = m.shape  # number of actions and states

    lnl0 = - np.log(S)
    for r in range(S):
        for c in range(S):
            m_prime = 0
            for a in range(A):
                m_prime += m[a, r, c]
            if n_prime[r] > 0 and m_prime > 0:
                lnl0 += m_prime * (np.log(m_prime) - np.log(n_prime[r]))

    return lnl0


def ln_l0_(m_prime, n_prime, n_states):
    '''
    Computes logarithm of the maximum likelihood of model M0 (i.e. open-loop assumption)
    Parameters:
        - m_prime -- array of transition counts m'(c',c), np.ndarray((S,S), dtype=int)
        - n_prime -- array of transition counts n'(c), np.ndarray(S, dtype=int)
        - n_states -- number of states, int
    Returns:
        - lnl0 -- logarithm of the maximum likelihood l0, float
    '''

    lnl0 = sum([np.dot(m_prime[:, sp],
        np.log(m_prime[:, sp], where=(m_prime[:, sp] > 0)) - np.log(n_prime, where=(n_prime > 0))) \
        for sp in range(n_states)])
    # lnl0 = np.sum(m_prime @ (np.log(m_prime, where=(m_prime > 0)) - np.log(n_prime, where=(n_prime > 0))))

    return lnl0 - np.log(n_states)


def ln_l1(m, n):
    '''
    Computes logarithm of the maximum likelihood of model M1 (i.e. closed-loop assumption)
    Parameters:
        - m -- array of transition counts m(c',c,a), np.ndarray((A,S,S), dtype=int)
        - n -- array of transition counts n(c,a), np.ndarray((S,A), dtype=int)
    Returns:
        - lnl1 -- logarithm of the maximum likelihood l1, float
    '''
    S, A = n.shape # number of states and actions

    lnl1 = - np.log(S)
    for a in range(A):
        for r in range(S):
            for c in range(S):
                if n[r, a] > 0 and m[a, r, c] > 0:
                    lnl1 += m[a, r, c] * (np.log(m[a, r, c]) - np.log(n[r, a]))

    return lnl1


def cdf(L, k):
    '''
    Computes the CDF for Chi-squared distribution
    Parameters:
        - L -- likelihood ratio, float
        - k -- the difference in degrees of freedom, int
    Returns:
        - regularized lower incomplete gamma function Gamma(k/2, L/2), float
    '''

    return gammainc(k/2.0, L/2.0)
