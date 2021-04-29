import numpy as np
from scipy.stats import truncnorm
from itertools import product as Itertools_product


class _BrokerBase(object):
    '''
    Base class for the broker example
    Required arguments:
        - n_suppliers -- number of suppliers, int, n_suppliers >= 2
        - n_prices -- number of price categories for a commodity, int, n_prices >= 2
    Optional arguments:
        - epsilon -- epsilon parameter for the transition probabilities, float, 0 <= epsilon < 1,
            defaults to 0.0
        - best_suppliers -- list of best ("good") suppliers separated by comma, list, defaults to [0]
        - mu_best_supp -- mean of normal distribution for generating rewards of choosing best
            ("good") suppliers, float, -inf < mu_best_supp < inf, defaults to 0.90
        - sigma_best_supp -- variance of normal distribution for generating rewards of choosing best
            ("good") suppliers, float, sigma_best_supp > 0, defaults to 0.1
        - lower_lim_best_supp -- lower limit of truncated normal distribution for generating rewards
            of choosing best ("good") suppliers, float, -inf < lower_lim_best_supp < inf, defaults to 0.5
        - upper_lim_best_supp -- upper limit of truncated normal distribution for generating rewards
            of choosing best ("good") suppliers, float, -inf < upper_lim_best_supp < inf, defaults to 1.0
        - mu_norm_supp -- mean of normal distribution for generating rewards of choosing normal
            ("mediocre") suppliers, float, -inf < mu_norm_supp < inf, defaults to 0.495
        - sigma_norm_supp -- variance of normal distribution for generating rewards of choosing normal
            ("mediocre") suppliers, float, sigma_norm_supp > 0, defaults to 0.1
        - lower_lim_norm_supp -- lower limit of truncated normal distribution for generating rewards
            of choosing normal ("mediocre") suppliers, float, -inf < lower_lim_norm_supp < inf,
            defaults to 0.0
        - upper_lim_norm_supp -- upper limit of truncated normal distribution for generating rewards
            of choosing normal ("mediocre") suppliers, float, -inf < upper_lim_norm_supp < inf,
            defaults to 0.5
        - max_rew_best_supp -- maximum value of reward of choosing best ("good") suppliers, float,
            -inf < max_rew_best_supp < inf, defaults to 1.0
        - min_rew_best_supp -- minimum value of reward of choosing best ("good") suppliers, float,
            -inf < min_rew_best_supp < inf, defaults to 0.455
        - uniform_pq -- generate p* and q* using uniform distribution. If False, the truncated normal
            distribution will be used, bool, defaults to False
        - lower_lim_pp -- lower limit of truncated normal distribution for generating p_+, float,
            0 <= lower_lim_pp <= 1, defaults to None. If None, then value of epsilon is used
        - upper_lim_pp -- upper limit of truncated normal distribution for generating p_+, float,
            0 <= upper_lim_pp <= 1, defaults to 1.0
        - mu_pp -- mean of normal distribution for generating p_+, float, -inf < mu_pp < inf,
            defaults to 0.7
        - sigma_pp -- variance of normal distribution for generating p_+, float, sigma_pp > 0,
            defaults to 0.1
        - lower_lim_pm -- lower limit of truncated normal distribution for generating p_-, float,
            0 <= lower_lim_pm <= 1, defaults to 0.0
        - upper_lim_pm -- upper limit of truncated normal distribution for generating p_-, float,
            0 <= upper_lim_pm <= 1, defaults to 1.0,
        - mu_pm -- mean of normal distribution for generating p_-, float, -inf < mu_pp < inf,
            defaults to 0.3
        - sigma_pm -- variance of normal distribution for generating p_-, float, sigma_pm > 0,
            defaults to 0.1
        - toy -- build only a toy broker example with 2 suppliers and 2 price categories,
            bool, defaults to False
        - toy_controlled -- build controlled MDP toy broker example, bool, defaults to False
    '''
    def __init__(
            self,
            n_suppliers,
            n_prices,
            epsilon=0.0,
            best_suppliers=[0],
            mu_best_supp=0.90,
            sigma_best_supp=0.1,
            lower_lim_best_supp=0.5,
            upper_lim_best_supp=1.0,
            mu_norm_supp=0.495,
            sigma_norm_supp=0.1,
            lower_lim_norm_supp=0.0,
            upper_lim_norm_supp=0.5,
            max_rew_best_supp=1.0,
            min_rew_best_supp=0.455,
            uniform_pq=False,
            lower_lim_pp=None,
            upper_lim_pp=1.0,
            mu_pp=0.7,
            sigma_pp=0.1,
            lower_lim_pm=0.0,
            upper_lim_pm=1.0,
            mu_pm=0.3,
            sigma_pm=0.1,
            toy=False,
            toy_controlled=False,
            ):
        '''
        Initialization
        '''

        self.n_suppliers = n_suppliers
        self.n_prices = n_prices
        self.n_states = pow(n_prices, n_suppliers)
        self.epsilon = epsilon
        self.best_suppliers = best_suppliers
        self.mu_best_supp = mu_best_supp
        self.sigma_best_supp = sigma_best_supp
        self.lower_lim_best_supp = lower_lim_best_supp
        self.upper_lim_best_supp = upper_lim_best_supp
        self.mu_norm_supp = mu_norm_supp
        self.sigma_norm_supp = sigma_norm_supp
        self.lower_lim_norm_supp = lower_lim_norm_supp
        self.upper_lim_norm_supp = upper_lim_norm_supp
        self.max_rew_best_supp = max_rew_best_supp
        self.min_rew_best_supp = min_rew_best_supp
        self.uniform_pq = uniform_pq
        self.lower_lim_pp = epsilon if lower_lim_pp is None else lower_lim_pp
        self.upper_lim_pp = upper_lim_pp
        self.mu_pp = mu_pp
        self.sigma_pp = sigma_pp
        self.lower_lim_pm = lower_lim_pm
        self.upper_lim_pm = upper_lim_pm
        self.mu_pm = mu_pm
        self.sigma_pm = sigma_pm
        self.toy = toy
        self.toy_controlled = toy_controlled

        self.state_context_dict = self._get_state_context_vec_dict()
        self.p_plus, self.p_minus, self.q_plus, self.q_minus = self._generate_pq()
        self.P = self._get_transition_tensor()
        self.R = self._get_reward_matrix()


    def _get_all_contexts(self):
        '''
        Finds all context vectors
        Parameters:
            - no input parameters
        Returns:
            - the Cartesian product of price categories, list
        '''

        prices = [x for x in range(self.n_prices)]

        return list(Itertools_product(prices, repeat=self.n_suppliers))


    def _get_state_index(self, context_vec):
        '''
        Computes the index of state/context
        Parameters:
            - context_vec -- context vector, np.ndarray(n_suppliers)
        Returns:
            - state index, int
        '''

        return int(sum([b * pow(self.n_prices, n) for n, b in enumerate(context_vec)]))


    def _get_state_context_vec_dict(self):
        '''
        Returns a sorted dictionary such that its keys correspond to a state index, and the values
            are context vectors
        Parameters:
            - no input parameters
        Returns:
            - state-context vector dictionary, dict
        '''

        d = {}
        all_contexts = self._get_all_contexts()
        for context in all_contexts:
            state = self._get_state_index(context)
            d[state] = np.array(context, dtype=int)
        assert len(d) == self.n_states

        return dict(sorted(d.items()))


    def _generate_pq(self):
        '''
        Generates p_* and q_* using either uniform or truncated normal distribution
        Parameters:
            - no input parameters
        Returns:
            - p_+, p_-, q_+, q_-, all are np.ndarray(n_suppliers)
        '''

        if self.toy:
            pp = np.ones(shape=self.n_suppliers) * 0.5
            qp = np.ones(shape=self.n_suppliers) * 0.5
            pm = np.ones(shape=self.n_suppliers) * 0.5
            if not self.toy_controlled:
                qm = np.ones(shape=self.n_suppliers) * 0.5
            else:
                qm = np.array([0.8, 0.2])
            return pp, pm, qp, qm

        if self.uniform_pq:
            pp = np.random.uniform(low=self.epsilon, high=1.0, size=self.n_suppliers)
        else:
            Xp = truncnorm(
                    (self.lower_lim_pp - self.mu_pp) / self.sigma_pp,
                    (self.upper_lim_pp - self.mu_pp) / self.sigma_pp,
                    loc=self.mu_pp,
                    scale=self.sigma_pp,
                    )
            pp = Xp.rvs(size=self.n_suppliers)

        qp = pp - self.epsilon
        assert (qp >= 0).all()

        if self.uniform_pq:
            pm = np.random.uniform(low=0.0, high=1.0, size=self.n_suppliers)
        else:
            Xm = truncnorm(
                    (self.lower_lim_pm - self.mu_pm) / self.sigma_pm,
                    (self.upper_lim_pm - self.mu_pm) / self.sigma_pm,
                    loc=self.mu_pm,
                    scale=self.sigma_pm,
                    )
            pm = Xm.rvs(size=self.n_suppliers)
        for i, x in enumerate(pp):
            while pm[i] + x > 1.0:
                if self.uniform_pq:
                    pm[i] = np.random.uniform(low=0.0, high=1.0, size=None)
                else:
                    pm[i] = Xm.rvs(size=None)
        assert ((pp + pm) <= 1.0).all()

        qm = pm + self.epsilon
        assert ((qp + qm) <= 1.0).all()

        return pp, pm, qp, qm


    def _is_transition_possible(self, diff):
        '''
        Verifies whether transition between two states is possible
        Parameters:
            - diff -- difference between two states, np.ndarray(n_suppliers)
        Returns:
            - True if transition possible, else False
        '''

        return (diff.max() <= 1 and diff.min() >=-1)


    def _get_mult_factor(self, pq_plus, pq_minus, d, i, s):
        '''
        Computes multiplication factor for transition probabilities
        Parameters:
            - pq_plus -- p_+ or q_+, np.ndarray(n_suppliers)
            - pq_minus -- p_- or q_-, np.ndarray(n_suppliers)
            - d -- difference in price categories for supplier i, int
            - i -- supplier, int
            - s -- state index for which transition probabilities are being computed, int
        Returns:
            - x -- multiplication factor, float
        '''

        if d == 1:
            x = pq_plus[i]
        elif d == -1:
            x = pq_minus[i]
        else:
            pq_p = 0 if self.state_context_dict[s][i] == self.n_prices - 1 else pq_plus[i]
            pq_m = 0 if self.state_context_dict[s][i] == 0 else pq_minus[i]
            x = (1 - pq_p - pq_m)

        return x


    def _get_stochastic_row(self, s, a):
        '''
        Computes stochastic row for state s and action a
        Parameters:
            - s -- state, int
            - a -- action, int
        Returns:
            - row -- stochastic row, np.ndarray(n_states)
        '''

        row = np.zeros(self.n_states)

        for sn in range(self.n_states):
            diff = self.state_context_dict[sn] - self.state_context_dict[s]
            if self._is_transition_possible(diff):
                result = 1.0
                for i, d in enumerate(diff):
                    if not i == a:
                        mf = self._get_mult_factor(self.q_plus, self.q_minus, d, i, s)
                    else:
                        mf = self._get_mult_factor(self.p_plus, self.p_minus, d, i, s)
                    result *= mf
                row[sn] = result

        return row


    def _get_transition_tensor(self):
        '''
        Computes the tensor of transition probabilities
        Parameters:
            - no input parameters
        Returns:
            - tp -- tensor of transition probabilities, np.ndarray((n_suppliers, n_states,
                n_states))
        '''

        tp = np.zeros(shape=(self.n_suppliers, self.n_states, self.n_states))

        for a in range(self.n_suppliers):
            for s in range(self.n_states):
                tp[a, s] = self._get_stochastic_row(s, a)

        return tp


    def check_P(self, verbose=False):
        '''
        Verifies whether all the rows of transition tensor are stochastic.
        Raises an AssertionError if one of them is not stochastic
        Optional parameters:
            - verbose -- notify about success by printing, bool (defaults to False)
        Returns:
            - None
        '''

        for a in range(self.n_suppliers):
            for s in range(self.n_states):
                assert np.isclose(np.sum(self.P[a, s]), 1)
        if verbose:
            print('Success! All the stochastic rows sum up to 1.')


    def check_open_loop_mdp(self):
        '''
        Checks whether MDP is open-loop
        Parameters:
            - no input parameters
        Returns:
            - None
        '''

        for a in range(self.n_suppliers):
            if not np.isclose(self.P[0], self.P[a]).all():
                print('This is a closed-loop MDP.')
                return
        print('This is an open-loop MDP.')


    def _get_reward_matrix(self):
        '''
        Generates the reward matrix using truncated normal distribution
        Parameters:
            - no input parameters
        Returns:
            - r -- reward matrix, np.ndarray((n_states, n_suppliers))
        '''

        r = np.zeros(shape=(self.n_states, self.n_suppliers))

        if self.toy:
            r[0, 0] = r[2, 0] = 15
            r[1, 0] = r[3, 0] = 2
            r[:, 1] = 1
            return r

        for a in range(self.n_suppliers):
            if a in self.best_suppliers:
                X = truncnorm(
                        (self.lower_lim_best_supp - self.mu_best_supp) / self.sigma_best_supp,
                        (self.upper_lim_best_supp - self.mu_best_supp) / self.sigma_best_supp,
                        loc=self.mu_best_supp,
                        scale=self.sigma_best_supp,
                        )
                x = X.rvs(size=self.n_prices)
                x[::-1].sort()
                x[0] = self.max_rew_best_supp
                x[-1] = self.min_rew_best_supp
            else:
                X = truncnorm(
                        (self.lower_lim_norm_supp - self.mu_norm_supp) / self.sigma_norm_supp,
                        (self.upper_lim_norm_supp - self.mu_norm_supp) / self.sigma_norm_supp,
                        loc=self.mu_norm_supp,
                        scale=self.sigma_norm_supp,
                        )
                x = X.rvs(size=self.n_prices)
                x[::-1].sort()
            for s in range(self.n_states):
                r[s, a] = x[self.state_context_dict[s][a]]

        return r
