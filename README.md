# Reinforcement Learning with Algorithms from Probabilistic Structure Estimation (RLAPSE)

In many reinforcement learning settings, the choice of the underlying algorithm is not obvious.
The RLAPSE framework provides the choice whether to use a lightweight myopic algorithm (e.g., Q-learning 
with discount factor 0) or more complicated (e.g., Q-learning with discount 
factor close to 1) reinforcement learning algorithm. The framework utilizes 
(i) the likelihood ratio test, and (ii) a variant of Q-learning analyzed by 
[(Even-Dar et al. 2003)](https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf)). 
Based on the collected statistics about the environment, the RLAPSE orchestrator switches to a more 
appropriate algorithm, if deemed necessary. This selection algorithm is called the *orchestrator*.
More details about this work can be found in [(Epperlein et al. 2021)](https://www.researchgate.net/publication/350087449_Reinforcement_Learning_with_Algorithms_from_Probabilistic_Structure_Estimation).

## Motivation

Consider a simple illustrative example for restaurant recommendation.
We have a good restaurant (*GR*) with very limited capacity, and a bad restaurant (*BR*) with near 
infinite capacity. Every time a user asks for a recommendation, we can send them to either of the 
two restaurants, hence our action space is 
![equation](https://latex.codecogs.com/svg.latex?A%20%3D%20%5C%7B1%2C%202%5C%7D), where 
* Action 1: Send user to *GR*;
* Action 2: Send user to *BR*.

*BR* is always able to seat customers, whereas *GR* might not be able to, hence we need at least 
two states, so the state space is 
![equation](https://latex.codecogs.com/svg.latex?S%20%3D%20%5C%7B1%2C%202%5C%7D), where 
* State 1: There is no wait in *GR*;
* State 2: There is a wait in *GR*.

Even when they have to wait, customers still prefer *GR*. Customers' enjoyment of *BR* is always 
the same, since they never have to wait there. So we can say that we have rewards 
![equation](https://latex.codecogs.com/svg.latex?r_1%20%3E%20r_2%20%3E%20r_3), where
* ![equation](https://latex.codecogs.com/svg.latex?r_1): 
reward for sending customer to *GR* while there is no wait;
* ![equation](https://latex.codecogs.com/svg.latex?r_2): 
reward for sending customer to *GR* even if there is a wait;
* ![equation](https://latex.codecogs.com/svg.latex?r_3): 
reward for sending customer to *BR*.

Sending a customer to *GR* while it is able to seat will, with high probability, say 
![equation](https://latex.codecogs.com/svg.latex?1%20-%20%5Cepsilon_%7B11%7D), lead to 
a crowded *GR*. If there was a wait already, then sending a customer there will also likely 
not change that, so say the probability of *GR* staying crowded is 
![equation](https://latex.codecogs.com/svg.latex?1%20-%20%5Cepsilon_%7B21%7D).
Not sending a customer there will likely leave *GR* without a wait, with probability 
![equation](https://latex.codecogs.com/svg.latex?1%20-%20%5Cepsilon_%7B12%7D), or lead to 
*GR* being able to seat the next customer, with probability 
![equation](https://latex.codecogs.com/svg.latex?1%20-%20%5Cepsilon_%7B22%7D). 
For simplicity and illustrative purposes, we assume now that all the small probabilities are equal, i.e.,
![equation](https://latex.codecogs.com/svg.latex?1%20%3E%20%5Cepsilon_%7B11%7D%20%3D%20%5Cepsilon_%7B12%7D%20%3D%20%5Cepsilon_%7B21%7D%20%3D%20%5Cepsilon_%7B22%7D%20%3D%20%5Cepsilon%20%3E%200).

This can be modeled as an MDP with transition probabilities 
![equation](https://latex.codecogs.com/svg.latex?P) and reward 
![equation](https://latex.codecogs.com/svg.latex?R). The probabilities to move from state 
![equation](https://latex.codecogs.com/svg.latex?s) to state 
![equation](https://latex.codecogs.com/svg.latex?s%27), i.e., 
![equation](https://latex.codecogs.com/svg.latex?P%28s%27%7Cs%2Ca%29) 
for each action are encoded in the transition probability matrices:
* If recommending GR, i.e.,
![equation](https://latex.codecogs.com/svg.latex?a%20%3D%201), then transition matrix

![equation](https://latex.codecogs.com/svg.latex?P%28%5Ccdot%7C%5Ccdot%2C1%29%3D%5Cbegin%7Bbmatrix%7D%20%5Cepsilon%20%26%201%20-%20%5Cepsilon%20%5C%5C%20%5Cepsilon%20%26%201%20-%20%5Cepsilon%20%5Cend%7Bbmatrix%7D.)

* If recommending BR, i.e.,
![equation](https://latex.codecogs.com/svg.latex?a%20%3D%202), then transition matrix

![equation](https://latex.codecogs.com/svg.latex?R%28s%2Ca%29%3D%5Cbegin%7Bbmatrix%7D%20%5C1%20-%20%5Cepsilon%20%26%20%5Cepsilon%20%5C%5C%20%5C1%20-%20%5Cepsilon%20%26%20%5Cepsilon%20%5Cend%7Bbmatrix%7D.)

The rewards for each state-action pair are 

![equation](https://latex.codecogs.com/svg.latex?R%28s%2Ca%29%3D%5Cbegin%7Bbmatrix%7D%20r_1%20%26%20r_3%20%5C%5C%20r_2%20%26%20r_3%20%5Cend%7Bbmatrix%7D.)

![equation]()
## Installation
The `VSRL` project requires Python 3.8.1 or higher. To install requirements:
```bash
pip install -r requirements.txt
``` 

## Usage
- to run a simulation:
```bash
python3 run.py --tp_type {1,2,3,4} --example {1,2,...} [-h] [--re_begin {1,2,...}]
              [--re_end {1,2,...}] [--S {1,2,...}] [--A {1,2,...}] [--T {1,2,...}]
              [--l_alpha >1] [--q_gamma [0-1]] [--q_epsilon [0-1]]
              [--q_epsilon_decay_rate [0-1]] [--q_epsilon_decay_interval [0-1]]
              [--q_omega 0.5-1] [--s_level [0-1]] [--T_start <=T] [--use_linucb]
              [--start_with_complicated] [--restaurant] [--eps_rest [0-1]] [--mp]
              [--seed {0,1,2,...}] [--no_time_warn]
```
- to plot the data:
```bash
python3 plot.py --tp_type {1,2,3,4} --example {1,2,...} --S {1,2,...} --A {1,2,...} --T
               {1,2,...} [-h] [--re_begin {1,2,...}] [--re_end {1,2,...}] [--use_linucb]
```
- to plot all the data for a given type of environment
```bash
python3 plot_all.py --tp_type {1,2,3,4} --S {1,2,...} --A {1,2,...} --T {1,2,...} [-h]
                   [--ex_numb {1,2,...}] [--re_begin {1,2,...}] [--re_end {1,2,...}]
                   [--use_linucb]
```
- to compute the average counts(number of switches) for a given type of environment
```bash
python3 avg_counts.py --tp_type {1,2,3,4} --S {1,2,...} --A {1,2,...} --T {1,2,...} --T_start
                     {1,2,...} [-h] [--ex_numb {1,2,...}] [--re_begin {1,2,...}]
                     [--re_end {1,2,...}]
```
- to plot type 2 errors
```bash
python3 plot_t2err.py --S {1,2,...} --A {1,2,...} --T {1,2,...} [-h] [--ex_numb {1,2,...}]
                     [--re_begin {1,2,...}] [--re_end {1,2,...}]
```
- to plot the data for the restaurant example:
```bash
python3 plot_rest.py --example {1,2,...} --T {1,2,...} [-h] [--re_begin {1,2,...}]
                    [--re_end {1,2,...}] [--use_linucb]
```
- to plot type 2 errors for the restaurant example
```bash
python3 plot_t2err_rest.py --example {1,2,...} --T {1,2,...} [-h] [--re_begin {1,2,...}]
                          [--re_end {1,2,...}]
```

Please note that a single simulation for a large number of states may take a long time. You may want to speed up the simulation using `--mp` option.

See help files for more details:
```bash
python3 SCRIPTNAME.py -h
```

## Authors
Anonymous Authors
Contact: anon.email@domain.com
