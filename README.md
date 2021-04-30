# Reinforcement Learning with Algorithms from Probabilistic Structure Estimation (RLAPSE)

The RLAPSE framework provides the choice whether to use a lightweight myopic algorithm (e.g., Q-learning 
with discount factor 0) or more complicated (e.g., Q-learning with discount 
factor close to 1) reinforcement learning (RL) algorithm.
Based on collected statistics about the environment, the VSRL switches to a more appropriate algorithm, if deemed necessary.
rithm with (i) likelihood ratio (LR) orchestrator and (ii) myopic Q-learning/LinUCB and hyperopic Q-learning algorithms.

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
