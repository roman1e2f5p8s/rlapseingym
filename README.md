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

![equation](https://latex.codecogs.com/gif.latex?R%28%5Ccdot%7C%5Ccdot%2C2%29%3D%5Cbegin%7Bbmatrix%7D%20%5C1%20-%20%5Cepsilon%20%26%20%5Cepsilon%20%5C%5C%20%5C1%20-%20%5Cepsilon%20%26%20%5Cepsilon%20%5Cend%7Bbmatrix%7D.)

The rewards for each state-action pair are 

![equation](https://latex.codecogs.com/svg.latex?R%28s%2Ca%29%3D%5Cbegin%7Bbmatrix%7D%20r_1%20%26%20r_3%20%5C%5C%20r_2%20%26%20r_3%20%5Cend%7Bbmatrix%7D.)

The analysis provided in 
[(Epperlein et al. 2021)](https://www.researchgate.net/publication/350087449_Reinforcement_Learning_with_Algorithms_from_Probabilistic_Structure_Estimation)
can be used to show that for 
![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon%20%3E%20%5Cfrac%7Br_1%20&plus;%20r_3%20-%202r_2%7D%7B2%28r_1%20-%20r_2%29%7D)
the myopic policy is optimal, and thus a simple lightweight algorithm would be more suitable for this 
problem. However, the value of 
![equation](https://latex.codecogs.com/gif.latex?%5Cepsilon) might be unknown from the outset, and 
sometimes more complicated algorithms can be required. RLAPSE is a tool which automatically selects 
the appropriate algorithm given an unknown environment.

**Note:** even though the restaurant example is not practical, this toy example is intended to
be a good illustration of the problem settings. A more realistic and complicated scenarios, the broker
example, is discussed in 
[(Epperlein et al.
2021)](https://www.researchgate.net/publication/350087449_Reinforcement_Learning_with_Algorithms_from_Probabilistic_Structure_Estimation).

## Getting Started
Please follow these instructions to install all the requirements and use the package correctly.

### Requirements and Installation
```bash
git clone https://github.com/roman1e2f5p8s/rlapseingym
pip3.6 install -r requirements.txt
python3.6 setup.py install
```

You may want to run the tests to make sure the software is fully compatible with your system:
```bash
python3.6 setup.py pytest
```
### Usage
RLAPSE includes three packages:
* `mdps`: provides wrappers for some MDP examples with ability to convert them into OpenAI environments;
* `algorithms`: implements reinforcement learning algorithms and the orchestrator;
* `utils`: includes some utilities to compute properties of infinite horizon MDPs and wrapper for 
[NumPy distributions](https://numpy.org/doc/1.16/reference/routines.random.html).

#### MDP examples
The following MDP examples are available in `rlapse.mdps.mdp`:
* `MDP`: create MDPs from transition probability tensor and reward matrix;
* `RandomMDP`: generate random MDPs using NumPy distributions.
* `RestaurantMDP`: create MDPs for the restaurant example;
* `BrokerMDP`: generate random MDPs for the broker example;
* `ToyBrokerMDP`: create MDPs for a toy broker example.

```python
from rlapse.mdps.mdp import RestaurantMDP

mdp = RestaurantMDP(epsilon=0.2)
mdp.validate()
```

#### Gym environments
Generated MDPs can be easily converted to OpenAI Gym environments:
```python
env = mdp.to_env()
```

The environment then can be rendered into Jupyter notebooks:
```python
env.render()
```

<div>
<img src="examples/restaurant/_render.png" alt="Restaurant example MDP" />
</div>
<b>Figure 1: Restaurant example MDP rendered into Jupyter notebook</b>

## Built With
* [MDP for OpenAI Gym](https://github.com/BlackHC/mdp)

## Authors
* **Roman Overko**

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would 
like to change.

Please make sure to update tests as appropriate.

## License
None
