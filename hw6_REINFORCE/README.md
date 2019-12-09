# Policy Gradient Reinforce Learning
We will be tackling this deep question by learning how to play Cartpole, using two RL algorithms: 

1) REINFORCE and 2) REINFORCE with Baseline.

In CartPole you are training a simple agent (a cart) to balance a pole. Hence, CartPole.

## Data
All data for this assignment will be collected interactively, via the CartPole-v1 environment of OpenAI Gym

OpenAI Gym provides an easy to use interface for AIs to play games.

You can give the gym environment an action, and it will 
1) update the state of the game,  

2) return you the next state, the reward of taking the action, and whether you arrived at a terminal state.

Also you'll need to install pkg:
`pip install gym` and `pip install matplotlib` 

## REINFORCE:
Run `python assignment.py REINFORCE`

My final average rewards can reach 380 after 650 episodes

## REINFORCE with baseline
Run `python assignment.py REINFORCE_BASELINE`

## Conclusion
We have provided the function, `visualize_data()`, which plots the reward through episodes.

In addition, if you would like to see your agent perform, call env.render() every time you perform an action. 

It is useful to see how your model is learning to perform a given task. BUT IT WILL BE MUCH SLOWER WHEN TRAINING. 

My final average rewards can reach 470 after 650 episodes
