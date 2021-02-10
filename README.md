# Con-RL
> Constructivist Reinforcement Learning - lifelong state space self-adaptation through constructivism

## Overview
This project contains Constructivist Reinforcement Learning 2.0, a framework enabling agents to learn and continuously adapt their state space representations in environments with heterogeneous input data with different time availability and space granularity.
# Prerequisites
* Conda or Virtualenv

# Requirements
* Graph-Tool
* NumPy
* OpenAI Gym
* Matplotlib

## Installation
```sh
$ git clone https://github.com/nhabbash/con-rl
$ cd con-rl
$ conda env create -f .\environment.yml
```
## Structure

The repository is structured as follows:

- [`src`](src) contains:
    - [`.`](src/) holding the implementations of ConRL, ML-GNG, GNG-U, Q-Learning, Sarsa(Lambda) and visualization utilities.
    - [`train`](src/train) which holds the hyperparameter tuning configuration and script to execute a sweep on Weights and Biases.
- [`notebooks`](noetbooks) containing the interactive notebooks used for experimentation with the framework in different environments and conditions.
- [`envs`](envs) containing a variety of basic environments.

# Notes

## Authors
* **Nassim Habbash** - [nhabbash](https://github.com/nhabbash)