#!/home/pytorch/pytorch/sandbox/bin/python3

from math import sqrt

from torch import cuda

C = sqrt(2)
games_per_training_session = 5
games_per_trainset_creation = 5
games_per_network_evaluation = 5
number_of_games_against_random_agent = 5
simulations_per_move = 10
number_of_episodes = 2
cuda_available = cuda.is_available()
get_net_parameters_from_file = False
