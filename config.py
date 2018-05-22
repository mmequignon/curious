#!/home/pytorch/pytorch/sandbox/bin/python3

from torch import cuda

C = 0.5
games_per_training_session = 50
games_per_trainset_creation = 10
games_per_network_evaluation = 50
simulations_per_move = 25
number_of_episodes = 100
cuda_available = False  # cuda.is_available()
