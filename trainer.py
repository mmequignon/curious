#!sandbox/bin/python3

import pickle

import numpy
import torch

from config import (
    games_per_training_session, games_per_network_evaluation,
    games_per_trainset_creation, number_of_episodes, cuda_available)
from tic_tac_toe import Game, Net
from mcts import MCTS


class Trainer():

    def __init__(self):
        self.load_net_parameters()
        if cuda_available:
            self.net.cuda()

    def run(self):
        for i in range(number_of_episodes):
            trainset = self.create_trainset()
            self.train(trainset)
            self.evaluate()

    def create_trainset(self):
        states = []
        for __ in range(games_per_trainset_creation):
            game = Game()
            while not game.game_is_over:
                tree = MCTS(game, self.net)
                game.move(tree.chosen_move)
                states.extend(tree.states)
        states = numpy.array(states)
        indexes = numpy.random.choice(
            len(states), games_per_training_session, replace=False)
        return states[indexes].tolist()

    def train(self, trainset):
        """Prepare three stacks of boards, pi values and rewards.
        Then passes boards stack to the network so it can predict
        pis and rewards.
        Finally, compare both pis and rewards with predicted ones and
        make a gradient descent over network in order to actualize
        parameters.
        """
        if cuda_available:
            boards = torch.stack(
                [state.game.get_board() for state in trainset]).cuda()
            ucts = torch.cuda.FloatTensor([state.uct for state in trainset])
            rewards = torch.cuda.FloatTensor(
                [state.reward for state in trainset])
        else:
            boards = torch.stack(
                [state.game.get_board() for state in trainset])
            ucts = torch.FloatTensor([state.uct for state in trainset])
            rewards = torch.FloatTensor([state.reward for state in trainset])
        predicted_ucts, predicted_rewards = self.net.forward(boards)
        loss = self.get_loss(predicted_ucts, predicted_rewards, ucts, rewards)
        loss.backward()

    def get_loss(self, predicted_ucts, predicted_rewards, ucts, rewards):
        """Computes mean-squared loss for rewards and 1G"""
        size = predicted_ucts.size()[0]
        pi_loss = torch.sum(ucts * predicted_ucts) / size
        v_loss = torch.sum((rewards - predicted_rewards.view(-1))**2) / size
        return pi_loss - v_loss

    def evaluate(self):
        for __ in range(games_per_network_evaluation):
            game = Game()
            while not game.game_is_over:
                tree = MCTS(game, self.net)
                game.move(tree.chosen_move)
        pass

    def save(self):
        with open("nets/%s-net" % Game.name, "wb") as f:
            pickle.dump(self.net, f)

    def load_net_parameters(self):
        try:
            f = open("nets/%s-net" % Game.name, "rb")
            self.net = pickle.load(f)
        except OSError:
            self.net = Net(Game)


if __name__ == "__main__":
    t = Trainer()
    t.run()
    t.save()
