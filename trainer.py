#!sandbox/bin/python3

from datetime import datetime
import pickle

import numpy
import torch
from torch.optim import Adam

from config import (
    games_per_training_session, games_per_network_evaluation,
    games_per_trainset_creation, number_of_episodes, cuda_available,
    get_net_parameters_from_file)
from utils import logger, load_net_parameters, save_net_parameters
from nn_games.tic_tac_toe import Game, Net
from mcts import MCTS


class Trainer():

    def __init__(self):
        self.net = load_net_parameters()
        if cuda_available:
            self.net.cuda()
        self.optimizer = Adam(self.net.parameters(), lr=1e-2)

    def run(self):
        for episode in range(number_of_episodes):
            start = datetime.today()
            logger.info(u"Episode %s starting at %s", episode, start)
            trainset = self.create_trainset()
            self.train(trainset)
            self.evaluate()
            end = datetime.today()
            logger.info(
                u"Episode %s ended. Time elapsed %s", episode, (end - start))

    def create_trainset(self):
        """Creates a neural network that plays games against itself.
        During games, it generates States that will be used to
        train the network.
        """
        start = datetime.today()
        logger.info(u"\tTrainset creation started at %s", start)
        states = []
        for __ in range(games_per_trainset_creation):
            game = Game()
            while not game.game_is_over:
                tree = MCTS(game, self.net)
                game.move(tree.chosen_move)
                states.extend(tree.dataset)
        states = numpy.array(states)
        indexes = numpy.random.choice(
            len(states), games_per_training_session, replace=False)
        end = datetime.today()
        logger.info(
            u"\tTrainset creation ended. Time elapsed %s", (end - start))
        return states[indexes].tolist()

    def train(self, trainset):
        """Prepare three stacks of boards, pi values and rewards.
        Then passes boards stack to the network so it can predict
        pis and rewards.
        Finally, compare both pis and rewards with predicted ones and
        make a gradient descent over network in order to actualize
        parameters.
        """
        start = datetime.today()
        logger.info(u"\tTraining phase started at %s", start)
        boards = [state.game.get_board() for state in trainset]
        boards = torch.Tensor(boards)
        pis = torch.Tensor([state.uct for state in trainset])
        rewards = torch.Tensor([state.reward for state in trainset])
        if cuda_available:
            boards.cuda()
            pis.cuda()
            rewards.cuda()
        predicted_pis, predicted_rewards = self.net.forward(boards)
        loss = self.net.compute_loss(
            predicted_pis, pis, predicted_rewards, rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        end = datetime.today()
        logger.info(
            u"\tTraining phase ended. Time elapsed %s", (end - start))

    def evaluate(self):
        start = datetime.today()
        logger.info(u"\tEvaluation phase started at %s", start)
        for __ in range(games_per_network_evaluation):
            game = Game()
            while not game.game_is_over:
                tree = MCTS(game, self.net)
                game.move(tree.chosen_move)
        end = datetime.today()
        logger.info(
            u"\tEvaluation phase ended. Time elapsed %s", (end - start))
        pass

    def save_net_parameters(self):
        with open("nets/%s-net" % Game.name, "wb") as f:
            pickle.dump(self.net, f)

    def load_net_parameters(self):
        if get_net_parameters_from_file:
            try:
                f = open("nets/%s-net" % Game.name, "rb")
                net = pickle.load(f)
            except OSError:
                net = Net()
        else:
            net = Net()
        return net


if __name__ == "__main__":
    t = Trainer()
    t.run()
    save_net_parameters(t.net)
