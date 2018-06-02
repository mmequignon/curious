#!/home/pytorch/pytorch/sandbox/bin/python3

import numpy
import torch

from config import cuda_available, simulations_per_move
from node import Node
from utils import logger


class State:
    def __init__(self, game, uct, reward):
        self.game = game
        self.uct = uct
        self.reward = reward


class MCTS:
    def __init__(self, game, net):
        self.net = net
        self.visited_nodes = []
        root_node = Node(game)
        for __ in range(simulations_per_move):
            self.expand(root_node)
        self.dataset = self.create_dataset()
        self.chosen_move = numpy.argmax(root_node.probas)

    def expand(self, root_node):
        """Does recursively a single exploration on the node passed as
        argument.
        """
        root_node.visit_count += 1
        if root_node.game.game_is_over:
            root_node.reward = -root_node.game.reward
        else:
            if not root_node.children:
                root_node.children = root_node.create_children()
                for child in root_node.children.values():
                    self.rollout(child)
                self.visited_nodes.append(root_node)
            board = torch.Tensor(root_node.game.get_board())
            if cuda_available:
                board.cuda()
            pis, v = self.net.forward(board)
            # hide pis that are non valid
            # pi * valid = 0 if non valid, pi else.
            pis = [
                pis[0][action].item() * is_valid for
                action, is_valid in enumerate(root_node.game.valid_moves())]
            # /!\ DEBUG
            for action, is_valid in enumerate(root_node.game.valid_moves()):
                if is_valid:
                    child = root_node.children[action]
                    child.uct = child.compute_uct(pis[action])
            # ↓ Once previous problem will be solved s/argmin/argmax ↓
            chosen_action = numpy.argmax(pis)
            try:
                chosen_node = root_node.children[chosen_action]
            except KeyError:
                logger.error(u"\t\tValid moves were all equal to 0.")
                chosen_node = numpy.random.choice(
                    list(root_node.children.values()))
            self.expand(chosen_node)
            root_node.reward = -sum(
                map(lambda c: c.reward, root_node.children.values()))

    def rollout(self, root_node):
        if root_node.game.game_is_over:
            root_node.reward = -root_node.game.reward
        else:
            children = root_node.create_children()
            chosen_node = numpy.random.choice(list(children.values()))
            self.rollout(chosen_node)
            root_node.reward = -chosen_node.reward

    def create_dataset(self):
        dataset = []
        for node in self.visited_nodes:
            node.compute_probas()
            state = State(node.game, node.probas, node.reward)
            dataset.append(state)
        return dataset
