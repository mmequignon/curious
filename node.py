#!/home/pytorch/pytorch/sandbox/bin/python3

from math import log, sqrt

from config import C


class Node:
    def __init__(self, game, parent=None, action=None):
        self.parent = parent
        self.visit_count = 0
        self.game = game
        self.children = {}
        self.action = action
        self.reward = 0
        self.state_collection = []
        self.uct = 0

    def create_children(self):
        """Returns a list of node that are direct children of current node
        depending of its game's valid moves.
        """
        children = {}
        for action, is_valid in enumerate(self.game.valid_moves()):
            if is_valid:
                child_game = self.game.copy()
                child_game.move(action)
                child = Node(child_game, self, action)
                children[action] = child
        return children

    def compute_uct(self, pi):
        """Computes upper confidence bound for the current node."""
        exploitation_componant = self.reward / (self.visit_count + 1.0)
        num = log(self.parent.visit_count)
        den = self.visit_count + 1.0
        exploration_component = sqrt(num / den)
        return exploitation_componant + C * pi * exploration_component

    def compute_probas(self):
        """Compute normalized counts for each child of the current node."""
        probas = []
        for action, is_valid in enumerate(self.game.valid_moves()):
            child = self.children.get(action, None)
            if child:
                probas.append(
                    float(child.visit_count) / float(self.visit_count))
            else:
                probas.append(0)
        self.probas = probas
