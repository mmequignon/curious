#!/home/pytorch/pytorch/sandbox/bin/python3

import copy
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
        self.ucb = 0

    def create_children(self):
        children = {}
        for action, is_valid in enumerate(self.game.valid_moves()):
            if is_valid:
                child_game = copy.deepcopy(self.game)
                child_game.move(action)
                child = Node(child_game, self, action)
                children[action] = child
        return children

    def compute_upper_confidence_tree(self):
        a = self.reward / self.parent.visit_count
        b = sqrt(log(self.parent.visit_count) / self.visit_count)
        return a + (b * C)
