#!/home/pytorch/pytorch/sandbox/bin/python3

import torch
import numpy

from config import simulations_per_move
from node import Node


class State:
    def __init__(self, game, uct, reward):
        self.game = game
        self.uct = uct
        self.reward = reward


class MCTS:
    def __init__(self, game, net):
        self.net = net
        for __ in range(simulations_per_move):
            root_node = Node(game)
            self.expand(root_node)

    def expand(self, root_node):
        root_node.visit_count += 1
        if root_node.game.game_is_over:
            root_node.reward = -root_node.game.reward
        else:
            if not root_node.children():
                root_node.create_children()
            pis, v = self.net.forward(root_node.game.get_board())
            # hide pis that are non valid
            pis = [
                is_valid and pis[action] or 0.0 for
                action, is_valid in enumerate(self.game.valid_moves)]
            chosen_action = numpy.argmax(pis)
            chosen_node = root_node.children[chosen_action]
            self.expand(chosen_node)
            root_node.reward = -sum(
                map(lambda c: c.reward, root_node.children))
















































#  class MCTS:
#      def __init__(self, root_game, net):
#          self.explored_nodes = []
#          self.net = net
#          root_node = Node(root_game)
#          for __ in range(simulations_per_move):
#              self.expand(root_node)
#          print([child.action for child in root_node.children])
#          exit(0)
#          visit_counts = [child.visit_count for child in root_node.children]
#          probs = [count/sum(visit_counts) for count in visit_counts]
#          self.chosen_move = numpy.argmaxself.select_child_node(root_node).action
#          self.states = self.build_states()
#
#      def expand(self, root_node):
#          root_node.visit_count += 1
#          if root_node.game.game_is_over:
#              root_node.reward = -root_node.game.reward
#          else:
#              if not root_node.children:
#                  self.explored_nodes.append(root_node)
#                  root_node.children = root_node.create_children()
#                  root_node.board = root_node.game.get_board()
#              pis, v = self.net.forward(root_node.game.get_board())
#              print(pis)
#              pis = pis * torch.FloatTensor(root_node.game.valid_moves())
#              index_val = numpy.argmax(pis)
#              chosen_node = root_node.children[index_val]
#              self.expand(chosen_node)
#              root_node.reward = -sum(
#                  [node.reward for node in root_node.children])
#          if root_node.children and root_node.parent:
#              root_node.upper_configence_tree = (
#                  root_node.compute_upper_confidence_tree())
#
#      def select_child_node(self, root_node):
#          return max(root_node.children, key=lambda n: n.reward)
#
#      def build_states(self):
#          states = []
#          for node in self.explored_nodes:
#              if node.parent:
#                  states.append(State(
#                      game=node.game,
#                      uct=[
#                          child.upper_confidence_tree for
#                          child in node.children],
#                      reward=node.reward))
#          return states
