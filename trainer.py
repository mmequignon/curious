#!/home/pytorch/pytorch/sandbox/bin/python3

import random
import re
import torch
import numpy
from tic_tac_toe import TicTacToe


class Trainer():

    def __init__(self):
        filename = "data/tic-tac-toe-dataset.txt"
        self.dataset = []
        with open(filename, "r") as f:
            for line in f:
                self.dataset.append(line.split())

    def get_leaves(self, parent):
        regex = re.compile("%s[0-9]+" % (parent))
        return [i for i in self.dataset if regex.match(i[0])]

    def best_branch(self, trunk, branches):
        """Depending on the parent given as argument, returns the best move.
        """
        # TODO: For the moment, this method do not evaluates losses
        # or nil options, in order to avoid a loss.
        childs = self.get_leaves(trunk)
        results = []
        for branch in branches:
            regex = re.compile("%s[0-9]*" % (trunk + str(branch)))
            leaves = [i for i in childs if regex.match(i[0])]
            # counters of victories for player one and two and for nil games
            one = two = nil = 0
            for leave in leaves:
                if leave[1].isnumeric():
                    if int(leave[1]) == 0:
                        one += 1
                    else:
                        two += 1
                else:
                    nil += 1
            # If the current player is One, we must maximze victories of One
            # else, Two.
            current = len(trunk) % 2 == 0 and one or two
            win = (current / len(leaves)) * 100
            results.append((win, branch))
        results.sort(reverse=True)
        return(int(results[0][-1]))

    def get_game_from_sequence(self, sequence):
        moves = [int(i) for i in sequence]
        game = TicTacToe()
        for move in moves:
            game.move(move)
            game.end_turn()
        return game

    def get_tensor_from_game(self, game):
        three_dim_array = [
            [[p[x + y] and 1 or 0 for y in range(3)] for
                x in range(0, 9, 3)] for p in game.table]
        n = numpy.array(three_dim_array)
        return torch.from_numpy(n)


if __name__ == "__main__":
    trainer = Trainer()
    game = TicTacToe()
    trunk = random.choice(trainer.dataset)[0][:3]
    game = trainer.get_game_from_sequence(trunk)
    print(trainer.get_tensor_from_game(game))
    print(game.representation())
