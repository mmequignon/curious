#!/home/pytorch/pytorch/sandbox/bin/python3

import random
import re
from tic_tac_toe import TicTacToe


class Trainer():

    def __init__(self):
        filename = "data/tic-tac-toe-dataset.txt"
        self.dataset = []
        with open(filename, "r") as f:
            for line in f:
                self.dataset.append(line.split())

    def get_childs(self, parent):
        regex = re.compile("%s[0-9]+" % (parent))
        return [i for i in self.dataset if regex.match(i[0])]

    def best_option(self, parent):
        """Depending of the parent, given as argument, returns the best move.
        """
        # TODO: For the moment, this method do not evaluates losses
        # or nil options, in order to avoid a loss.
        childs = self.get_childs(parent)
        branches = tuple(set(child[0][len(parent)] for child in childs))
        results = []
        for branch in branches:
            regex = re.compile("%s[0-9]*" % (parent + branch))
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
            current = len(parent) % 2 == 0 and one or two
            win = (current / len(leaves)) * 100
            results.append((win, branch))
        results.sort(reverse=True)
        return(int(results[0][-1]))


if __name__ == "__main__":
    trainer = Trainer()
    game = TicTacToe()
    moves = []
    while True:
        game.representation()
        move = trainer.best_option("".join(str(i) for i in moves))
        game.move(move)
        moves.append(move)
        if game.game_is_over():
            game.representation()
            break
        game.end_turn()
    #  choice = random.choice(trainer.dataset)
    #  print(choice[0])
    #  parent = choice[0][:3]
    #  print(trainer.best_option(parent))
