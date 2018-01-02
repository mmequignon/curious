#!/home/pytorch/pytorch/sandbox/bin/python3

import random
from itertools import permutations

from tic_tac_toe import TicTacToe


class Explorator():

    def explore(self, game):
        """Recursive function that returns moves and the winner.
        """
        valid_moves = game.valid_moves()
        move = random.choice(valid_moves)
        game.move(move)
        game.end_turn()
        if not game.game_is_over():
            self.explore(game)

    def compute_games(self, tries):
        games = []
        for i in range(tries):
            game = TicTacToe()
            self.explore(game)
            games.append((game.get_sequence(), str(game.winner)))
        return games

    def parse_games(self, games):
        ratios = {}
        for sequence, w in games:
            if w.isnumeric():
                winner = int(w)
            else:
                winner = 2
            while sequence:
                p1 = [sequence[i] for
                      i in range(len(sequence)) if i % 2 == 0]
                p2 = [sequence[i] for
                      i in range(len(sequence)) if i % 2 == 1]
                for i in permutations(p1, len(p1)):
                    for j in permutations(p2, len(p2)):
                        neighbour = "".join(
                            [k % 2 == 0 and p1[k // 2] or p2[k // 2] for
                             k in range(len(p1 + p2))])
                        if neighbour not in ratios:
                            ratios[neighbour] = [0, 0, 0]
                        ratios[neighbour][winner] += 1
                sequence = sequence[:-1]
        return ratios

    def compute_ratios(self, games, out_filename):
        ratios = self.parse_games(games)
        with open(out_filename, "w") as f:
            for sequence, ratio in ratios.items():
                f.write(" ".join(
                    [sequence, str(ratio[0]), str(ratio[1]), str(ratio[2])]) +
                    "\n")

    def __init__(self):
        tries = 1000000
        ratios_filename = "data/tic-tac-toe-ratios-dataset.csv"
        games = self.compute_games(tries)
        self.compute_ratios(games, ratios_filename)


if __name__ == "__main__":
    exploration = Explorator()
