#!/home/pytorch/pytorch/sandbox/bin/python3

import random

from tic_tac_toe import TicTacToe


class Explorator():

    def explore(self, game, played_moves):
        """Recursive function that returns moves and the winner.
        """
        valid_moves = game.valid_moves()
        move = random.choice(valid_moves)
        game.move(move)
        played_moves.append(move)
        if game.game_is_over():
            winner = game.turn % 2 if game.game_is_won() else False
            return played_moves, winner
        game.end_turn()
        return self.explore(game, played_moves)

    def __init__(self):
        """Make a number 'tries' of random games and store results in a file
        'filename' formatted as followed :
        0123456789 1
        """
        tries = 300000
        filename = "data/tic-tac-toe-dataset.txt"
        with open(filename, "w") as f:
            for i in range(tries):
                game = TicTacToe()
                moves, winner = self.explore(game, [])
                f.write("%s %s\n" % ("".join(
                    str(i) for i in moves), str(winner)))


if __name__ == "__main__":
    exploration = Explorator()
