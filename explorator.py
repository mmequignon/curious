#!/home/pytorch/pytorch/sandbox/bin/python3

import random

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

    def __init__(self):
        """Make a number 'tries' of random games and store results in a file
        'filename' formatted as followed :
        012345678 02468 1357 1
            - first value defines the game move sequence
            - second value defines the player 1 move sequence
            - third valud defines the Player 2 move sequence
            - fourth value defines the winner of the game (False if nil)
        """
        tries = 1000000
        filename = "data/tic-tac-toe-dataset.txt"
        with open(filename, "w") as f:
            for i in range(tries):
                game = TicTacToe()
                self.explore(game)
                sequence = game.get_sequence()
                seq_1 = "".join(
                    [sequence[i] for i in range(len(sequence)) if i % 2 == 0])
                seq_2 = "".join(
                    [sequence[i] for i in range(len(sequence)) if i % 2 == 1])
                winner = str(game.winner)
                f.write(" ".join([sequence, seq_1, seq_2, winner]) + "\n")


if __name__ == "__main__":
    exploration = Explorator()
