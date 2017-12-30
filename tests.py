#!/home/pytorch/pytorch/sandbox/bin/python3

import unittest
import random
from tic_tac_toe import TicTacToe
from trainer import Trainer


class TestMechanics(unittest.TestCase):

    def setUp(self):
        self.game = TicTacToe()

    def test_valid_moves_count(self):
        self.assertEqual(len(self.game.valid_moves()), 9)
        moves = [
            (0, 8), (1, 7), (2, 6), (3, 5), (4, 4),
            (5, 3), (7, 2), (6, 1), (8, 0),
        ]
        for move, expected_qty in moves:
            self.game.move(move)
            self.game.end_turn()
            self.assertEqual(len(self.game.valid_moves()), expected_qty)

    def test_diagonal_win(self):
        moves = [0, 3, 4, 5, 8]
        for move in moves:
            self.game.move(move)
            self.game.end_turn()
        self.assertTrue(self.game.game_is_won())

    def test_horizontal_win(self):
        moves = [0, 3, 1, 4, 2]
        for move in moves:
            self.game.move(move)
            self.game.end_turn()
        self.assertTrue(self.game.game_is_won())

    def test_vertical_win(self):
        moves = [0, 1, 3, 4, 6]
        for move in moves:
            self.game.move(move)
            self.game.end_turn()
        self.assertTrue(self.game.game_is_won())

    def test_player_1_win(self):
        moves = [0, 1, 3, 4, 6]
        for move in moves:
            self.game.move(move)
            self.game.end_turn()
        self.assertEqual(self.game.winner, 0)

    def test_player_2_win(self):
        moves = [0, 1, 6, 4, 8, 7]
        for move in moves:
            self.game.move(move)
            self.game.end_turn()
        self.assertEqual(self.game.winner, 1)

    def test_nil(self):
        moves = [0, 1, 2, 3, 4, 5, 7, 6, 8]
        for move in moves:
            self.game.move(move)
            self.game.end_turn()
        self.assertFalse(self.game.winner)


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()

    def test_correct_winner(self):
        for i in range(20):
            winner = False
            game = TicTacToe()
            index = random.randrange(0, len(self.trainer.dataset) - 1)
            data = self.trainer.dataset[index]
            if data[-1].isnumeric():
                winner = int(data[-1])
            moves = [int(m) for m in data[0]]
            for move in moves:
                game.move(move)
                game.end_turn()
            self.assertEqual(game.winner, winner)

    def test_sequences(self):
        for i in range(20):
            game = TicTacToe()
            index = random.randrange(0, len(self.trainer.dataset) - 1)
            data = self.trainer.dataset[index]
            moves = [int(m) for m in data[0]]
            for move in moves:
                game.move(move)
                game.end_turn()
            self.assertEqual(data[0], game.get_sequence())


if __name__ == '__main__':
    unittest.main()
