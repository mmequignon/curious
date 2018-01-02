#!/home/pytorch/pytorch/sandbox/bin/python3

import random

import torch
import numpy

from tic_tac_toe import TicTacToe


class Trainer():

    __slots__ = ["dataset", "trainset", "testset"]

    def load_dataset(self, filename):
        ratios = {}
        with open(filename, "r") as f:
            for line in f:
                sequence, one, two, nil = line.split()
                ratios[sequence] = {
                    "one": int(one),
                    "two": int(two),
                    "nil": int(nil)
                }
        return ratios

    def __init__(self):
        """Gets each item from the dataset file and splits that dataset into
        two sets. One for training, one for testing.
        """
        filename = "data/tic-tac-toe-ratios-dataset.csv"
        self.dataset = self.load_dataset(filename)
        sequences = list(self.dataset.keys())
        split_ratio = 0.8
        split_indice = int(len(sequences) * split_ratio)
        self.trainset = sequences[:split_indice]
        self.testset = sequences[split_indice:]

    def best_branch(self, trunk, branches):
        """Depending on the parent given as argument, returns the best move.
        """
        results = []
        for branch in branches:
            data = self.dataset.get(trunk + str(branch), None)
            if data is None:
                results.append((0, branch))
                continue
            # counters of victories for player one and two and for nil games
            current = len(trunk) % 2 == 0 and "one" or "two"
            qty = sum(list(data.values()))
            win = (data[current] / qty) * 100
            results.append((win, branch))
        results.sort(reverse=True)
        return(int(results[0][-1]))

    def train(self, trunk):
        game = trainer.get_game_from_sequence(trunk)
        branches = game.valid_moves()
        choice = random.choice(branches)
        best_branch = self.best_branch(trunk, branches)
        return choice == best_branch and 0 or 1

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

    def split_table(self, table):
        return (table[:, :2, :2], table[:, :2, 1:],
                table[:, 1:, :2], table[:, 1:, 1:])


def chunker(data, size=2000):
    return [data[i:i+size] for i in range(0, len(data), size)]


if __name__ == "__main__":
    trainer = Trainer()
    chunks = chunker(trainer.trainset)
    for epoch, chunk in enumerate(chunks):
        current_loss = 0
        args = []
        for i, sequence in enumerate(chunk):
            if len(sequence) < 3:
                index = 0
            else:
                index = random.randrange(0, len(sequence) - 2)
            root = "".join(sequence[:index])
            current_loss += trainer.train(root)
        rate = (current_loss / len(chunk)) * 100
        print("epoch N°%s of size %s, loss = %s" % (
            epoch, len(chunk), rate))
