#!/home/pytorch/pytorch/sandbox/bin/python3

import random

import torch
from torch.autograd import Variable

from tic_tac_toe import TicTacToe
from net import Net
from utils import chunker, plot


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

    def evaluate(self, trunk, branches, _type="win"):
        results = []
        current = (
            _type == "nil" and "nil" or
            len(trunk) % 2 == 0 and "one" or
            "two")
        for branch in branches:
            data = self.dataset.get(trunk + str(branch), None)
            if data is None:
                results.append((0, branch))
                continue
            # counters of victories for player one and two and for nil games
            qty = sum(list(data.values()))
            win = (data[current] / qty) * 100
            results.append((win, int(branch)))
        results.sort(reverse=True)
        return(results[0])

    def best_branch(self, trunk, branches):
        """Depending on the parent given as argument, returns the best move.
        """
        ratio_win, branch = self.evaluate(trunk, branches)
        if ratio_win < 50:
            ratio_nil, nil = self.evaluate(trunk, branches, "nil")
            if ratio_nil > ratio_win:
                branch = nil
        return branch

    def get_game_from_sequence(self, sequence):
        moves = [int(i) for i in sequence]
        game = TicTacToe()
        for move in moves:
            game.move(move)
            game.end_turn()
        return game

    def get_matrix_from_game(self, game):
        table = [
            [game.table[0][i + j] and 1 or game.table[1][i + j] and 2 or 0
             for i in range(3)] for j in range(0, 9, 3)]
        return [table]

    def get_tensor_from_game(self, game):
        table = self.get_matrix_from_game(game)
        tensor = torch.Tensor([table])
        return Variable(tensor, requires_grad=True)

    def get_branch_from_tensor(self, tensor):
        best_val, best_index = 0, None
        for i, branch in enumerate(tensor):
            if branch > best_val:
                best_val = branch
                best_index = i
        return best_index

    def get_data_from_chunk(self, chunk):
        games = []
        moves = []
        valid_moves = []
        for sequence in chunk:
            if len(sequence) < 3:
                index = 0
            else:
                index = random.randrange(0, len(sequence) - 2)
            root = "".join(sequence[:index])
            game = trainer.get_game_from_sequence(root)
            branches = game.valid_moves()
            best_branch = trainer.best_branch(root, branches)
            games.append(trainer.get_matrix_from_game(game))
            moves.append(best_branch)
            valid_moves.append(game.valid_moves())
        games = Variable(torch.Tensor(games))
        moves = Variable(torch.LongTensor(moves))
        return games, moves, valid_moves

    def get_infos(self, correct_moves, tensors, valid_moves):
        pairs = zip(correct_moves.data, tensors.data, valid_moves)
        correct_count = 0
        error_count = 0
        four_count = 0
        for target, tensor, moves in pairs:
            guess = self.get_branch_from_tensor(tensor)
            if target == guess:
                correct_count += 1
            if guess not in moves:
                error_count += 1
            if guess == 4:
                four_count += 1
        corrects = (correct_count / len(correct_moves)) * 100
        errors = (error_count / len(correct_moves)) * 100
        fours = (four_count / len(correct_moves)) * 100
        return corrects, errors, fours


if __name__ == "__main__":
    trainer = Trainer()
    net = Net()
    chunks = chunker(trainer.trainset)
    plots = [
        [("accuracy", "epoch"), (0, len(chunks)), (0, 100), 221, "Accuracy"],
        [("loss", "epoch"), (0, len(chunks)), (0, 3), 222, "Loss"],
        [("errors", "epoch"), (0, len(chunks)), (0, 100), 223, "Errors"],
        [("fours", "epoch"), (0, len(chunks)), (0, 100), 224, "Fours"],
    ]
    with plot(plots, filename="plots/accuracy.png", grid=True) as p:
        for epoch, chunk in enumerate(chunks):
            games, moves, valid_moves = trainer.get_data_from_chunk(chunk)
            out = net.forward(games)
            loss = net.criterion(out, moves)
            loss.backward()
            net.optimizer.step()
            accuracy, errors, fours = trainer.get_infos(
                moves, out, valid_moves)
            new_values = {
                "accuracy": {"x_data": epoch, "y_data": accuracy},
                "loss": {"x_data": epoch, "y_data": loss.data[0]},
                "errors": {"x_data": epoch, "y_data": errors},
                "fours": {"x_data": epoch, "y_data": fours},
            }
            p.update(new_values)
            print("epoch N°%s of size %s, loss = %s" % (
                epoch, len(chunk), loss.data[0]))
