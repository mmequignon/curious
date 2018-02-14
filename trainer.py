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

    def get_tensor_from_game(self, game):
        #  one, two, none = [], [], []
        #  for i in range(0, 9, 3):
        #      line_one, line_two, line_none = [], [], []
        #      for j in range(3):
        #          if game.table[0][i+j]:
        #              line_one.append(1)
        #              line_two.append(0)
        #              line_none.append(0)
        #          elif game.table[1][i+j]:
        #              line_one.append(0)
        #              line_two.append(1)
        #              line_none.append(0)
        #          else:
        #              line_one.append(0)
        #              line_two.append(0)
        #              line_none.append(1)
        #      one.append(line_one)
        #      two.append(line_two)
        #      none.append(line_none)
        #  array = []
        #  array.append(one)
        #  array.append(two)
        #  array.append(none)
        #  tensor = torch.Tensor([array])
        table = [
            [game.table[0][i + j] and 1 or game.table[1][i + j] and 2 or 0
             for i in range(3)] for j in range(0, 9, 3)]
        tensor = torch.Tensor([[table]])
        return Variable(tensor, requires_grad=True)

    def get_tensor_from_branch(self, branch):
        tensor = torch.LongTensor([branch])
        return Variable(tensor)

    def get_branch_from_tensor(self, variable):
        best_val, best_index = 0, None
        tensor = variable.data[0]
        for i, branch in enumerate(tensor):
            if branch > best_val:
                best_val = branch
                best_index = i
        return best_index


if __name__ == "__main__":
    trainer = Trainer()
    net = Net()
    chunks = chunker(trainer.trainset)
    plots = [
        [("accuracy", "epoch"), (0, len(chunks)),
         (0, 100), 221, "Correct moves rate"],
        [("loss", "epoch"), (0, len(chunks)), (0, 3), 222, "Loss"],
        [("wrong moves", "epoch"), (0, len(chunks)),
         (0, 100), 223, "Invalid moves rate"],
        [("4 moves", "epoch"), (0, len(chunks)),
         (0, 100), 224, "4 moves rate"],
    ]
    with plot(plots, filename="plots/accuracy.png", grid=True) as p:
        for epoch, chunk in enumerate(chunks):
            current_loss = 0.0
            current_correctness = 0
            current_wrong_moves = 0
            current_4_moves = 0
            for sequence in chunk:
                net.optimizer.zero_grad()
                if len(sequence) < 3:
                    index = 0
                else:
                    index = random.randrange(0, len(sequence) - 2)
                root = "".join(sequence[:index])
                game = trainer.get_game_from_sequence(root)
                branches = game.valid_moves()
                best_branch = trainer.best_branch(root, branches)
                correct_tensor = trainer.get_tensor_from_branch(best_branch)
                tensor = trainer.get_tensor_from_game(game)
                predicted_tensor = net.forward(tensor)
                predicted_branch = trainer.get_branch_from_tensor(
                    predicted_tensor)
                loss = net.criterion(predicted_tensor, correct_tensor)
                loss.backward()
                net.optimizer.step()
                current_loss += loss.data[0]
                current_correctness += (
                    predicted_branch == best_branch and 1 or 0)
                current_wrong_moves += (
                    predicted_branch not in game.valid_moves() and 1 or 0)
                current_4_moves += (predicted_branch == 4 and 1 or 0)
            loss = current_loss / len(chunk)
            accuracy = (current_correctness / len(chunk)) * 100
            wrong_moves = (current_wrong_moves / len(chunk)) * 100
            four_moves = (current_4_moves / len(chunk)) * 100
            new_values = {
                "accuracy": {"x_data": epoch, "y_data": accuracy},
                "loss": {"x_data": epoch, "y_data": loss},
                "wrong moves": {"x_data": epoch, "y_data": wrong_moves},
                "4 moves": {"x_data": epoch, "y_data": four_moves},
            }
            p.update(new_values)
            print("epoch N°%s of size %s, loss = %s, accuracy = %s" % (
                epoch, len(chunk), loss, accuracy))
    game = TicTacToe()
    while not game.game_is_over():
        game.representation()
        move = 9
        while move not in game.valid_moves():
            if game.turn % 2 == 1:
                move = int(input("votre coup : "))
            else:
                move_tensor = net.forward(tensor)
                move = trainer.get_branch_from_tensor(move_tensor)
            if move not in game.valid_moves():
                print("%s FAIL !" % move)
        game.move(move)
        game.end_turn()
