#!/home/pytorch/pytorch/sandbox/bin/python3

import random
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional


class Game():
    """The table is represented by two matrix.
    At each end of the turn, each matrix is swapped.
    So, during the current move, the matrix of the current player is
    self.table[0] and the matrix of the opponent is self.table[-1].
    Player 1 plays first.
    At each turn, its matrix is self.table[turn % 2 == 0],
    and the matrix of the player 2 is self.table[turn % 2 == 1].
    Player 1 -> X
    Player 2 -> O
    """

    # Definition of "What is a line ?".
    horizontals = [[i + j for i in range(3)] for j in range(0, 9, 3)]
    verticals = [[i + j for i in range(0, 9, 3)] for j in range(3)]
    diagonals = [[0, 4, 8], [2, 4, 6]]
    lines = horizontals + verticals + diagonals
    game_size = (3, 3)
    game_moves_vector = 9
    name = "tic-tac-toe"

    __slots__ = [
        "table", "turn", "sequence", "winner", "reward", "game_is_over"]

    def __init__(self):
        self.table = [0 for i in range(9)]
        self.reward = 0
        self.turn = 0
        self.sequence = []
        self.winner = 0
        self.game_is_over = False

    def game_is_won(self):
        """Returns True is a player won the game, else False.
        """
        for line in Game.lines:
            for player in (1, 2):
                if all(self.table[case] == player for case in line):
                    return True
        return False

    def no_moves_left(self):
        return all(self.table[case] != 0 for case in range(9))

    def valid_moves(self):
        """Return a vector containing wether 1 or 0 depending on the
        positions validities.
        example : [0, 1, 0, 1, 0, 1, 0, 1, 0]
        """
        return [i == 0 and 1 or 0 for i in self.table]

    def valid_actions(self):
        """Returns a vector containing valid actions.
        example : [1, 3, 8]
        """
        return [i for i in range(Game.game_moves_vector) if self.table[i] == 0]

    def get_board(self):
        """Return a tensor of the form 3*3*3.
        3 layers :
            - positions of P1 moves;
            - positions of P2 moves;
            - 0 or 1 that defines the current player.
        """
        board = []
        for column in range(0, 9, 3):
            board.append(self.table[column:column + 3])
        return board

    def end_turn(self):
        if self.game_is_won() or self.no_moves_left():
            self.game_is_over = True
            if self.game_is_won():
                self.winner = self.get_current_player()
                if self.winner == 1:
                    self.reward = 1
                elif self.winner == 2:
                    self.reward = -1
        else:
            self.turn += 1

    def move(self, case):
        self.table[case] = self.get_current_player()
        self.sequence.append(case)
        self.end_turn()

    def get_current_player(self):
        return self.turn % 2 + 1

    def representation(self):
        separator = "-------"
        print(separator)
        for l in range(0, 9, 3):
            line = [self.table[l + c] == 1 and "X" or
                    self.table[l + c] == 2 and "O" or
                    " " for c in range(3)]
            print("|%s|%s|%s|" % (line[0], line[1], line[2]))
            print(separator)

    def copy(self):
        return deepcopy(self)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.board_x, self.board_y = Game.game_size
        self.vector_length = Game.game_moves_vector
        self.conv1 = nn.Conv2d(1, 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 9)
        self.fc4 = nn.Linear(512, 1)

    def value_head(self, s):
        v = self.fc4(s)
        return v

    def pi_head(self, s):
        s = self.fc3(s)
        return functional.softmax(s, dim=-1)

    def dropout_layers(self, s):
        s = functional.dropout(
            functional.relu(self.fc1(s)), p=0.3, training=self.training)
        s = functional.dropout(
            functional.relu(self.fc2(s)), p=0.3, training=self.training)
        return s

    def convolution_layers(self, s):
        s = functional.relu(self.bn1(self.conv1(s)))
        s = functional.relu(self.bn2(self.conv2(s)))
        s = functional.relu(self.conv3(s))
        s = s.view(-1, 512)
        return s

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = self.convolution_layers(s)
        s = self.dropout_layers(s)
        pi = self.pi_head(s)
        v = self.value_head(s)
        print(pi)
        return pi, v

    def compute_loss(self, predicted_pis, pis, predicted_rewards, rewards):
        """Computes mean-squared loss for rewards and 1G"""
        size = predicted_pis.size()[0]
        pi_loss = torch.sum(pis * predicted_pis) / size
        v_loss = torch.sum((rewards - predicted_rewards)**2) / size
        return pi_loss - v_loss


if __name__ == "__main__":
    game = Game()
    game.representation()
    while not game.game_is_over:
        move = random.choice(game.valid_actions())
        game.move(move)
        game.representation()
    print(game.winner)
