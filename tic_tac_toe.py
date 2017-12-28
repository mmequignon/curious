#!/home/pytorch/pytorch/sandbox/bin/python3


class TicTacToe():
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

    def __init__(self):
        self.table = [[False for i in range(9)] for j in range(2)]
        horizontals = [[i + j for i in [0, 1, 2]] for j in range(0, 9, 3)]
        verticals = [[i + j for i in [0, 3, 6]] for j in range(3)]
        diagonals = [[0, 4, 8], [2, 4, 6]]
        self.lines = horizontals + verticals + diagonals
        self.turn = 0

    def game_is_won(self):
        """Returns True is a player won the game, else False.
        """
        return any(any(
            all(player[case] for case in line) for line in self.lines)
            for player in self.table)

    def no_moves_left(self):
        return all(self.table[0][i] or self.table[-1][i] for i in range(9))

    def game_is_over(self):
        return self.game_is_won() or self.no_moves_left()

    def valid_moves(self):
        """Return all positions where there is no piece.
        """
        return [i for i in range(9) if
                not (self.table[0][i] or self.table[-1][i])]

    def swap(self):
        self.table[0], self.table[1] = self.table[1], self.table[0]

    def end_turn(self):
        self.swap()
        self.turn += 1

    def move(self, case):
        self.table[0][case] = True

    def representation(self):
        separator = "-------"
        print(separator)
        for l in range(0, 9, 3):
            line = [self.table[0][l+c] and "X" or
                    self.table[-1][l+c] and "O" or
                    " " for c in range(3)]
            print("|%s|%s|%s|" % (line[0], line[1], line[2]))
            print(separator)
        print()
