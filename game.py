"""
File to hold the tic tac toe game
"""

from neuralnetwork import NeuralNetwork
import random

class Game():
    """ Represents a specific game of Tic Tac Toe """
    NAUGHT = 'O'
    CROSS = 'X'
    EMPTY = '-'
    _NAUGHT_TURN = True
    _won = False
    _nn1 = None

    def __init__(self, NW = 1, CW = -1, EW = 0):
        """ Defines the board of size 3 """
        self.board = [[self.EMPTY for i in range(3)] for i in range(3)]
        self._nn = NeuralNetwork()
        self._NW = NW
        self._CW = CW
        self._EW = EW

    def reset(self):
        """ Reset the board to its default state. Retains neural networks """
        self._NAUGHT_TURN = True
        self._won = False
        self.board.clear()
        self.board = [[self.EMPTY for i in range(3)] for i in range(3)]

    def create_nn(self):
        """ Creates a default neural network and attaches to this game """
        self._nn1 = NeuralNetwork()
        return self._nn1

    def new_layer(nn, nodes, inputs, alpha=0.5):
        """ Add a layer to a neural network """
        nn.new_layer(nodes, inputs, alpha)

    def _train(self, nn, data, lr=2):
        """
        Takes in a board, output as a 1-D list of tuples and
        trains a nn using it
        """
        for row, output in data:
            for i, e in enumerate(row):
                if e == self.CROSS:
                    row[i] = self._CW
                elif e == self.NAUGHT:
                    row[i] = self._NW
                else:
                    row[i] = self._EW
            for i, e in enumerate(output):
                if e == self.CROSS:
                    output[i] = self._CW
                elif e == self.NAUGHT:
                    output[i] = self._NW
                else:
                    output[i] = self._EW
        nn.back_prop(data, alpha=lr)

    def display(self):
        """ Prints out the current board """
        for row in self.board:
            print("|".join(row))

    def _play(self, x, y):
        """ Make a move to a specific location. Knows who's turn it is """
        symbol = self.NAUGHT if self._NAUGHT_TURN else self.CROSS
        self._NAUGHT_TURN = False if self._NAUGHT_TURN else True

        assert self.board[x][y] == self.EMPTY

        self.board[x][y] = symbol
        return self._test_win()

    def req_play(self):
        """ Requests a human player makes input """
        x = input("Input row: ")
        y = input("Input column: ")
        if self._play(int(x), int(y)):
            print("You win!!")

    def start_game(self, nn):
        """ Allows a human player to play against a neural network """
        while not self._won:
            self.display()
            self.req_play()
            if self._test_win():
                print("Game finished!")
                return
            self._nn_play()
            self.display()
            if self._test_win():
                print("Game finished!")
                return

    def _nn_play(self, nn):
        """ Makes the move the neural network believes is best """
        #Convert to list
        internal = []
        for row in self.board:
            for pos in row:
                if pos == self.NAUGHT:
                    internal.append(self._NW)
                elif pos == self.CROSS:
                    internal.append(self._CW)
                else:
                    internal.append(self._EW)
        res = nn.run(internal)
        move, x, y = self._find_max(res)
        self._play(x, y)

    def convert_to_board(move):
        y = move % 3
        x = (move - y) // 3
        return x, y

    def _find_max(self, res):
        """ Finds the best legal move from a list of outputs """
        move = -1
        for i, pos in enumerate(res):
            x, y = Game.convert_to_board(i)
            #Special case
            if move < 0 and self.board[x][y] == self.EMPTY:
                move = i
            if pos > res[move] and self.board[x][y] == self.EMPTY:
                move = i
        x, y = Game.convert_to_board(move)
        return move, x, y

    def _legal(self, x, y):
        """ Determines whether a move is legal or not """
        return self.board[x][y] == self.EMPTY

    def _test_win(self):
        """ Internal method to test who has won """
        for row in self.board:
            if row[0] == row[1] and row[1] == row[2] and row[0] != self.EMPTY:
                self._won = True
                return True
        transposedBoard = zip(*self.board)
        for row in transposedBoard:
            if row[0] == row[1] and row[1] == row[2] and row[0] != self.EMPTY:
                self._won = True
                return True
        row = self.board
        if row[0][0] == row[1][1] and row[0][0] == row[2][2] and row[0][0] != self.EMPTY:
            self._won = True
            return True
        if row[2][0] == row[1][1] and row[2][0] == row[0][2] and row[2][0] != self.EMPTY:
            self._won = True
            return True
        return False

    def _test_draw(self):
        """ Tests to see if the game is a draw or not """
        for row in self.board:
            for e in row:
                if e == self.EMPTY:
                    return False
        self._won = False
        return True

    def _random_move(self):
        """ Chooses a random legal move """
        r = random.randint(0, 8)
        x, y = Game.convert_to_board(r)
        while not self._legal(x, y):
            r = random.randint(0, 8)
            x, y = Game.convert_to_board(r)
        self._play(x, y)

    def test_against_random(g, loops=3500):
        wins = 0
        loses = 0
        draws = 0
        for _ in range(loops):
            while not g._test_win() and not g._test_draw():
                g._nn_play(g._nn1)
                if g._test_win():
                    wins = wins + 1
                    break
                elif g._test_draw():
                    draws = draws + 1
                    break
                g._random_move()
                if g._test_win():
                    loses = loses + 1
                    break
                elif g._test_draw():
                    draws = draws + 1
                    break
            g.reset()
        pct_win = float(100.0/loops)*wins
        pct_loss = float(100.0/loops)*loses
        pct_draw = float(100.0/loops)*draws
        print("Wins: {} ({}%) Loses: {} ({}%) Draws: {} ({}%)".format(wins, \
                    pct_win, loses, pct_loss, draws, pct_draw))
        return pct_win, pct_loss, pct_draw

    def _flatten(self, board):
        flat = []
        for row in board:
            for e in row:
                flat.append(e)
        return flat

    def _create_rnd_board(self):
        """ Will create an unwon board """
        #Create a random board
        rnd_int = random.randint(0, 5)
        for _ in range(rnd_int):
            self._random_move()
        if self._test_win() or self._test_draw():
            self.reset()
            return self._create_rnd_board()
        else:
            return self.board

    def _train_example(self):
        """ Internal method for semi-supervisied training """
        print("--")
        self.reset()
        #Create a random board
        start_board = self._flatten(self._create_rnd_board())
        #Ask the user for a move
        g.display()
        print("Move as: {}".format(self.NAUGHT if self._NAUGHT_TURN else self.CROSS))
        g.req_play()
        end_board = self._flatten(self.board)
        g.display()
        return (start_board, end_board)

    def create_examples(self, n):
        """ Creates n exampl boards for semi-supervisied learning """
        pairs = []
        for _ in range(n):
            print("New board")
            pairs.append(self._train_example())
        return pairs
