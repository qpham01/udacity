""" Test for isolation game """
import unittest
from isolation import GameState
#from isolation_udacity import GameState

class GameStateTest(unittest.TestCase):
    block = (2, 1)
    move_1 = (1, 2)
    move_2 = (3, 2)
    xlim = 4
    ylim = 4

    """ Test GameState class """
    def test_00_forecast_move(self):
        game = GameState()
        game = game.forecast_move((0, 0))
        assert game._board[0][0] == 1
        assert game._board[2][1] == 1, 'value is {}'.format(game._board[2][1])
        game = game.forecast_move((1, 0))
        assert game._board[0][0] == 1
        assert game._board[1][0] == 1, 'value is {}'.format(game._board[1][0])
        assert game._board[2][1] == 1

    def first_two_moves(self):
        """ Create a game and make two moves """
        width = GameStateTest.xlim
        height = GameStateTest.ylim
        game = GameState(xlim=width, ylim=height)

        # Get legal moves for player 1
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == width * height - 1
        #assert GameStateTest.block not in legal_moves

        # Player 1 moves
        game = game.forecast_move(GameStateTest.move_1)

        # Get legal moves for player 2
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) == width * height - 2, "legal moves count: {}".\
            format(len(legal_moves))
        assert GameStateTest.move_1 not in legal_moves

        # Player 2 moves
        game = game.forecast_move(GameStateTest.move_2)

        return game

    def check_good_bad_moves(self, good_moves, bad_moves):
        """ Test good and bad moves """
        game = self.first_two_moves()

        # Get legal moves for player 1, turn 2
        legal_moves = game.get_legal_moves()

        for move in good_moves:
            assert move in legal_moves, 'good {} is not in {}'.format(move, legal_moves)
        for move in bad_moves:
            assert move not in legal_moves, 'bad {} is in {}'.format(move, legal_moves)


    def test_10_legal_moves_players_block(self):
        """ Check player positions and block at (2, 1) not in legal_moves """
        bad_moves = [GameStateTest.move_1, GameStateTest.move_2, GameStateTest.block]
        self.check_good_bad_moves([], bad_moves)


    def test_20_legal_moves_north(self):
        """ Check north legal moves """
        good_moves = [(1, 0), (1, 1)]
        bad_moves = [(1, -1)]
        self.check_good_bad_moves(good_moves, bad_moves)


    def test_30_legal_moves_northeast(self):
        """ Check northeast legal moves """
        bad_moves = [(2, 1), (3, 0), (4, -1)]
        self.check_good_bad_moves([], bad_moves)


    def test_40_legal_moves_east(self):
        """ Check east legal moves """
        good_moves = [(2, 2)]
        bad_moves = [(4, 2), (3, 2)]
        self.check_good_bad_moves(good_moves, bad_moves)


    def test_50_legal_moves_southeast(self):
        """ Check east legal moves """
        good_moves = [(2, 3)]
        bad_moves = [(3, 4)]
        self.check_good_bad_moves(good_moves, bad_moves)


    def test_60_legal_moves_south(self):
        """ Check south legal moves """
        good_moves = [(1, 3)]
        bad_moves = [(1, 4)]
        self.check_good_bad_moves(good_moves, bad_moves)


    def test_70_legal_moves_southwest(self):
        """ Check south legal moves """
        good_moves = [(0, 3)]
        bad_moves = [(-1, 4)]
        self.check_good_bad_moves(good_moves, bad_moves)


    def test_80_legal_moves_west(self):
        """ Check east legal moves """
        good_moves = [(0, 2)]
        bad_moves = [(-1, 2)]
        self.check_good_bad_moves(good_moves, bad_moves)


    def test_90_legal_moves_northwest(self):
        """ Check east legal moves """
        good_moves = [(0, 1)]
        bad_moves = [(-1, 0)]
        self.check_good_bad_moves(good_moves, bad_moves)

if __name__ == '__main__':
    unittest.main()
