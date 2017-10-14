"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""
import timeit
import unittest

import isolation
import game_agent

from importlib import reload


def make_move(player, board, time_limit=500):
    """ Make a move with player """
    time_millis = lambda: 1000 * timeit.default_timer()
    move_start = time_millis()
    time_left = lambda: time_limit - (time_millis() - move_start)
    move = player.get_move(board, time_left)
    new_board = board.forecast_move(move)
    return move, new_board

class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)

    def test_01_depth(self):
        """ Test that depth-limited minimax performs to the right depth """
        target_depth = 4
        player1 = game_agent.MinimaxPlayer(search_depth=target_depth)
        player2 = game_agent.MinimaxPlayer(search_depth=target_depth)
        board = isolation.Board(player1, player2)
        assert player1.depth_searched == 0
        make_move(player1, board)
        assert player1.depth_searched == target_depth, 'depth searched: {}'.\
            format(player1.depth_searched)

    def test_02_iterative_deepening(self):
        """ Test that depth-limited minimax performs to the right depth """
        player1 = game_agent.AlphaBetaPlayer()
        player2 = game_agent.AlphaBetaPlayer()
        board = isolation.Board(player1, player2)

        move, board = make_move(player1, board)
        assert player1.depth_searched > 5
        #print()
        #print("1 move", move)

        move, board = make_move(player2, board)
        assert player2.depth_searched > 5
        #print("2 move", move)

if __name__ == '__main__':
    unittest.main()
