""" Helper functions for minimax determination """
import sys

def terminal_test(gameState):
    """ Return True if the game is over for the active player
    and False otherwise.
    """
    return not gameState.get_legal_moves()


def min_value(gameState):
    """ Return the value for a win (+1) if the game is over,
    otherwise return the minimum value over all legal child
    nodes.
    """
    if terminal_test(gameState):
        return 1
    value = sys.maxsize
    legal_moves = gameState.get_legal_moves()
    for move in legal_moves:
        game = gameState.forecast_move(move)
        value = min(value, max_value(game))
    return value


def max_value(gameState):
    """ Return the value for a loss (-1) if the game is over,
    otherwise return the maximum value over all legal child
    nodes.
    """
    if terminal_test(gameState):
        return -1
    value = -sys.maxsize
    legal_moves = gameState.get_legal_moves()
    for move in legal_moves:
        game = gameState.forecast_move(move)
        value = max(value, min_value(game))
    return value
