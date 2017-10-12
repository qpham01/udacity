""" Implement minimax decision """
from minimax_helpers import *

def _minimax_decision(gameState):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.
    
    You can ignore the special case of calling this function
    from a terminal state.
    """
    # The built in `max()` function can be used as argmax!
    return max(gameState.get_legal_moves(),
               key=lambda m: min_value(gameState.forecast_move(m)))

def minimax_decision(gameState):
    """ Return the move along a branch of the game tree that
    has the best possible value.  A move is a pair of coordinates
    in (column, row) order corresponding to a legal move for
    the searching player.
    
    You can ignore the special case of calling this function
    from a terminal state.
    """
    value = -sys.maxsize
    best_value = -sys.maxsize
    best_move = None
    legal_moves = gameState.get_legal_moves()
    for move in legal_moves:
        game = gameState.forecast_move(move)
        value = max(value, min_value(game))
        if value > best_value:
            best_value = value
            best_move = move
    return best_move
