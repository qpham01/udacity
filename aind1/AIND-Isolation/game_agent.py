"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

PREVIOUS_WINNING_MOVES = {2541362466717371775, 6613518629096764163, -5922411857726037879, \
    -4124497621896485880, 3712459401281602566, 8959178795773461642, 2043278012265184782, \
    -3117231302084458479, 1179086284973040145, -4065377259998740588, -5643407236606792939, \
    -8880693091086586596, -7205252939492120675, -9219103124958940003, 4595044614575255580, \
    -8454898365370349406, -7562532989805496157, -3528899285924755678, -7411251582680342621, \
    -681149453922084829, -1639357473832979670, -9112372332266814035, 3131934224197803057, \
    -717265128158893004, 2634389482399694004, -6835707712614575433, -2107432665631315914, \
    4523663662488875447, 4178600175680483647, -5475465816669476157, 9151901460630510014, \
    4454756738268245574, 4783577801001473222, -2556277200415811511, 2410160771677001291, \
    -5318178240155740844, -8564206493023033258, 2760371433826102867, 2252284256375038421, \
    -4698803984634587306, -8124041784644533284, 1914665153764618205, -2340381771048152733, \
    -7461629609994424475, -8132711426459648409, -6408641414864738585, 1109001614428474598, \
    2378083494567628513, -3233339490522568983, -6142529134493233941, -5172695250098518419, \
    -2942012007243071891, -4689516257051816591, -2033223071635331472, 6499624790210474479, \
    8150728969479665519, -833970151060394892, 5742102486159145462, -4430211693584937218, \
    -7793216138332559870}

PREVIOUS_WINNING_MOVES2 = {-2266353293002449055, -650875449337648158, 5525512056910805665, \
    -2364816481388250909, 4911500270585164644, -2828323045933224216, 3627434641136602726, \
    -8016088851845327542, -7419125688417755986, 812416312008381563, -6908546411081592330, \
    -8359326560832983721, -1375765300315435754, 7885614127719850548, 3629181729033305238, \
    3192378902674493656, 627632426918275578, -8660370879263009826, -418406813429635108, \
    6050162101678600475}

BEST_CENTER_IMPROVED_MOVES = []
BEST_IMPROVED_MOVES = []

USE_CENTER_IMPROVED_MOVES = 0


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    global USE_CENTER_IMPROVED_MOVES

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # If is center move, return inf
    location = game.get_player_location(game.active_player)
    if location == (4, 4):
        return float("inf")

    # Identify partitions
    our_moves = set(game.get_legal_moves())
    opp_moves = set(game.get_legal_moves(game.inactive_player))

    shared_moves = our_moves & opp_moves
    if not shared_moves:
        # If partition found, identify move as winning or losing
        if len(our_moves) >= len(opp_moves):
            return float("inf")
        else:
            return float("-inf")


    return improved_score(game, player, factor=2)



def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return improved_score(game, player, factor=2.0)



def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return improved_score(game, player) - 0.2 * center_score(game, player)


MOVES = set()

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    move_library = None
    move_count = 5

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.game_moves = []

        if IsolationPlayer.move_library is None:
            IsolationPlayer.move_library = set()
            for move in MOVES:
                IsolationPlayer.move_library.add(move)

    def collect_moves(self):
        """ Print the first 'count' moves in a game """
        for move in self.game_moves[:IsolationPlayer.move_count]:
            IsolationPlayer.move_library.add(move)


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """    
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        super(MinimaxPlayer, self).__init__(search_depth, score_fn, timeout)
        self.depth_searched = 0

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        self.game_moves.append(game.forecast_move(best_move).hash())

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        self.depth_searched = 0
        best_value = float('-Inf')
        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0] if len(legal_moves) > 0 else (-1, -1)
        for move in legal_moves:
            value = self.min_value(game.forecast_move(move), depth - 1)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def terminal_test(self, game):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return not game.get_legal_moves()


    def min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #if self.terminal_test(game) or depth == 0:
        if depth == 0:
            self.depth_searched = self.search_depth - depth
            return self.score(game, self)

        value = float('Inf')
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            value = min(value, self.max_value(game.forecast_move(move), depth - 1))
        return value


    def max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #if self.terminal_test(game) or depth == 0:
        if depth == 0:
            self.depth_searched = self.search_depth - depth
            return self.score(game, self)

        value = float('-Inf')
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            value = max(value, self.min_value(game.forecast_move(move), depth - 1))
        return value


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        super(AlphaBetaPlayer, self).__init__(search_depth, score_fn, timeout)
        self.depth_searched = 0

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        self.depth_searched = 0
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            search_depth = 1
            while True:
                # Iterative deepening:  continue searching while there is still time,
                # but save the best result full search to return when time runs out.
                move = self.alphabeta(game, search_depth)
                if move is None:
                    break
                best_move = move
                self.depth_searched = search_depth
                search_depth += 1

        except SearchTimeout:
            self.game_moves.append(game.forecast_move(best_move).hash())
            return best_move

        self.game_moves.append(game.forecast_move(best_move).hash())

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        value = float('-Inf')
        best_value = float('-Inf')
        legal_moves = game.get_legal_moves()
        best_move = legal_moves[0] if len(legal_moves) > 0 else (-1, -1)        
        for move in legal_moves:
            value = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
        #print("Best value, move at depth: {}, {} at {}".format(best_value, best_move, depth))
        return best_move

    def terminal_test(self, game):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return not game.get_legal_moves()


    def min_value(self, game, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game) or depth == 0:
            return self.score(game, self)
       
        value = float('Inf')
        for move in game.get_legal_moves():
            value = min(value, self.max_value(game.forecast_move(move), depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value


    def max_value(self, game, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game) or depth == 0:
            return self.score(game, self)

        value = float('-Inf')
        for move in game.get_legal_moves():
            value = max(value, self.min_value(game.forecast_move(move), depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value


# The following methods are copied from sample_player.py and are used in the composite
# scoring schemes in combination with one or more other scoring functions.  The check
# for losing or winning are removed from these methods because they are already performed
# in the custom score functions.
def open_move_score(game, player):
    """The basic evaluation function described in lecture that outputs a score
    equal to the number of moves open for your computer player on the board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    return float(len(game.get_legal_moves(player)))

MAX_IMPROVED_SCORE = float('-Inf')
MIN_IMPROVED_SCORE = float('Inf')

def improved_score(game, player, factor=1):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    global MAX_IMPROVED_SCORE, MIN_IMPROVED_SCORE

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    score = float(factor * own_moves - opp_moves)
    if score > MAX_IMPROVED_SCORE:
        MAX_IMPROVED_SCORE = score
    if score < MIN_IMPROVED_SCORE:
        MIN_IMPROVED_SCORE = score
    return score

MAX_CENTER_SCORE = float('-Inf')
MIN_CENTER_SCORE = float('Inf')

def center_score(game, player):
    """Outputs a score equal to square of the distance from the center of the
    board to the position of the player.

    This heuristic is only used by the autograder for testing.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    global MAX_CENTER_SCORE, MIN_CENTER_SCORE

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    score = float((h - y)**2 + (w - x)**2)
    if score > MAX_CENTER_SCORE:
        MAX_CENTER_SCORE = score
    if score < MIN_CENTER_SCORE:
        MIN_CENTER_SCORE = score
    return score
