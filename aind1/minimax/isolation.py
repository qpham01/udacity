""" Implement the isolation game including AI """
import copy

class GameState:
    max_size = 10000

    """ The isolation game state """
    def __init__(self, xlim=3, ylim=2, blocked=[(2, 1)]):
        self.xlim = xlim
        self.ylim = ylim
        self._board = [[0 for _ in range(self.ylim)] for _ in range(self.xlim)]
        for block in blocked:
            self._board[block[0]][block[1]] = 1
        self._parity = 0
        self.last_moves = {}
        self.last_moves[0] = None
        self.last_moves[1] = None

    def forecast_move(self, move):
        """ Return a new board object with the specified move
        applied to the current game state.

        Parameters
        ----------
        move: tuple
            The target position for the active player's next move
        """
        self._board[move[0]][move[1]] = 1
        self.last_moves[self._parity] = move
        self._parity = 1 if self._parity == 0 else 0
        return copy.deepcopy(self)

    def get_legal_moves(self):
        """ Return a list of all legal moves available to the
        active player.  Each player should get a list of all
        empty spaces on the board on their first move, and
        otherwise they should get a list of all open spaces
        in a straight line along any row, column or diagonal
        from their current position. (Players CANNOT move
        through obstacles or blocked squares.) Moves should
        be a pair of integers in (column, row) order specifying
        the zero-indexed coordinates on the board.
        """
        last_move = self.last_moves[self._parity]
        legal_moves = []
        if last_move is None:
            # Return all empty spaces
            for x in range(self.xlim):
                for y in range(self.ylim):
                    if self._board[x][y] == 0:
                        legal_moves.append((x, y))
        else:
            # block indicator in 8 directions, starting north and going clockwise.
            blocks = [0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(1, GameState.max_size):
                for direction in range(0, 8):
                    # Skip blocked directions
                    if blocks[direction] != 0:
                        continue
                    move = self.get_legal_move(direction, i)
                    if move is None:
                        blocks[direction] = 1
                    else:
                        legal_moves.append(move)
                all_blocks = [x for x in blocks if x != 0]
                if len(all_blocks) >= 8:
                    break
        return legal_moves

    def get_legal_move(self, direction, i):
        """ Check if move is legal """
        last_x, last_y = self.last_moves[self._parity]
        if direction in [0, 4]:
            x = last_x
        elif direction in [1, 2, 3]:
            x = last_x + i
        elif direction in [5, 6, 7]:
            x = last_x - i
        else:
            raise ValueError(direction)

        if direction in [2, 6]:
            y = last_y
        elif direction in [3, 4, 5]:
            y = last_y + i
        elif direction in [7, 0, 1]:
            y = last_y - i
        else:
            raise ValueError(direction)

        if x < 0 or y < 0 or x >= self.xlim or y >= self.ylim or self._board[x][y] != 0:
            return None
        return (x, y)
