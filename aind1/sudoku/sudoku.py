""" Sudoku Solver """
import copy

def cross(a, b):
    return [s+t for s in a for t in b]

class Sudoku:
    """ Sudoku solver """
    def __init__(self):
        self.rows = 'ABCDEFGHI'
        self.cols = '123456789'
        self.boxes = cross(self.rows, self.cols)

        self.row_units = [cross(r, self.cols) for r in self.rows]
        self.column_units = [cross(self.rows, c) for c in self.cols]
        self.square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in \
            ('123', '456', '789')]
        self.unitlist = self.row_units + self.column_units + self.square_units
        self.units = dict((s, [u for u in self.unitlist if s in u]) for s in self.boxes)
        self.peers = dict((s, set(sum(self.units[s], [])) - set([s])) for s in self.boxes)

    def grid_values(self, grid):
        """Convert grid string into {<box>: <value>} dict with '123456789' value for empties.

        Args:
            grid: Sudoku grid in string form, 81 characters long
        Returns:
            Sudoku grid in dictionary form:
            - keys: Box labels, e.g. 'A1'
            - values: Value in corresponding box, e.g. '8', or '123456789' if it is empty.
        """
        """
        Produce a dictionary of cells labeled by A1 through I9 whose values are from a string like:
        '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
        """
        values = dict()
        for i, label in enumerate(self.boxes):
            values[label] = grid[i] if not grid[i] == '.' else '123456789'

        return values

    def eliminate(self, value_dict):
        """Eliminate values from peers of each box with a single value.

        Go through all the boxes, and whenever there is a box with a single value,
        eliminate this value from the set of values of all its peers.

        Args:
            values: Sudoku in dictionary form.
        Returns:
            Resulting Sudoku in dictionary form after eliminating values.
        """
        """
        Given a value dict with unfilled cells ('.'), eliminate invalid values and returns a new
        dictionary where '.' are replaced by a string of possible valid values.
        """
        all_values = '123456789'
        values = copy.copy(value_dict)
        for i, label in enumerate(self.boxes):
            if value_dict[label] == all_values:
                row_index = ord(label[0]) - ord('A')
                row_unit = self.row_units[row_index]
                col_index = int(label[1]) - 1
                column_unit = self.column_units[col_index]
                square_index = row_index // 3 * 3 + col_index // 3
                square_unit = self.square_units[square_index]

                for unit_label in row_unit:
                    value = value_dict[unit_label]
                    if value != all_values:
                        values[label] = values[label].replace(value, '')
                for unit_label in column_unit:
                    value = value_dict[unit_label]
                    if value != all_values:
                        values[label] = values[label].replace(value, '')
                for unit_label in square_unit:
                    value = value_dict[unit_label]
                    if value != all_values:
                        values[label] = values[label].replace(value, '')
        return values
