from utils import *

def grid_values(grid):
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
    for i, label in enumerate(boxes):
        values[label] = grid[i] if not grid[i] == '.' else '123456789'

    # values = eliminate_values(values)
    return values

def eliminate(value_dict):
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
    for i, label in enumerate(boxes):
        if value_dict[label] == all_values:
            row_index = ord(label[0]) - ord('A')
            row_unit = row_units[row_index]
            col_index = int(label[1]) - 1
            column_unit = column_units[col_index]
            square_index = row_index // 3 * 3 + col_index // 3
            square_unit = square_units[square_index]

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