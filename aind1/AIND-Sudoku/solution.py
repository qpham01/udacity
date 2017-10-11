assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

boxes = cross(rows, cols)

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
diag_units1 = [rows[i] + cols[i] for i, _ in enumerate(rows)]
diag_units2 = [rows[i] + cols[8 - i] for i, _ in enumerate(rows)]
print(diag_units1)
print(diag_units2)
unitlist = [diag_units1, diag_units2] + row_units + column_units + square_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    for unit in row_units:
        values = eliminate_naked_twins_numbers(unit, values)
    for unit in column_units:
        values = eliminate_naked_twins_numbers(unit, values)
    for unit in square_units:
        values = eliminate_naked_twins_numbers(unit, values)
    values = eliminate_naked_twins_numbers(diag_units1, values)
    values = eliminate_naked_twins_numbers(diag_units2, values)
    return values
    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    values = dict()
    for label in boxes:
        values[label] = grid[label] if label in grid else '123456789'

    return values

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    print()
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit, '')
    return values

def only_choice(values):
    for unit in unitlist:
        for digit in '123456789':            
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1 and len(values[dplaces[0]]) > 1:
                #print("dplaces for {}: {}, {}".format(digit, dplaces, [values[box] for box in dplaces]))            
                assign_value(values, dplaces[0], digit)
    return values

def eliminate_naked_twins_numbers(unit, values):
    """ Find a list of numbers in naked twins in a unit, if any """
    twin_numbers = dict()
    for box1 in unit:
        box1_len = len(values[box1])
        if box1_len == 2:
            for box2 in unit:
                if box1 == box2:
                    continue
                if values[box1] == values[box2]:
                    for number in values[box1]:
                        twin_numbers[number] = [box1, box2]
    #new_values = values.copy()
    for number in twin_numbers:
        twin_boxes = twin_numbers[number]
        non_twin_boxes = [box for box in unit if box not in twin_boxes]
        for box in non_twin_boxes:
            current_value = values[box]
            test = current_value.replace(number, '')
            assign_value(values, box, test)            
    return values


def reduce_puzzle(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        #values = naked_twins(values)
        values = eliminate(values)
        values = only_choice(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes):
        return values ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    _, s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    values = search(values)
    if values:
        solved_values = [box for box in values.keys() if len(values[box]) == 1]   
        assert len(solved_values) == len(values)
        # display(values)
    else:
        print("No solution found!")
    return values

def test_valid_solution(values):
    for box in values:
        assert len(values[box]) == 1
        assert values[box] in '123456789'

    for unit in unitlist:
        for i, box1 in enumerate(unit):
            for j, box2 in enumerate(unit):
                if i != j:
                    assert values[box1] != values[box2] 
    return True

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
