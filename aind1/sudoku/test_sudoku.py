""" Tests for Sudoku """

import unittest
from utils import display
from sudoku import Sudoku

EASY_GRID = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
HARD_GRID = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
ALL_VALUES = '123456789'

class SudokuTests(unittest.TestCase):
    """ Tests for Sudoku """
    def test_01_grid_values(self):
        """ Test for grid_values method """
        sudoku = Sudoku()
        values = sudoku.grid_values(EASY_GRID)
        assert len(values) == 81
        assert values['A1'] == ALL_VALUES
        assert values['A2'] == ALL_VALUES
        assert values['A3'] == '3'
        assert values['I7'] == '3'
        assert values['I8'] == ALL_VALUES
        assert values['I9'] == ALL_VALUES

    def test_02_eliminate(self):
        """ Test eliminate method """
        sudoku = Sudoku()
        values = sudoku.grid_values(EASY_GRID)
        values = sudoku.eliminate(values)
        assert values['A1'] == '45'
        assert values['A2'] == '4578'
        assert values['A3'] == '3'
        assert values['I7'] == '3'
        assert values['I8'] == '24678'
        assert values['I9'] == '2467'

    def test_03_only_choice(self):
        """ Tests only choice method """
        sudoku = Sudoku()
        values = sudoku.grid_values(EASY_GRID)
        values = sudoku.eliminate(values)
        for _ in range(0, 4):
            values = sudoku.only_choice(values)

        assert values['A6'] == '1', "{}: {}".format('A6', values['A6'])
        assert values['E2'] == '2', "{}: {}".format('C9', values['C9'])

    def test_04_reduce_puzzle(self):
        """ Tests reduce_puzzle """
        sudoku = Sudoku()
        values = sudoku.grid_values(EASY_GRID)
        values = sudoku.reduce_puzzle(values)
        for label in sudoku.boxes:
            assert len(values[label]) == 1

    def test_05_reduce_hard_puzzle(self):
        """ Tests reduce_puzzle """
        sudoku = Sudoku()
        values = sudoku.grid_values(HARD_GRID)
        values = sudoku.reduce_puzzle(values)
        #for label in sudoku.boxes:
        #    assert len(values[label]) == 1
        display(values)
        print()
        values['G2'] = '9'
        values = sudoku.reduce_puzzle(values)
        display(values)

if __name__ == '__main__':
    unittest.main()
