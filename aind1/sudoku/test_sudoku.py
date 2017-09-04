""" Tests for Sudoku """

import unittest
from sudoku import Sudoku

TEST_GRID = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
ALL_VALUES = '123456789'

class SudokuTests(unittest.TestCase):
    """ Tests for Sudoku """
    def test_01_grid_values(self):
        """ Test for grid_values method """
        sudoku = Sudoku()
        values = sudoku.grid_values(TEST_GRID)
        assert len(values) == 81
        assert values['A1'] == ALL_VALUES
        assert values['A2'] == ALL_VALUES
        assert values['A3'] == '3'
        assert values['I7'] == '3'
        assert values['I8'] == ALL_VALUES
        assert values['I9'] == ALL_VALUES

    def test_01_eliminate(self):
        """ Test eliminate method """
        sudoku = Sudoku()
        values = sudoku.grid_values(TEST_GRID)
        values = sudoku.eliminate(values)
        assert values['A1'] == '45'
        assert values['A2'] == '4578'
        assert values['A3'] == '3'
        assert values['I7'] == '3'
        assert values['I8'] == '24678'
        assert values['I9'] == '2467'


if __name__ == '__main__':
    unittest.main()
