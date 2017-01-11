"""
Contains traffic sign classification unit tests
"""

import unittest
from traffic_signs import TrafficSignData

class TrafficSignTest(unittest.TestCase):
    """
    Test classification of traffic signs
    """
    def test_traffic_sign_data(self):
        """
        Test traffic sign data initialization
        """
        data = TrafficSignData('train.p', 'test.p', True, 0.9)
        data.print_data_dimensions()
        expected_data_size = 39209
        self.assertEqual(expected_data_size, data.n_train)

        expected_train_size = int(data.train_ratio * data.n_train)
        self.assertEqual(expected_train_size, len(data.x_train))
        self.assertEqual(expected_train_size, len(data.y_train))

        expected_valid_size = expected_data_size - expected_train_size
        self.assertEqual(expected_valid_size, len(data.x_valid))
        self.assertEqual(expected_valid_size, len(data.y_valid))

if __name__ == '__main__':
    unittest.main()
