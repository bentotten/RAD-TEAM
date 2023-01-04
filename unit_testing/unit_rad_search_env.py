import unittest
from gym_rad_search.envs import RadSearch
from gym_rad_search.envs.rad_search_env import BBox 

# https://github.com/cgoldberg/python-unittest-tutorial

class UnitTestModule(unittest.TestCase):
    def test_dummy(self):
        print('\n:: test_dummy ::')
        self.assertEqual(2+2,4)


class MathHelpers(unittest.TestCase):
    def setUp(self):
        self.env = RadSearch()

    def tearDown(self):
        del self.env

    def test_bbox(self):
        """ Test bboxß"""
        print('\n:: test_bbox ::')

        # Make large box
        test_bbox = tuple(((0.0, 0.0), (1000000.0, 0.0), (1000000.0, 1000000.0), (0.0, 1000000.0)))
        test_env = RadSearch(bbox = test_bbox)
        self.assertEqual(test_env.bbox, test_bbox)   
        
        # Fail for too small an searchable space
        test_bbox = tuple(((0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)))
        self.assertRaises(AssertionError, RadSearch, bbox=test_bbox)


if __name__ == '__main__':
    unittest.main()