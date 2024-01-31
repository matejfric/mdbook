from abc import ABC
from collections.abc import Iterable
import unittest

class Summable(Iterable, ABC):
    @property
    def sum(self):
        s = 0
        for x in self:
            for y in x:
                s += y
        return s

class SingleValue(Summable):
    def __init__(self, value):
        self.value = value
    def __iter__(self):
        yield self.value

# Notice that order of inheritance matters here
class ManyValues(list, Summable):
    pass

class FirstTestSuite(unittest.TestCase):
    def test(self):
        single_value = SingleValue(11)
        other_values = ManyValues()
        other_values.append(22)
        other_values.append(33)
        # make a list of all values
        all_values = ManyValues()
        all_values.append(single_value)
        all_values.append(other_values)
        self.assertEqual(all_values.sum, 66)

if __name__ == '__main__':
    unittest.main()