import unittest
from parameterized import parameterized

class MaxHeap:
    def __init__(self):
        self.heap = []

    def _left_child(self, idx):
        return 2 * idx + 1

    def _right_child(self, idx):
        return 2 * idx + 2

    def _parent(self, idx):
        return (idx - 1) // 2

    def _swap(self, idx1, idx2):
        self.heap[idx1], self.heap[idx2] = self.heap[idx2], self.heap[idx1]

    def insert(self, value):
        self.heap.append(value)
        current = len(self.heap) - 1

        while current > 0 and self.heap[current] > self.heap[self._parent(current)]:
            self._swap(current, self._parent(current))
            current = self._parent(current)

    def _sink_down(self, index):
        max_index = index
        while True:
            left_index = self._left_child(index)
            right_index = self._right_child(index)

            if (left_index < len(self.heap) and 
                    self.heap[left_index] > self.heap[max_index]):
                max_index = left_index

            if (right_index < len(self.heap) and 
                    self.heap[right_index] > self.heap[max_index]):
                max_index = right_index

            if max_index != index:
                self._swap(index, max_index)
                index = max_index
            else:
                return
                       
    def pop(self):
        if len(self.heap) == 0:
            return None

        if len(self.heap) == 1:
            return self.heap.pop()

        max_value = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sink_down(0)

        return max_value
          
          
class TestMaxHeap(unittest.TestCase):
    def setUp(self):
        self.heap = MaxHeap()

    @parameterized.expand([
        ([100, 99, 61, 58, 72], [100]),
        ([100, 99, 75, 58, 72, 61], [100, 75]),
    ])
    def test_insert(self, expected_heap, values):
        self.heap.insert(99)
        self.heap.insert(72)
        self.heap.insert(61)
        self.heap.insert(58)
        for value in values:
            self.heap.insert(value)
        self.assertEqual(self.heap.heap, expected_heap)

    @parameterized.expand([
        ([95, 75, 80, 55, 60, 50, 65], 0),
        ([80, 75, 65, 55, 60, 50], 1),
        ([75, 60, 65, 55, 50], 2),
    ])
    def test_pop(self, expected_heap, n_pop):
        self.heap.insert(95)
        self.heap.insert(75)
        self.heap.insert(80)
        self.heap.insert(55)
        self.heap.insert(60)
        self.heap.insert(50)
        self.heap.insert(65)
        for _ in range(n_pop):
            self.heap.pop()
        self.assertEqual(self.heap.heap, expected_heap)

if __name__ == '__main__':
    unittest.main()
