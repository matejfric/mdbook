from enum import Enum

class State(Enum):
    LOCKED = 0
    OPEN = 1
    ERROR = 2

class CombinationLock:
    def __init__(self, combination):
        self.state = State.LOCKED
        self.status = self.state.name
        self.combination = ''.join(map(str, combination))

    def reset(self):
        self.state = State.LOCKED
        self.status = self.state.name

    def enter_digit(self, digit):
        if self.state == State.OPEN:
            return

        if self.status == self.state.name:
            self.status = str(digit)
        else:
            self.status += str(digit)

        if not self.combination.startswith(self.status):
            self.state = State.ERROR
            self.status = self.state.name

        if self.combination == self.status:
            self.state = State.OPEN
            self.status = self.state.name

def test_success():
    cl = CombinationLock([1, 2, 3, 4, 5])
    assert('LOCKED' == cl.status)
    cl.enter_digit(1)
    assert('1' == cl.status)
    cl.enter_digit(2)
    assert('12' == cl.status)
    cl.enter_digit(3)
    assert('123' == cl.status)
    cl.enter_digit(4)
    assert('1234' == cl.status)
    cl.enter_digit(5)
    assert('OPEN'== cl.status)

test_success()