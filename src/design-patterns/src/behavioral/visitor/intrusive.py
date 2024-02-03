class DoubleExpression:
    def __init__(self, value):
        self.value = value

    def print(self, buffer: list[str]):
        buffer.append(str(self.value))

    def eval(self): return self.value


class AdditionExpression:
    def __init__(self, left, right):
        self.right = right
        self.left = left

    def print(self, buffer: list[str]):
        buffer.append('(')
        self.left.print(buffer)
        buffer.append('+')
        self.right.print(buffer)
        buffer.append(')')

    def eval(self):
        return self.left.eval() + self.right.eval()


if __name__ == '__main__':
    # 1+(2+3)
    e = AdditionExpression(
        DoubleExpression(1),
        AdditionExpression(
            DoubleExpression(2),
            DoubleExpression(3)
        )
    )

    # Here `buffer` acts as an intrusive visitor. 
    buffer = []
    e.print(buffer)
    print(''.join(buffer), '=', e.eval())
