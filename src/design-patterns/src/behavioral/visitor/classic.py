from abc import ABC

# taken from https://tavianator.com/the-visitor-pattern-in-python/
# ↓↓↓ LIBRARY CODE ↓↓↓ 

def _qualname(obj):
    """Get the fully-qualified name of an object (including module)."""
    return obj.__module__ + '.' + obj.__qualname__


def _declaring_class(obj):
    """Get the name of the class that declared an object."""
    name = _qualname(obj)
    return name[:name.rfind('.')]


# Stores the actual visitor methods
_methods = {}


# Delegating visitor implementation
def _visitor_impl(self, arg):
    """Actual visitor method implementation."""
    method = _methods[(_qualname(type(self)), type(arg))]
    return method(self, arg)


# The actual @visitor decorator
def visitor(arg_type):
    """Decorator that creates a visitor method."""

    def decorator(fn):
        declaring_class = _declaring_class(fn)
        _methods[(declaring_class, arg_type)] = fn

        # Replace all decorated methods with _visitor_impl
        return _visitor_impl

    return decorator

# ↑↑↑ LIBRARY CODE ↑↑↑


class Visitor(ABC):
    def visit(self, obj):
        raise NotImplementedError()


class Expression(ABC):
    def accept(self, visitor: Visitor):
        visitor.visit(self)


class DoubleExpression(Expression):
    def __init__(self, value):
        self.value = value


class AdditionExpression(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class ExpressionPrinter(Visitor):
    """ Visitor decorator finds the right overload """
    def __init__(self):
        self.buffer = []

    @visitor(DoubleExpression)
    def visit(self, de: DoubleExpression):
        self.buffer.append(str(de.value))

    @visitor(AdditionExpression)
    def visit(self, ae: AdditionExpression):
        self.buffer.append('(')
        self.visit(ae.left)  # ae.left.accept(self) # double dispatch
        self.buffer.append('+')
        self.visit(ae.right)  # ae.right.accept(self) # double dispatch
        self.buffer.append(')')

    def __str__(self):
        return ''.join(self.buffer)


class ExpressionEvaluator(Visitor):
  def __init__(self):
    self.value = None

  @visitor(DoubleExpression)
  def visit(self, de: DoubleExpression):
    self.value = de.value

  @visitor(AdditionExpression)
  def visit(self, ae: AdditionExpression):
    self.visit(ae.left)  # ae.left.accept(self) # double dispatch
    temp = self.value
    self.visit(ae.right)  # ae.right.accept(self) # double dispatch
    self.value += temp


if __name__ == '__main__':
    # represents 1+(2+3)
    e = AdditionExpression(
        DoubleExpression(1),
        AdditionExpression(
            DoubleExpression(2),
            DoubleExpression(3)
        )
    )

    printer = ExpressionPrinter()
    printer.visit(e)

    evaluator = ExpressionEvaluator()
    evaluator.visit(e)

    print(f'{printer} = {evaluator.value}')
