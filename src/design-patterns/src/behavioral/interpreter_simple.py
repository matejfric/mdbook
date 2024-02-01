from enum import Enum, auto
import unittest
import math

class Token:
    class Type(Enum):
        INTEGER = auto()
        PLUS = auto()
        MINUS = auto()

    def __init__(self, type, text):
        self.type = type
        self.text = text

    def __str__(self):
        return f'`{self.text}`'
    

class BinaryOperation:
    def __init__(self):
        self.type = None
        self.left = None
        self.right = None

    @property
    def value(self):
        if self.type == Token.Type.PLUS:
            return self.left + self.right
        elif self.type == Token.Type.MINUS:
            return self.left - self.right
            

class ExpressionProcessor:
    def __init__(self):
        self.variables = {}

    def calculate(self, expression):
        try:
            lexed = self._lex(expression)
            result = self._parse_eval(lexed)
        except ValueError as e:
            print(e)
            return math.nan
        return result
        
    def _lex(self, input):
        result = []
        i = 0
        while i < len(input):
            if input[i] == '+':
                result.append(Token(Token.Type.PLUS, '+'))
            elif input[i] == '-':
                result.append(Token(Token.Type.MINUS, '-'))
            elif input[i].isalpha():
                alphas = []
                while i < len(input) and input[i].isalpha():
                    alphas.append(input[i])
                    i += 1
                if len(alphas) > 1:
                    raise ValueError(f"Variables with more than one letter are not supported. Got `{''.join(alphas)}`.")
                if input[i-1] in self.variables:
                    result.append(Token(Token.Type.INTEGER, str(self.variables[input[i-1]])))
                else:
                    raise ValueError(f'Variable `{input[i-1]}` not found.')
                continue  # skip i += 1
            elif input[i].isdigit():
                digits = []
                while i < len(input) and input[i].isdigit():
                    digits.append(input[i])
                    i += 1
                result.append(Token(Token.Type.INTEGER, ''.join(digits)))
                continue  # skip i += 1
            else:
                raise ValueError(f'Invalid character found: {input[i]}')
            i += 1
        return result
    
    def _parse_eval(self, tokens):
        has_lhs = False
        bin_op = BinaryOperation()
        result = 0
        for token in tokens:
            if token.type == Token.Type.INTEGER:
                if not has_lhs:
                    bin_op.left = int(token.text)
                    result = bin_op.left
                    has_lhs = True
                else:
                    bin_op.right = int(token.text)
                    if bin_op.type is None:
                        raise ValueError('Invalid syntax. No operator found.')
                    result = bin_op.value
                    bin_op.left = result
            elif token.type == Token.Type.PLUS:
                bin_op.type = Token.Type.PLUS
            elif token.type == Token.Type.MINUS:
                bin_op.type = Token.Type.MINUS
        return result
    

class FirstTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ep = ExpressionProcessor()
        ep.variables['x'] = 5
        cls.ep = ep

    def test_simple(self):
        self.assertEqual(1, self.ep.calculate('1'))

    def test_addition(self):
        self.assertEqual(6, self.ep.calculate('1+2+3'))

    def test_addition_with_variable(self):
        self.assertEqual(6, self.ep.calculate('1+x'))

    def test_failure(self):
        result = self.ep.calculate('1+xy')
        self.assertIs(math.nan, result)
        
    
if __name__=="__main__":
    unittest.main()
