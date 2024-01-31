# 3. Structural Patterns

Structural patterns compose classes or objects.

- [1. Adapter](#1-adapter)
  - [1.1. Adapter Example](#11-adapter-example)
  - [1.2. Adapter with Caching](#12-adapter-with-caching)
- [2. Decorator](#2-decorator)
  - [2.1. Python Functional Decorator](#21-python-functional-decorator)
  - [2.2. Standard OOP Decorator](#22-standard-oop-decorator)
  - [2.3. Dynamic Decorator](#23-dynamic-decorator)
- [3. Bridge](#3-bridge)
- [4. Façade](#4-façade)
- [5. Flyweight](#5-flyweight)
- [6. Proxy](#6-proxy)
- [7. Composite](#7-composite)

## 1. Adapter

- When an API doesn't provide the exact implementation we need, we use an adapter.

### 1.1. Adapter Example

```python
from unittest import TestCase

class Square:
    def __init__(self, side=0):
        self.side = side

def calculate_area(rc):
    return rc.width * rc.height


class SquareToRectangleAdapter:
    def __init__(self, square):
        # Save the reference to the square object
        self.square = square

    @property
    def width(self):
        return self.square.side

    @property
    def height(self):
        return self.square.side


class Evaluate(TestCase):
    def test_exercise(self):
        sq = Square(11)
        adapter = SquareToRectangleAdapter(sq)
        self.assertEqual(121, calculate_area(adapter))
        sq.side = 10
        self.assertEqual(100, calculate_area(adapter))
```

### 1.2. Adapter with Caching

```python
class Point:
    def __init__(self, x, y):
        self.y = y
        self.x = x


def draw_point(p):
    print('.', end='')


# ^^ you are given this

# vv you are working with this
class Line:
    def __init__(self, start, end):
        self.end = end
        self.start = start


class Rectangle(list):
    """ Represented as a list of lines. """

    def __init__(self, x, y, width, height):
        super().__init__()
        self.append(Line(Point(x, y), Point(x + width, y)))
        self.append(Line(Point(x + width, y), Point(x + width, y + height)))
        self.append(Line(Point(x, y), Point(x, y + height)))
        self.append(Line(Point(x, y + height), Point(x + width, y + height)))


class LineToPointAdapter:
  """Adapts a line to a list of points"""
    count = 0
    cache = {}  # for caching previously calculated points

    def __init__(self, line):
        self.h = hash(line)
        if self.h in self.cache:
            return

        super().__init__()
        self.count += 1
        print(f'{self.count}: Generating points for line ' +
              f'[{line.start.x},{line.start.y}]→[{line.end.x},{line.end.y}]')

        left = min(line.start.x, line.end.x)
        right = max(line.start.x, line.end.x)
        top = min(line.start.y, line.end.y)
        bottom = min(line.start.y, line.end.y)

        points = []

        if right - left == 0:
            for y in range(top, bottom):
                points.append(Point(left, y))
        elif line.end.y - line.start.y == 0:
            for x in range(left, right):
                points.append(Point(x, top))

        self.cache[self.h] = points

    def __iter__(self):
        return iter(self.cache[self.h])

def draw(rcs):
    print('Drawing some rectangles...')
    for rc in rcs:
        for line in rc:
            adapter = LineToPointAdapter(line)
            for p in adapter:
                draw_point(p)
    print('\n')


if __name__ == '__main__':
    rs = [
        Rectangle(1, 1, 10, 10),
        Rectangle(3, 3, 6, 6)
    ]

    draw(rs)
    draw(rs)

    # you can define your own hashes or use the defaults
    print(hash(Line(Point(1, 1), Point(10, 10))))
```

## 2. Decorator

- A class that keeps reference to an object it decorates.
- A decorator can add utility attributes and methods to augment the object's features.
- Decorator changes the functionality of an object **without modifying it's internal state/code**.

### 2.1. Python Functional Decorator

```python
import time

def time_it(func):
  def wrapper():
    start = time.time()
    result = func()
    end = time.time()
    print(f'{func.__name__} took {int((end-start)*1000)}ms')
  return wrapper

@time_it
def some_op():
  print('Starting op')
  time.sleep(1)
  print('We are done')
  return 123

if __name__ == '__main__':
  # some_op()
  # time_it(some_op)()
  some_op()
```

### 2.2. Standard OOP Decorator

```python
from abc import ABC


class Shape(ABC):
    def __str__(self):
        return ''


class Circle(Shape):
    def __init__(self, radius=0.0):
        self.radius = radius

    def resize(self, factor):
        self.radius *= factor

    def __str__(self):
        return f'A circle of radius {self.radius}'


class ColoredShape(Shape):
    def __init__(self, shape, color):
        if isinstance(shape, ColoredShape):
            raise Exception('Cannot apply ColoredDecorator twice')
        self.shape = shape
        self.color = color

    def __str__(self):
        return f'{self.shape} has the color {self.color}'


class TransparentShape(Shape):
    def __init__(self, shape, transparency):
        self.shape = shape
        self.transparency = transparency

    def __str__(self):
        return f'{self.shape} has {self.transparency * 100.0}% transparency'


if __name__ == '__main__':
    circle = Circle(2)
    print(circle)

    # Apply 'ColoredShape' decorator
    red_circle = ColoredShape(circle, "red")
    print(red_circle)

    # ColoredShape doesn't have resize()
    # red_circle.resize(3)

    red_half_transparent_circle = TransparentShape(red_circle, 0.5)
    print(red_half_transparent_circle)

    # Nothing prevents double application
    mixed = ColoredShape(ColoredShape(Circle(3), 'red'), 'blue')
    print(mixed)
```

### 2.3. Dynamic Decorator

- solves the issue of 'Standard Decorator' which prevents us from using underlying method of an object wrapped by a decorator

```python
class FileWithLogging:
  def __init__(self, file):
    self.file = file

  def writelines(self, strings):
    self.file.writelines(strings)
    print(f'wrote {len(strings)} lines')

  def __iter__(self):
    return self.file.__iter__()

  def __next__(self):
    return self.file.__next__()

  def __getattr__(self, item):
    return getattr(self.__dict__['file'], item)

  def __setattr__(self, key, value):
    if key == 'file':
      self.__dict__[key] = value
    else:
      setattr(self.__dict__['file'], key)

  def __delattr__(self, item):
    delattr(self.__dict__['file'], item)


if __name__ == '__main__':
  file = FileWithLogging(open('hello.txt', 'w'))
  file.writelines(['hello', 'world'])
  file.write('testing')
  file.close()
```

```python
class Circle:
  def __init__(self, radius):
    self.radius = radius

  def resize(self, factor):
    self.radius *= factor

  def __str__(self):
    return 'A circle of radius %s' % self.radius

class Square:
  def __init__(self, side):
    self.side = side

  def __str__(self):
    return 'A square with side %s' % self.side


class ColoredShape:
  def __init__(self, shape, color):
    self.color = color
    self.shape = shape

  def resize(self, factor):
    # Notice this:
    r = getattr(self.shape, 'resize', None)
    if callable(r):
      self.shape.resize(factor)

  def __str__(self):
    return "%s has the color %s" %\
           (self.shape, self.color)
```

## 3. Bridge

- **Decouple** abstraction from implementation.
- A stronger form of encapsulation.
- Build a bridge between two hierarchies.

```python
# Circles and squares
# Each can be rendered in vector or raster form

class Renderer():
    def render_circle(self, radius):
        pass


class VectorRenderer(Renderer):
    def render_circle(self, radius):
        print(f'Drawing a circle of radius {radius}')


class RasterRenderer(Renderer):
    def render_circle(self, radius):
        print(f'Drawing pixels for circle of radius {radius}')


class Shape:
    def __init__(self, renderer):
        self.renderer = renderer

    def draw(self): pass
    def resize(self, factor): pass


class Circle(Shape):
    def __init__(self, renderer, radius):
        super().__init__(renderer)
        self.radius = radius

    def draw(self):
        self.renderer.render_circle(self.radius)

    def resize(self, factor):
        self.radius *= factor


if __name__ == '__main__':
    raster = RasterRenderer()
    vector = VectorRenderer()
    circle = Circle(vector, 5)
    circle.draw()
    circle.resize(2)
    circle.draw()
```

## 4. Façade

- Simplify API over a set of classes.
- Optionally expose internal methods for power users.

```python
class Buffer:
  def __init__(self, width=30, height=20):
    self.width = width
    self.height = height
    self.buffer = [' '] * (width*height)

  def __getitem__(self, item):
    return self.buffer.__getitem__(item)

  def write(self, text):
    self.buffer += text

class Viewport:
  def __init__(self, buffer=Buffer()):
    self.buffer = buffer
    self.offset = 0

  def get_char_at(self, index):
    return self.buffer[self.offset+index]

  def append(self, text):
    self.buffer += text

# Facade
class Console:
  def __init__(self):
    b = Buffer()
    self.current_viewport = Viewport(b)
    self.buffers = [b]
    self.viewports = [self.current_viewport]

  # high-level
  def write(self, text):
    self.current_viewport.buffer.write(text)

  # low-level
  def get_char_at(self, index):
    return self.current_viewport.get_char_at(index)


if __name__ == '__main__':
  c = Console()
  c.write('hello')
  ch = c.get_char_at(0)
```

## 5. Flyweight

- Spatial complexity optimization technique.
- Store common data externally by specifying an **index**, **range**, or **reference** into the external data store.

```python
class FormattedText:
    def __init__(self, plain_text):
        self.plain_text = plain_text
        self.caps = [False] * len(plain_text)

    def capitalize(self, start, end):
        for i in range(start, end):
            self.caps[i] = True

    def __str__(self):
        result = []
        for i in range(len(self.plain_text)):
            c = self.plain_text[i]
            result.append(c.upper() if self.caps[i] else c)
        return ''.join(result)


class BetterFormattedText:
    def __init__(self, plain_text):
        self.plain_text = plain_text
        self.formatting = []

    class TextRange:
        def __init__(self, start, end, capitalize=False, bold=False, italic=False):
            self.end = end
            self.bold = bold
            self.capitalize = capitalize
            self.italic = italic
            self.start = start

        def covers(self, position):
            return self.start <= position <= self.end

    def get_range(self, start, end):
        range = self.TextRange(start, end)
        self.formatting.append(range)
        return range

    def __str__(self):
        result = []
        for i in range(len(self.plain_text)):
            c = self.plain_text[i]
            for r in self.formatting:
                if r.covers(i) and r.capitalize:
                    c = c.upper()
            result.append(c)
        return ''.join(result)


if __name__ == '__main__':
    ft = FormattedText('This is a brave new world')
    ft.capitalize(10, 15)
    print(ft)

    bft = BetterFormattedText('This is a brave new world')
    bft.get_range(16, 19).capitalize = True
    print(bft)
```

## 6. Proxy

## 7. Composite

> **Composite** is a mechanism for treating individual (scalar) objects and compositions of objects in a *uniform manner*.

- Some *singular* and *composed* objects need similar or identical behavior.
- For example, scalars and vectors.
- Composition lets us make compound objects.

><details><summary> Code example </summary>
>
>```python
>from abc import ABC
>from collections.abc import Iterable
>import unittest
>
>class Summable(Iterable, ABC):
>    @property
>    def sum(self):
>        s = 0
>        for x in self:
>            for y in x:
>                s += y
>        return s
>
>class SingleValue(Summable):
>    def __init__(self, value):
>        self.value = value
>    def __iter__(self):
>        yield self.value
>
># Notice that order of inheritance matters here
>class ManyValues(list, Summable):
>    pass
>
>class FirstTestSuite(unittest.TestCase):
>    def test(self):
>        single_value = SingleValue(11)
>        other_values = ManyValues()
>        other_values.append(22)
>        other_values.append(33)
>        # Make a list of all values (uniform usage)
>        all_values = ManyValues()
>        all_values.append(single_value)  # SingleValue
>        all_values.append(other_values)  # ManyValues
>        self.assertEqual(all_values.sum, 66)
>
>if __name__ == '__main__':
>    unittest.main()
>```
>
></details>
