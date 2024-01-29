# Creational Patterns

Creational patterns defer some part of object creation to a subclass
or another object.

- [1. Builder](#1-builder)
- [2. Factory](#2-factory)
- [3. Abstract Factory](#3-abstract-factory)
- [4. Prototype](#4-prototype)
  - [4.1. Prototype Factory](#41-prototype-factory)
- [5. Singleton](#5-singleton)

## 1. Builder

- creation of a complex object in parts

## 2. Factory

- Defines an interface to create an object, but let subclasses decide which class to instantiate.
- Factory method is any method that returns an object (typically static).
- Factory class groups factory methods (it can be a nested class).

```python
class Person:
    def __init__(self, id, name):
        self.id = id
        self.name = name

class PersonFactory:
    id = 0

    def create_person(self, name):
        p = Person(PersonFactory.id, name)
        PersonFactory.id += 1
        return p
```

```python
from enum import Enum
from math import *


class CoordinateSystem(Enum):
    CARTESIAN = 1
    POLAR = 2


class Point:
    # def __init__(self, x, y):
    #     self.x = x
    #     self.y = y

    def __str__(self):
        return f'x: {self.x}, y: {self.y}'

    # redeclaration won't work
    # def __init__(self, rho, theta):

    def __init__(self, a, b, system=CoordinateSystem.CARTESIAN):
        if system == CoordinateSystem.CARTESIAN:
            self.x = a
            self.y = b
        elif system == CoordinateSystem.POLAR:
            self.x = a * sin(b)
            self.y = a * cos(b)

        # steps to add a new system
        # 1. augment CoordinateSystem
        # 2. change init method

    @staticmethod
    def new_cartesian_point(x, y):
        # factory method
        return Point(x, y)

    @staticmethod
    def new_polar_point(rho, theta):
        # factory method
        return Point(rho * sin(theta), rho * cos(theta))

    class Factory:
        """Factory class"""
        def new_cartesian_point(self, x, y):
            return Point(x, y)
        
        def new_polar_point(self, rho, theta):
            return Point(rho * sin(theta), rho * cos(theta))

    # This is effectively a singleton factory instance
    factory = Factory()

# Take out factory methods to a separate class
class PointFactory:
    @staticmethod
    def new_cartesian_point(x, y):
        return Point(x, y)

    @staticmethod
    def new_polar_point(rho, theta):
        return Point(rho * sin(theta), rho * cos(theta))


if __name__ == '__main__':
    # standard initializer
    p1 = Point(2, 3, CoordinateSystem.CARTESIAN)
    # note that the user may not know about the factory class
    p2 = PointFactory.new_cartesian_point(1, 2) 
    # or you can expose factory through the type
    p3 = Point.Factory.new_cartesian_point(5, 6)
    p4 = Point.factory.new_cartesian_point(7, 8)
    print(p1, p2, p3, p4)
```

## 3. Abstract Factory

- Hierarchy of factories.

```python
from abc import ABC
from enum import Enum, auto


class HotDrink(ABC):
    def consume(self):
        pass


class Tea(HotDrink):
    def consume(self):
        print('This tea is nice but I\'d prefer it with milk')


class Coffee(HotDrink):
    def consume(self):
        print('This coffee is delicious')


class HotDrinkFactory(ABC):
    def prepare(self, amount):
        pass


class TeaFactory(HotDrinkFactory):
    def prepare(self, amount):
        print(f'Put in tea bag, boil water, pour {amount}ml, enjoy!')
        return Tea()


class CoffeeFactory(HotDrinkFactory):
    def prepare(self, amount):
        print(f'Grind some beans, boil water, pour {amount}ml, enjoy!')
        return Coffee()


class HotDrinkMachine:
    class AvailableDrink(Enum):  # violates OCP
        COFFEE = auto()
        TEA = auto()

    factories = []
    initialized = False

    def __init__(self):
        if not self.initialized:
            self.initialized = True
            for d in self.AvailableDrink:
                name = d.name[0] + d.name[1:].lower()
                factory_name = name + 'Factory'
                factory_instance = eval(factory_name)()
                self.factories.append((name, factory_instance))





    def make_drink(self):
        print('Available drinks:')
        for f in self.factories:
            print(f[0])

        s = input(f'Please pick drink (0-{len(self.factories)-1}): ')
        idx = int(s)
        s = input(f'Specify amount: ')
        amount = int(s)
        return self.factories[idx][1].prepare(amount)



def make_drink(type):
    if type == 'tea':
        return TeaFactory().prepare(200)
    elif type == 'coffee':
        return CoffeeFactory().prepare(50)
    else:
        return None


if __name__ == '__main__':
    # entry = input('What kind of drink would you like?')
    # drink = make_drink(entry)
    # drink.consume()

    hdm = HotDrinkMachine()
    drink = hdm.make_drink()
    drink.consume()
```

## 4. Prototype

- When it's easier to create a copy of an object instead of creating a new one.
- `copy.deepcopy()` - create a **deep** copy of an object by recursively copying all attributes
- `copy.copy()` - create a **shallow** copy of an object without the recursive scan

```python
import copy

class Address:
    def __init__(self, street_address, city, country):
        self.country = country
        self.city = city
        self.street_address = street_address

    def __str__(self):
        return f'{self.street_address}, {self.city}, {self.country}'


class Person:
    def __init__(self, name, address):
        self.name = name
        self.address = address

    def __str__(self):
        return f'{self.name} lives at {self.address}'

john = Person("John", Address("123 London Road", "London", "UK"))
print(john)
# jane = john  # this would copy the reference 
jane = copy.deepcopy(john)
jane.name = "Jane"
jane.address.street_address = "124 London Road"
print(john, jane)
```

### 4.1. Prototype Factory

- Sometimes it gets tedious creating copies manually.

```python
import copy

class Address:
    def __init__(self, street_address, suite, city):
        self.suite = suite
        self.city = city
        self.street_address = street_address

    def __str__(self):
        return f'{self.street_address}, Suite #{self.suite}, {self.city}'

class Employee:
    def __init__(self, name, address):
        self.address = address
        self.name = name

    def __str__(self):
        return f'{self.name} works at {self.address}'


class EmployeeFactory:
    main_office_employee = Employee("", Address("123 East Dr", 0, "London"))
    aux_office_employee = Employee("", Address("123B East Dr", 0, "London"))

    @staticmethod
    def __new_employee(proto, name, suite):
        result = copy.deepcopy(proto)
        result.name = name
        result.address.suite = suite
        return result

    @staticmethod
    def new_main_office_employee(name, suite):
        return EmployeeFactory.__new_employee(
            EmployeeFactory.main_office_employee,
            name, suite
        )

    @staticmethod
    def new_aux_office_employee(name, suite):
        return EmployeeFactory.__new_employee(
            EmployeeFactory.aux_office_employee,
            name, suite
        )

# main_office_employee = Employee("", Address("123 East Dr", 0, "London"))
# aux_office_employee = Employee("", Address("123B East Dr", 0, "London"))

# john = copy.deepcopy(main_office_employee)
#john.name = "John"
#john.address.suite = 101
#print(john)

# would prefer to write just one line of code
jane = EmployeeFactory.new_aux_office_employee("Jane", 200)
print(jane)
```

## 5. Singleton
