# 4. Behavioral Patterns

Behavioral Patterns describe algorithms or cooperation of objects.

- [1. Observer](#1-observer)
  - [1.1. Observable Events](#11-observable-events)
  - [1.2. Property Observers](#12-property-observers)
  - [1.3. Property dependencies](#13-property-dependencies)
  - [1.4. Full Example on Events](#14-full-example-on-events)
- [2. Chain of Responsibility](#2-chain-of-responsibility)
  - [2.1. Method Chain](#21-method-chain)
  - [2.2. Broker Chain](#22-broker-chain)
  - [2.3. Simpler Broker Chain](#23-simpler-broker-chain)
- [3. Command](#3-command)
- [4. Iterator](#4-iterator)
- [5. Template Method](#5-template-method)
- [6. Strategy](#6-strategy)

1. Chain of Responsibility
2. Command
3. Interpreter
4. Iterator - přístup k prvkům kolekce bez znalosti imlementace dané kolekce
5. Memento - "quicksave"
6. Mediator - definuje, jak by spolu měla množina objektů interagovat
7. Observer - definuje 1--* závislost, pokud jeden objekt změní stav, tak všechny závislé objekty jsou automaticky aktualizovány
8. State - umožňuje změnu chování objektu na základě změny vnitřního stavu
9. Strategy - definuje skupinu algoritmů, zapouzdří je a docílí jejich vzájemné zaměnitelnosti
10. Visitor - reprezentuje operaci, která se provádí na prvcích objektu
11. Template method - definuje "skeleton" algoritmu

## 1. Observer

- Events that you can subscribe to.
- An event is a list of function references.
  - Subscription and unsubscription is handled with addition/removal of items in the list.

### 1.1. Observable Events

```python
class Event(list):
    def __call__(self, *args, **kwargs):
        for item in self:
            item(*args, **kwargs)


class Person:
    def __init__(self, name, address):
        self.name = name
        self.address = address
        # Define a list of events
        self.falls_ill = Event()

    def catch_a_cold(self):
      """Add a new event to the event list"""
        self.falls_ill(self.name, self.address)


def call_doctor(name, address):
    print(f'A doctor has been called to {address}')

if __name__ == '__main__':
    person = Person('Sherlock', '221B Baker St')

    person.falls_ill.append(lambda name, addr: print(f'{name} is ill'))

    person.falls_ill.append(call_doctor)

    person.catch_a_cold()

    # and you can remove subscriptions too
    person.falls_ill.remove(call_doctor)
    person.catch_a_cold()
```

### 1.2. Property Observers

```python
class Event(list):
  def __call__(self, *args, **kwargs):
    for item in self:
      item(*args, **kwargs)


class PropertyObservable:
  def __init__(self):
    self.property_changed = Event()


class Person(PropertyObservable):
  def __init__(self, age=0):
    super().__init__()
    self._age = age

  @property
  def age(self):
    return self._age

  @age.setter
  def age(self, value):
    if self._age == value:
      return
    self._age = value
    self.property_changed('age', value)


class TrafficAuthority:
  def __init__(self, person):
    self.person = person
    person.property_changed.append(self.person_changed)

  def person_changed(self, name, value):
    if name == 'age':
      if value < 18:
        print('Sorry, you still cannot drive')
      else:
        print('Okay, you can drive now')
        self.person.property_changed.remove(
          self.person_changed
        )


if __name__ == '__main__':
  p = Person()
  ta = TrafficAuthority(p)
  for age in range(16, 21):
    print(f'Setting age to {age}')
    p.age = age
```

### 1.3. Property dependencies

```python
class Event(list):
  def __call__(self, *args, **kwargs):
    for item in self:
      item(*args, **kwargs)


class PropertyObservable:
  def __init__(self):
    self.property_changed = Event()


class Person(PropertyObservable):
  def __init__(self, age=0):
    super().__init__()
    self._age = age

  @property
  def can_vote(self):
    return self._age >= 18

  @property
  def age(self):
    return self._age

  # @age.setter
  # def age(self, value):
  #   if self._age == value:
  #     return
  #   self._age = value
  #   self.property_changed('age', value)

  @age.setter
  def age(self, value):
    if self._age == value:
      return

    old_can_vote = self.can_vote

    self._age = value
    self.property_changed('age', value)

    if old_can_vote != self.can_vote:
      self.property_changed('can_vote', self.can_vote)


if __name__ == '__main__':
  def person_changed(name, value):
    if name == 'can_vote':
      print(f'Voting status changed to {value}')

  p = Person()
  p.property_changed.append(
    person_changed
  )

  for age in range(16, 21):
    print(f'Changing age to {age}')
    p.age = age
```

### 1.4. Full Example on Events

>Imagine a game where one or more rats can attack a player. Each individual rat has an initial attack value of `1`. However, rats attack as a swarm, so each rat's attack value is actually equal to the total number of rats in play.
>
>Given that a rat enters play through the initializer and leaves play (dies) via its `__exit__` method, please implement the `Game` and `Rat` classes so that, at any point in the game, the `attack` value of a `rat` is always consistent.

```python
from unittest import TestCase


class Event(list):
  """
  Event is a list of function references.

  Subscription and unsubscription is handled
  with addition/removal of items in the list.
  """
    def __call__(self, *args, **kwargs):
        for item in self:
            item(*args, **kwargs)


class Game:
    def __init__(self):
        self.rat_enters = Event()
        self.rat_dies = Event()
        self.notify_rat = Event()


class Rat:
    def __init__(self, game):
        self.game = game
        self.attack = 1

        # Append the functions to the event lists
        game.rat_enters.append(self.rat_enters)
        game.notify_rat.append(self.notify_rat)
        game.rat_dies.append(self.rat_dies)

        # Invokes Event.__call__ which in turn
        # invokes all the functions in the list
        # with the argument 'self' (which_rat)
        self.game.rat_enters(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.game.rat_dies(self)

    def rat_enters(self, which_rat):
        if which_rat != self:
            self.attack += 1
            self.game.notify_rat(which_rat)

    def notify_rat(self, which_rat):
        if which_rat == self:
            self.attack += 1

    def rat_dies(self, which_rat):
        self.attack -= 1


class Evaluate(TestCase):
    def test_single_rat(self):
        game = Game()
        rat = Rat(game)
        self.assertEqual(1, rat.attack)

    def test_two_rats(self):
        game = Game()
        rat = Rat(game)
        rat2 = Rat(game)
        self.assertEqual(2, rat.attack)
        self.assertEqual(2, rat2.attack)

    def test_three_rats_one_dies(self):
        game = Game()

        rat = Rat(game)
        self.assertEqual(1, rat.attack)

        rat2 = Rat(game)
        self.assertEqual(2, rat.attack)
        self.assertEqual(2, rat2.attack)

        with Rat(game) as rat3:
            self.assertEqual(3, rat.attack)
            self.assertEqual(3, rat2.attack)
            self.assertEqual(3, rat3.attack)

        self.assertEqual(2, rat.attack)
        self.assertEqual(2, rat2.attack)
```

## 2. Chain of Responsibility

- Given a hierarchy apply something to all objects.

### 2.1. Method Chain

```python
class Creature:
    def __init__(self, name, attack, defense):
        self.defense = defense
        self.attack = attack
        self.name = name

    def __str__(self):
        return f'{self.name} ({self.attack}/{self.defense})'


class CreatureModifier:
    def __init__(self, creature):
        self.creature = creature
        self.next_modifier = None

    def add_modifier(self, modifier):
        if self.next_modifier:
            self.next_modifier.add_modifier(modifier)
        else:
            self.next_modifier = modifier

    def handle(self):
        if self.next_modifier:
            self.next_modifier.handle()


class NoBonusesModifier(CreatureModifier):
    def handle(self):
        print('No bonuses for you!')


class DoubleAttackModifier(CreatureModifier):
    def handle(self):
        print(f'Doubling {self.creature.name}''s attack')
        self.creature.attack *= 2
        super().handle()


class IncreaseDefenseModifier(CreatureModifier):
    def handle(self):
        if self.creature.attack <= 2:
            print(f'Increasing {self.creature.name}''s defense')
            self.creature.defense += 1
        super().handle()


if __name__ == '__main__':
    goblin = Creature('Goblin', 1, 1)
    print(goblin)

    root = CreatureModifier(goblin)

    root.add_modifier(NoBonusesModifier(goblin))

    root.add_modifier(DoubleAttackModifier(goblin))
    root.add_modifier(DoubleAttackModifier(goblin))

    # no effect
    root.add_modifier(IncreaseDefenseModifier(goblin))

    root.handle()  # apply modifiers
    print(goblin)
```

### 2.2. Broker Chain

- This approach uses the Observer design pattern (events) and Command Query Separation (CQS)
  - *command* to *set* something
  - *query* to *get* something (without modifying anything)

```python
# 1) event broker
# 2) command-query separation (cqs)
# 3) observer
from abc import ABC
from enum import Enum


class Event(list):
    def __call__(self, *args, **kwargs):
        for item in self:
            item(*args, **kwargs)


class WhatToQuery(Enum):
    ATTACK = 1
    DEFENSE = 2


class Query:
    def __init__(self, creature_name, what_to_query, default_value):
        self.value = default_value  # bidirectional
        self.what_to_query = what_to_query # enum (attack/defense)
        self.creature_name = creature_name # sender


class Game:
    def __init__(self):
        self.queries = Event()

    def perform_query(self, sender, query):
        self.queries(sender, query)


class Creature:
    def __init__(self, game, name, attack, defense):
        self.initial_defense = defense
        self.initial_attack = attack
        self.name = name
        self.game = game

    @property
    def attack(self):
        q = Query(self.name, WhatToQuery.ATTACK, self.initial_attack)
        self.game.perform_query(self, q)
        return q.value

    @property
    def defense(self):
        q = Query(self.name, WhatToQuery.DEFENSE, self.initial_attack)
        self.game.perform_query(self, q)
        return q.value

    def __str__(self):
        return f'{self.name} ({self.attack}/{self.defense})'


class CreatureModifier(ABC):
    def __init__(self, game, creature):
        self.creature = creature
        self.game = game
        # New modifier is added into the game event list
        self.game.queries.append(self.handle)

    def handle(self, sender, query):
        pass

    def __enter__(self):
      """Start of the 'with' block"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
      """End of the 'with' block, removes the modifier"""
        self.game.queries.remove(self.handle)


class DoubleAttackModifier(CreatureModifier):
    def handle(self, sender, query):
        if (sender.name == self.creature.name and
                query.what_to_query == WhatToQuery.ATTACK):
            query.value *= 2


class IncreaseDefenseModifier(CreatureModifier):
    def handle(self, sender, query):
        if (sender.name == self.creature.name and
                query.what_to_query == WhatToQuery.DEFENSE):
            query.value += 3


if __name__ == '__main__':
    game = Game()
    goblin = Creature(game, 'Strong Goblin', 2, 2)
    print(goblin)

    with DoubleAttackModifier(game, goblin):
        print(goblin)
        with IncreaseDefenseModifier(game, goblin):
            print(goblin)

    print(goblin)
```

### 2.3. Simpler Broker Chain

- Simplified solution without unsubscription.

```python
import unittest
from abc import ABC
from enum import Enum

# creature removal (unsubscription) ignored in this exercise solution

class Creature(ABC):
    def __init__(self, game, attack, defense):
        self.initial_defense = defense
        self.initial_attack = attack
        self.game = game

    @property
    def attack(self): pass

    @property
    def defense(self): pass

    def query(self, source, query): pass


class WhatToQuery(Enum):
    ATTACK = 1
    DEFENSE = 2


class Goblin(Creature):

    def __init__(self, game, attack=1, defense=1):
        super().__init__(game, attack, defense)

    @property
    def attack(self):
        q = Query(self.initial_attack, WhatToQuery.ATTACK)
        for c in self.game.creatures:
            c.query(self, q)
        return q.value

    @property
    def defense(self):
        q = Query(self.initial_defense, WhatToQuery.DEFENSE)
        for c in self.game.creatures:
            c.query(self, q)
        return q.value

    def query(self, source, query):
        if self != source and query.what_to_query == WhatToQuery.DEFENSE:
            query.value += 1


class GoblinKing(Goblin):

    def __init__(self, game):
        super().__init__(game, 3, 3)

    def query(self, source, query):
        if self != source and query.what_to_query == WhatToQuery.ATTACK:
            query.value += 1
        else:
            super().query(source, query)


class Query:
    def __init__(self, initial_value, what_to_query):
        self.what_to_query = what_to_query
        self.value = initial_value

class Game:
    def __init__(self):
        self.creatures = []


class FirstTestSuite(unittest.TestCase):
    def test(self):
        game = Game()
        goblin = Goblin(game)
        game.creatures.append(goblin)

        self.assertEqual(1, goblin.attack)
        self.assertEqual(1, goblin.defense)

        goblin2 = Goblin(game)
        game.creatures.append(goblin2)

        self.assertEqual(1, goblin.attack)
        self.assertEqual(2, goblin.defense)

        goblin3 = GoblinKing(game)
        game.creatures.append(goblin3)

        self.assertEqual(2, goblin.attack)
        self.assertEqual(3, goblin.defense)
```

- My stupid approach to the example above not utilizing the Chain of Responsibility design pattern:

```python
from abc import ABC

class Creature(ABC):
    def __init__(self, game, name, attack, defense):
        self.game = game
        self.initial_defense = defense
        self.initial_attack = attack
        self.defense = defense
        self.attack = attack
        self.name = name

    def __str__(self):
        return '{} ({}/{})'.format(self.name, self.attack, self.defense)


class Goblin(Creature):
    def __init__(self, game, attack=1, defense=1):
        Creature.__init__(self, game, 'Goblin', attack, defense)
        
        
class GoblinKing(Goblin):
    def __init__(self, game, attack=3, defense=3):
        Creature.__init__(self, game, 'Goblin King', attack, defense)


class CreatureList(list):
        def append(self, value):
            super().append(value)  # Call the original append method
            defense_modifier = 0
            attack_modifier = 0
            n_goblins = 0
            for creature in self:
                if isinstance(creature, Goblin):
                    defense_modifier += 1
                    n_goblins += 1
                if isinstance(creature, GoblinKing):
                    attack_modifier += 1
            if n_goblins > 0:
                defense_modifier -= 1
            for creature in self:
                if isinstance(creature, Goblin) and\
                    not isinstance(creature, GoblinKing):
                    creature.defense = creature.initial_defense + defense_modifier
                    creature.attack = creature.initial_attack + attack_modifier   


class Game:
    def __init__(self):
        self.creatures = CreatureList()
```

## 3. Command

<img src="figures/command.png" alt="" style="width: 500px;">

Místo implementace spousty různých tlačítek budeme implementovat několik **tříd příkazů** pro všechny možné operace a propojíme je s konkrétními tlačítky v závislosti nad zamýšleným chováním tlačítek.

## 4. Iterator

- má dvě metody `get_next` a `has_next`
- jednotné rozhraní pro průchod heterogenních kolekcí
- konkrétní iterátor je úzce spjatý s datovou strukturou, jejíž průchod chceme implementovat
  - přistupuje k interním proměnným dané datové struktury
  - jak lze přistoupit k interní proměnné kolekce přes zapouzdření? Např. v C# mají kolekce vnořenou třídu Enumerable

## 5. Template Method

- Define an algorithm at a high level in parent class.
- Define abstract methods/properties.
- **Inherit** the algorithm class, providing necessary overrides.
- Notice that while *Template Method* defines an algorithm via *inheritance*, *Strategy* uses *composition*.

<img src="figures/template-method.png" alt="template-method" width="300px">

```python
from abc import ABC


class Game(ABC):

    def __init__(self, number_of_players):
        self.number_of_players = number_of_players
        self.current_player = 0

    def run(self):
        self.start()
        while not self.have_winner:
            self.take_turn()
        print(f'Player {self.winning_player} wins!')

    def start(self): pass

    @property
    def have_winner(self): pass

    def take_turn(self): pass

    @property
    def winning_player(self): pass


class Chess(Game):
    def __init__(self):
        super().__init__(2)
        self.max_turns = 10
        self.turn = 1

    def start(self):
        print(f'Starting a game of chess with {self.number_of_players} players.')

    @property
    def have_winner(self):
        return self.turn == self.max_turns

    def take_turn(self):
        print(f'Turn {self.turn} taken by player {self.current_player}')
        self.turn += 1
        self.current_player = 1 - self.current_player

    @property
    def winning_player(self):
        return self.current_player


if __name__ == '__main__':
    chess = Chess()
    chess.run()
```

## 6. Strategy

- Define an algorithm at a high level and define the interface that every strategy must follow.
- Provide dynamic **composition** of strategies in the resulting object.
- Notice that while *Template Method* defines an algorithm via *inheritance*, *Strategy* uses *composition*.

```python
import math
import cmath
from unittest import TestCase
from abc import ABC


class DiscriminantStrategy(ABC):
    def calculate_discriminant(self, a, b, c):
        pass


class OrdinaryDiscriminantStrategy(DiscriminantStrategy):
    def calculate_discriminant(self, a, b, c):
        return b*b - 4*a*c


class RealDiscriminantStrategy(DiscriminantStrategy):
    def calculate_discriminant(self, a, b, c):
        result = b*b-4*a*c
        return result if result >= 0 else float('nan')


class QuadraticEquationSolver:
    def __init__(self, strategy):
        self.strategy = strategy

    def solve(self, a, b, c):
        """ Returns a pair of complex (!) values """
        disc = complex(self.strategy.calculate_discriminant(a, b, c), 0)
        root_disc = cmath.sqrt(disc)
        return (
            (-b + root_disc) / (2 * a),
            (-b - root_disc) / (2 * a)
        )
```