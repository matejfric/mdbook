# 1. Behavioral Patterns

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
- [3. Iterator](#3-iterator)
- [4. Template Method](#4-template-method)
- [5. Strategy](#5-strategy)
- [6. Command](#6-command)
  - [6.1. Composite Command (Macro)](#61-composite-command-macro)
- [7. Interpreter](#7-interpreter)
  - [7.1. Lexing and Parsing](#71-lexing-and-parsing)
- [8. Memento](#8-memento)
- [9. Mediator](#9-mediator)
- [10. Visitor](#10-visitor)

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

## 3. Iterator

> Iterator is a class that facilitates the traversal of a data structure.

- Traditional iterator has two methods:
  - `get_next -> item`
  - `has_next -> bool`
- In Python:
  - `__iter__()` expose the iterator,
  - `__next__()` get the next element,
  - `raise StopIteration` when there are no more elements.
- Unified interface for traversal of a (heterogenic) collection.
- Single Responsibility Principle (SRP) / Separation of Concerns (SoC).
- Stateful iterator cannot be recursive.
- For example: [Tree traversal](https://en.wikipedia.org/wiki/Tree_traversal).

<details><summary> Code example: Binary tree in-order traversal </summary>

```python
{{#include src/behavioral/iterator/tree_traversal.py}}
```

</details>

<details><summary> Code example: Binary tree pre-order traversal </summary>

```python
{{#include src/behavioral/iterator/preorder_traversal.py}}
```

</details>

## 4. Template Method

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

## 5. Strategy

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

## 6. Command

> Command is an object which represents an instruction to perform a particular **action** and contains all the necessary information for the action to be taken.

- Ordinary statements are perishable.
  - One cannot undo a member assignment.
  - One cannot directly serialize a sequence of actions (calls, macro).
- GUI actions are usually implemented by commands (clicking on buttons, undo/redo), macro recording and more...
- How can we create an object that represents an operation?

<details><summary> Code example: Bank account with undo </summary>

```python
from abc import ABC
from enum import Enum


class BankAccount:
    OVERDRAFT_LIMIT = -500

    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited {amount}, balance={self.balance}")

    def withdraw(self, amount):
        if self.balance - amount < self.OVERDRAFT_LIMIT:
            return False
        self.balance -= amount
        print(f"Withdrew {amount}, balance={self.balance}")
        return True

    def __str__(self):
        return f"Balance={self.balance}"


class Command(ABC):
    def invoke(self):
        pass

    def undo(self):
        pass


class BankAccountCommand(Command):
    def __init__(self, account, action, amount):
        self.account = account
        self.amount = amount
        self.action = action
        self.success = None
        self.logs = []  # Keep track of successful actions (Event Sourcing).

    class Action(Enum):
        DEPOSIT = 0
        WITHDRAW = 1

    def invoke(self):
        if self.action == self.Action.DEPOSIT:
            self.account.deposit(self.amount)
            self.success = True
        elif self.action == self.Action.WITHDRAW:
            self.success = self.account.withdraw(self.amount)

        # Add to logs
        if self.success:
            self.logs.append(self.action)

    def undo(self):
        """We no longer have to rely on success"""
        if not self.logs:
            return
        most_recent_user_action = self.logs.pop()
        if most_recent_user_action == self.Action.WITHDRAW:
            self.account.deposit(self.amount)
            self.action = self.Action.DEPOSIT
        elif most_recent_user_action == self.Action.DEPOSIT:
            self.account.withdraw(self.amount)
            self.action = self.Action.WITHDRAW


if __name__ == "__main__":
    ba = BankAccount()
    print("> Init")
    print(ba)

    print("> Add +100")
    cmd = BankAccountCommand(ba, BankAccountCommand.Action.DEPOSIT, 100)
    cmd.invoke()
    print(ba)

    print("> Undo")
    cmd.undo()
    print(ba)

    # Fixed broken undo
    print("> Fixed: Undo once again")
    cmd.undo()
    print(ba)

    # But now we cycle thorugh
    print("> Wee! We have the entire history of you")
    cmd.undo()
    print(ba)

    # Even if we go too far
    print("> We don't go beyond what we know")
    cmd.undo()
    print(ba)

    print("> Withdraw 500")
    illegal_cmd = BankAccountCommand(ba, BankAccountCommand.Action.WITHDRAW, 500)
    illegal_cmd.invoke()
    print(ba)

    print("> Withdraw too much")
    illegal_cmd = BankAccountCommand(ba, BankAccountCommand.Action.WITHDRAW, 5000)
    illegal_cmd.invoke()
    print(ba)
```

</details>

### 6.1. Composite Command (Macro)

How to group multiple commands?

<details><summary> Code example: Transfer between bank accounts </summary>

```python
# Composite Command a.k.a. Macro
# also: Composite design pattern ;)

import unittest
from abc import ABC, abstractmethod
from enum import Enum


class BankAccount:
    OVERDRAFT_LIMIT = -500

    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f'Deposited {amount}, balance = {self.balance}')

    def withdraw(self, amount):
        if self.balance - amount >= BankAccount.OVERDRAFT_LIMIT:
            self.balance -= amount
            print(f'Withdrew {amount}, balance = {self.balance}')
            return True
        return False

    def __str__(self):
        return f'Balance = {self.balance}'


class Command(ABC):
    def __init__(self):
        self.success = False

    def invoke(self):
        pass

    def undo(self):
        pass


class BankAccountCommand(Command):
    def __init__(self, account, action, amount):
        super().__init__()
        self.amount = amount
        self.action = action
        self.account = account

    class Action(Enum):
        DEPOSIT = 0
        WITHDRAW = 1

    def invoke(self):
        if self.action == self.Action.DEPOSIT:
            self.account.deposit(self.amount)
            self.success = True
        elif self.action == self.Action.WITHDRAW:
            self.success = self.account.withdraw(self.amount)

    def undo(self):
        if not self.success:
            return
        # strictly speaking this is not correct
        # (you don't undo a deposit by withdrawing)
        # but it works for this demo, so...
        if self.action == self.Action.DEPOSIT:
            self.account.withdraw(self.amount)
        elif self.action == self.Action.WITHDRAW:
            self.account.deposit(self.amount)


# try using this before using MoneyTransferCommand!
class CompositeBankAccountCommand(Command, list):
    def __init__(self, items=[]):
        super().__init__()
        for i in items:
            self.append(i)

    def invoke(self):
        for x in self:
            x.invoke()

    def undo(self):
        for x in reversed(self):
            x.undo()


class MoneyTransferCommand(CompositeBankAccountCommand):
    def __init__(self, from_acct, to_acct, amount):
        super().__init__([
            BankAccountCommand(from_acct,
                               BankAccountCommand.Action.WITHDRAW,
                               amount),
            BankAccountCommand(to_acct,
                               BankAccountCommand.Action.DEPOSIT,
                               amount)])

    def invoke(self):
        ok = True
        for cmd in self:
            if ok:
                cmd.invoke()
                ok = cmd.success
            else:
                cmd.success = False
        self.success = ok


class TestSuite(unittest.TestCase):
    def test_composite_deposit(self):
        print("\ntest_composite_deposit")
        ba = BankAccount()
        deposit1 = BankAccountCommand(ba, BankAccountCommand.Action.DEPOSIT, 1000)
        deposit2 = BankAccountCommand(ba, BankAccountCommand.Action.DEPOSIT, 1000)
        composite = CompositeBankAccountCommand([deposit1, deposit2])
        composite.invoke()
        print(ba)
        composite.undo()
        print(ba)

    def test_transfer_fail(self):
        print("\ntest_transfer_fail")
        ba1 = BankAccount(100)
        ba2 = BankAccount()

        # composite isn't so good because of failure
        amount = 1000  # try 1000: no transactions should happen
        wc = BankAccountCommand(ba1, BankAccountCommand.Action.WITHDRAW, amount)
        dc = BankAccountCommand(ba2, BankAccountCommand.Action.DEPOSIT, amount)

        transfer = CompositeBankAccountCommand([wc, dc])

        transfer.invoke()
        print('ba1:', ba1, 'ba2:', ba2)  # end up in incorrect state
        transfer.undo()
        print('ba1:', ba1, 'ba2:', ba2)

    def test_better_tranfer(self):
        print("\ntest_better_tranfer")
        ba1 = BankAccount(100)
        ba2 = BankAccount()

        amount = 100

        transfer = MoneyTransferCommand(ba1, ba2, amount)
        transfer.invoke()
        print('ba1:', ba1, 'ba2:', ba2)
        print('success:', transfer.success)
        transfer.undo()
        print('ba1:', ba1, 'ba2:', ba2)
        print('success:', transfer.success)
        with self.assertRaises(Exception):
            # this should fail (multiple undo is not implemented!)
            transfer.undo()  
            print('ba1:', ba1, 'ba2:', ba2)
            print('success:', transfer.success)

if __name__ == '__main__':
    unittest.main()
```

</details>

## 7. Interpreter

> Interpreter is a component that processes structured text data by turning it into separate lexical tokens *(lexing)* and then interpreting sequences of said tokens *(parsing)*.

Examples:

- programming language compilers, interpreters, and IDEs
- HTML, XML
- RegEx
- Numerical expressions (1+2*3)

### 7.1. Lexing and Parsing

1. Lex the input into tokens.
2. Parse the tokens into a datastructure (tree).
3. Evaluate the parsed tokens.

<details><summary> Code example: Lexing and parsing of a binary numerical expression </summary>

```python
from enum import Enum


class Token:
    class Type(Enum):
        INTEGER = 0
        PLUS = 1
        MINUS = 2
        LPAREN = 3
        RPAREN = 4

    def __init__(self, type, text):
        self.type = type
        self.text = text

    def __str__(self):
        return f'`{self.text}`'


def lex(input):
    result = []

    i = 0
    while i < len(input):
        if input[i] == '+':
            result.append(Token(Token.Type.PLUS, '+'))
        elif input[i] == '-':
            result.append(Token(Token.Type.MINUS, '-'))
        elif input[i] == '(':
            result.append(Token(Token.Type.LPAREN, '('))
        elif input[i] == ')':
            result.append(Token(Token.Type.RPAREN, ')'))
        else:  # must be a number
            digits = [input[i]]
            for j in range(i + 1, len(input)):
                if input[j].isdigit():
                    digits.append(input[j])
                    i += 1
                else:
                    result.append(Token(Token.Type.INTEGER,
                                        ''.join(digits)))
                    break
        i += 1

    return result


# ↑↑↑ lexing ↑↑↑

# ↓↓↓ parsing ↓↓↓

class Integer:
    def __init__(self, value):
        self.value = value


class BinaryOperation:
    class Type(Enum):
        ADDITION = 0
        SUBTRACTION = 1

    def __init__(self):
        self.type = None
        self.left = None
        self.right = None

    @property
    def value(self):
        if self.type == self.Type.ADDITION:
            return self.left.value + self.right.value
        elif self.type == self.Type.SUBTRACTION:
            return self.left.value - self.right.value


def parse(tokens):
    result = BinaryOperation()
    have_lhs = False
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == Token.Type.INTEGER:
            integer = Integer(int(token.text))
            if not have_lhs:
                result.left = integer
                have_lhs = True
            else:
                result.right = integer
        elif token.type == Token.Type.PLUS:
            result.type = BinaryOperation.Type.ADDITION
        elif token.type == Token.Type.MINUS:
            result.type = BinaryOperation.Type.SUBTRACTION
        elif token.type == Token.Type.LPAREN:  # note: no if >for RPAREN
            j = i
            while j < len(tokens):
                if tokens[j].type == Token.Type.RPAREN:
                    break
                j += 1
            # preprocess subexpression
            subexpression = tokens[i + 1:j]
            element = parse(subexpression)
            if not have_lhs:
                result.left = element
                have_lhs = True
            else:
                result.right = element
            i = j  # advance
        i += 1
    return result

def eval(input):
    tokens = lex(input)
    print(' '.join(map(str, tokens)))

    parsed = parse(tokens)
    print(f'{input} = {parsed.value}')

if __name__ == '__main__':
    eval('(13+4)-(12+1)')
    eval('1+(3-4)')

    # This won't work (ternary expression).
    eval('1+2+(3-4)')
```

</details>

Different example:

<details><summary> Code example: Lexing and parsing of plus, minus numerical expressions /summary>

```python
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
                    raise ValueError(f"Variables with more than one letter are not upported. Got `{''.join(alphas)}`.")
                if input[i-1] in self.variables:
                    result.append(Token(Token.Type.INTEGER, str(self.variables[inputi-1]])))
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

```

</details>

## 8. Memento

> **Memento** is a token/handle class representing the system state (typically without methods). Memento lets us *roll back* to the state when the token was generated.

- Keep a memento of an object's state to return to that state.
- Memento can be used to implement **undo/redo** operations.

<details><summary> Code example: Bank account snapshot </summary>

```python
{{#include src/behavioral/memento/memento.py}}
```

</details>

<details><summary> Code example: Undo/redo </summary>

```python
{{#include src/behavioral/memento/undo_redo.py}}
```

</details>

<details><summary> Code example: TokenMachine and tricky references in Python </summary>

```python
{{#include src/behavioral/memento/token_machine.py}}
```

</details>

## 9. Mediator

> Mediator is a component that facilitates communication between other components without them necessarily being aware of each other or having direct access (reference) to each other.

- For example:
  - player in a MMO game,
  - participants in a chat room.
- Create the mediator and have each object in the system refer to it (typically, the mediator is passed as an argument during instantiation of objects).
- Mediator engages in bi-directional communication with its connected components.
  - Components have methods the mediator can call.
  - This is usually implemented via event processing (e.g., [RxPY](https://github.com/ReactiveX/RxPY)).

<details><summary> Code example: Simple mediator with events </summary>

Our system has any number of instances of `Participant` classes. Each `Participant` has a `value` integer attribute, initially zero.

A participant can `say()` a particular `value`, which is broadcast to all other participants. At this point in time, every other participant is obliged to increase their `value` by the `value` being broadcast.

Example:

- Two participants start with values `0` and `0` respectively.
- Participant 1 broadcasts the value `3`. We now have Participant 1 `value=0`, Participant 2 `value=3`
- Participant 2 broadcasts the value `2`. We now have Participant 1 `value=2`, Participant 2 `value=3`

```python
{{#include src/behavioral/mediator/mediator.py}}
```

</details>

<details><summary> Code example: Chat room </summary>

```python
{{#include src/behavioral/mediator/chat_room.py}}
```

</details>

<details><summary> Code example: Football match with events </summary>

```python
{{#include src/behavioral/mediator/chat_room.py}}
```

</details>

## 10. Visitor

> Visitor is a component that knows how to traverse a data structure composed of (possibly related) types.

Motivation: How to define a new operation on an entire class hierarchy?

<details><summary> Code example: Intrusive visitor </summary>

- Intrusively add functionality to all classes.
- Breaks *Open Closed Principle (OCP)*. Intrusive visitor requires to modify the entire hierarchy.
- What is more likely? A new type or a new operation?

```python
{{#include src/behavioral/visitor/intrusive.py}}
```

</details>

<details><summary> Code example: Reflective visitor </summary>

- Still breaks *Open Closed Principle (OCP)*, because new types require $M×N$ modifications.
- Single Responsibility Principle (SRP) / Separation of Concerns (SoC).

```python
{{#include src/behavioral/visitor/reflective.py}}
```

</details>

<details><summary> Code example: Dynamic visitor </summary>

```python
{{#include src/behavioral/visitor/classic.py}}
```

</details>

<details><summary> Code example: Dynamic visitor refined </summary>

You are asked to implement a visitor called `ExpressionPrinter` for printing different mathematical expressions. The range of expressions covers addition and multiplication. Put round brackets around addition operations, but not multiplication ones!

Notice that there is no need for an `Expression` base class due to duck typing.

```python
{{#include src/behavioral/visitor/classic2.py}}
```

</details>
