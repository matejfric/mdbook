# SOLID Design Principles

- [1. Single Responsibility Principle (SRP) or Separation of Concerns (SoC)](#1-single-responsibility-principle-srp-or-separation-of-concerns-soc)
- [2. Open-Close Principle (OCP)](#2-open-close-principle-ocp)
- [3. Liskov Substitution Principle (LSP)](#3-liskov-substitution-principle-lsp)
- [4. Interface Segregation Principle (ISP)](#4-interface-segregation-principle-isp)
- [5. Dependency Inversion Principle (DIP)](#5-dependency-inversion-principle-dip)

## 1. Single Responsibility Principle (SRP) or Separation of Concerns (SoC)

- a class should have its primary responsibility and it shouldn't take on other responsibilities
- anti-pattern - god object - one class that does all kinds of stuff
- e.g., `class Journal`
  - primary function: adding entries, removing entries, printing entries
  - secondary function: saving, loading, etc.   $\Rightarrow$ `class PersistenceManager`  

## 2. Open-Close Principle (OCP)

- open for extension, closed for modification
- industry **Specification** design pattern

## 3. Liskov Substitution Principle (LSP)

- Barbara Liskov (Turingova cena 2008)
- Pokud mám interface a nějakou hierarchii tříd, která implementuje tento interface, tak bych měl být vždy schopen nahradit předka potomkem bez omezení správnosti všech metod!

## 4. Interface Segregation Principle (ISP)

- Interface by neměl mít příliš velké množství metod. Je lepší takové rozhraní rozdělit na více rozhraní.
- YAGNI - You Ain't Going to Need It

```python
from abc import ABC, abstractmethod

# wrong - YAGNI

class Machine(ABC):
    @abstractmethod
    def print(self, document):
        raise NotImplementedError()

    @abstractmethod
    def fax(self, document):
        raise NotImplementedError()

    @abstractmethod
    def scan(self, document):
        raise NotImplementedError()
```

```python
# this is better

class Printer(ABC):
    @abstractmethod
    def print(self, document): pass


class Scanner(ABC):
    @abstractmethod
    def scan(self, document): pass
```

## 5. Dependency Inversion Principle (DIP)

- Instead of depending on a low level module directly, create an interface for the low level modules, and let the high level module depend on the interface.
