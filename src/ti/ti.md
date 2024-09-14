# Teoretická informatika (TI)

- [1. Automaty](#1-automaty)
  - [1.1. Deterministický konečný automat](#11-deterministický-konečný-automat)
  - [1.2. Nedeterministický konečný automat](#12-nedeterministický-konečný-automat)
  - [1.3. Zobecněný nedeterministický konečný automat](#13-zobecněný-nedeterministický-konečný-automat)
  - [1.4. Aplikace](#14-aplikace)
- [2. Bezkontextové jazyky](#2-bezkontextové-jazyky)
  - [2.1. Zásobníkový automat](#21-zásobníkový-automat)
  - [2.2. Turingův stroj](#22-turingův-stroj)
  - [2.3. Chomského hierarchie](#23-chomského-hierarchie)

**Algoritmus** — mechanický postup, jak něco spočítat. Algoritmy slouží k řešení různých problémů. Konkrétní vstup nějakého problému se nazývá **instance** problému.

V zadání problému musí být určeno:

- co je množinou možných vstupů,
- co je množinou možných výstupů,
- jaký je vztah mezi vstupy a výstupy.

Algoritmus **řeší** daný problém, pokud:

1. Se pro každý vstup po konečném počtu kroků zastaví.
2. Pro každý vstup vydá správný výstup.

**Korektnost algoritmu** — ověření toho, že daný algoritmus skutečně řeší daný problém.

**Výpočetní složitost algoritmu**:

- **Časová složitost** — jak závisí doba výpočtu na velikosti vstupu.
- **Paměťová (prostorová) složitost** — jak závisí množství použité paměti na velikosti vstupu.

> **(Formální) jazyk** $L$ v abecedě $\Sigma$ je libovolná podmnožina množiny > $\Sigma^*$, tj. $L\subseteq\Sigma^*$, kde
>
> - abeceda $\Sigma$ je neprázdná konečná množina symbolů,
> - slovo je konečná sekvence symbolů abecedy $\Sigma$,
> - jazyk je množina slov,
> - množina všech slov abecedy se označuje $\Sigma^*$.
>
> $L^+=L\cdot L^* = \bigcup\limits_{k\geq1}L^k$

## 1. Automaty

### 1.1. Deterministický konečný automat

>Deterministický konečný automat (DKA) je pětice $(Q, \Sigma, \delta, q_0, F)$, kde
>
>- $Q$ je neprázdná konečná množina stavů,
>- $\Sigma$ je abeceda (neprázdná konečná množina symbolů),
>- $\delta : Q \times \Sigma \rightarrow Q$ je přechodová funkce, tzn. dvojici (stav, symbol) přiřadí stav,
>- $q_0 \in Q$ je počáteční stav,
>- $F \subseteq Q$ je množina přijímajících stavů.
>
>DFA má pouze konečnou paměť. Např. není schopný vyřešit **parity problem** (kontrola uzavřených závorek).

### 1.2. Nedeterministický konečný automat

Nedeterministický konečný automat (NKA) se od DKA liší množinou počátečních stavů $I\subset Q$ a přechodovou funkcí $\delta : Q \times \Sigma \rightarrow \mathcal{P}(Q)$, kde $\mathcal{P}$ je potenční množina.

### 1.3. Zobecněný nedeterministický konečný automat

Zobecněný nedeterministický konečný automat (ZNKA) se od NKA liší jen přechodovou funkcí $\delta : Q \times (\Sigma\cup\{\varepsilon\}) \rightarrow \mathcal{P}(Q)$.

### 1.4. Aplikace

1. Převod konečného automatu na regulární výraz

    <img src="figures/dfa-to-regex.png" alt="dfa-to-regex" width="175px">

2. Zřetězení jazyků

    <img src="figures/lang-concat.png" alt="lang-concat" width="400px">

3. Iterace jazyků

    <img src="figures/lang-iter.png" alt="lang-iter" width="400px">

4. Sjednocení jazyků

    <img src="figures/lang-union.png" alt="lang-union" width="400px">

## 2. Bezkontextové jazyky

> Bezkontextová gramatika je definována jako uspořádaná čtveřice $G = (\Pi, \Sigma, S, P)$, kde:
>
> - $\Pi$ je konečná množina *neterminálních symbolů* (neterminálů),
> - $\Sigma$ je konečná množina *terminálních symbolů* (terminálů), přičemž $\Pi \cap \Sigma = \emptyset$,
> - $S \in \Pi$ je *počáteční* (startovací) *neterminál*,
> - $P$ je konečná množina *pravidel* typu $A \rightarrow β$, kde:
>   - $A$ je neterminál, tedy $A \in \Pi$,
>   - $β$ je řetězec složený z terminálů a neterminálů, tedy $β \in (\Pi \cup \Sigma)^*$.

Pokud je jazyk regulární, tak je bezkontextový.

### 2.1. Zásobníkový automat

> Zásobníkový automat (ZA) $M$ je definován jako šestice $M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0)$, kde:
>
> - $Q$ je konečná neprázdná množina *stavů*,
> - $\Sigma$ je konečná neprázdná množina *vstupních symbolů* (vstupní abeceda),
> - $\Gamma$ je konečná neprázdná množina *zásobníkových symbolů* (zásobníková abeceda),
> - $q_0 \in Q$ je *počáteční stav*,
> - $Z_0 \in \Gamma$ je *počáteční zásobníkový symbol* a
> - $\delta$ je zobrazení množiny $Q \times (\Sigma \cup \{\epsilon\}) \times \Gamma$ do množiny všech konečných podmnožin množiny $Q \times \Gamma^*$.

<img src="figures/stack-automaton.png" alt="stack-automaton" width="350px">

Alternativní zkrácený zápis:

<img src="figures/stack-automaton-2.png" alt="stack-automaton" width="125px">

<div class="warning">

**Ekvivalence bezkontextových gramatik a zásobníkových automatů.**

Ke každé bezkontextové gramtice $G$ lze sestrojit ekvivalentní (nedeterministický) zásobníkový automat. Navíc ke každému ZA lze sestrojit ekvivalentní bezkontextovou gramatiku.

</div>

### 2.2. Turingův stroj

Oproti zásobníkovému automatu umožňuje navíc:

- pohyb "hlavy" oběma směry,
- možnost zápisu na "pásku" na aktuální pozici "hlavy",
- "páska" je nekonečná.

Příklad *přechodové funkce*: $\boxed{\delta(q_1, b)=(q_2,x,+1)}$. Jsem ve stavu $g_1$ a na pásce je znak $b$. Přejdu do stavu $q_2$, přepíšu znak na pásce na $x$ a posunu se na pásce o jedno pole doprava.

**Church-Turingova teze.** Každý algoritmus lze realizovat Turingovým strojem.

### 2.3. Chomského hierarchie

<img src="figures/chomsky-hierarchy.png" alt="chomsky-hierarchy" width="250px">

- **Typ 0 - rekurzivně spočetné** jazyky:
  - obecné generativní gramatiky
  - Turingovy stroje (deterministické, nedeterministické)

- **Typ 1 - kontextové** jazyky:
  - kontextové gramatiky
  - nedeterministické lineárně omezené automaty

- **Typ 2 - bezkontextové** jazyky:
  - bezkontextové gramatiky
  - nedeterministické zásobníkové automaty

- **Typ 3 - regulární** jazyky:
  - regulární gramatiky
  - konečné automaty (deterministické, nedeterministické)
  - regulární výrazy
