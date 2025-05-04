# Matematické základy informatiky

- [1. Výpočetní složitost algoritmů. Techniky analýzy výpočetní složitosti algoritmů: analýza rekurzivních algoritmů, amortizovaná analýza, složitost algoritmů v průměrném případě](#1-výpočetní-složitost-algoritmů-techniky-analýzy-výpočetní-složitosti-algoritmů-analýza-rekurzivních-algoritmů-amortizovaná-analýza-složitost-algoritmů-v-průměrném-případě)
- [2. Matematické modely algoritmů – Turingovy stroje a stroje RAM. Algoritmicky nerozhodnutelné problémy](#2-matematické-modely-algoritmů--turingovy-stroje-a-stroje-ram-algoritmicky-nerozhodnutelné-problémy)
- [3. Třídy složitosti problémů. Třída PTIME a NPTIME, NP-úplné problémy. Další třídy složitosti (PSPACE, EXPTIME, EXPSPACE, polynomiální hierarchie, NLOGSPACE, LOGSPACE, ...)](#3-třídy-složitosti-problémů-třída-ptime-a-nptime-np-úplné-problémy-další-třídy-složitosti-pspace-exptime-expspace-polynomiální-hierarchie-nlogspace-logspace-)
- [4. Výpočetní modely pro paralelní a distribuované algoritmy. Výpočetní složitost paralelních algoritmů. Komunikační složitost](#4-výpočetní-modely-pro-paralelní-a-distribuované-algoritmy-výpočetní-složitost-paralelních-algoritmů-komunikační-složitost)
- [5. Jazyk predikátové logiky prvního řádu. Práce s kvantifikátory a ekvivalentní transformace formulí](#5-jazyk-predikátové-logiky-prvního-řádu-práce-s-kvantifikátory-a-ekvivalentní-transformace-formulí)
- [6. Pojem relace, operace s relacemi, vlastnosti binárních homogenních relací. Relace ekvivalence a relace uspořádání a jejich aplikace](#6-pojem-relace-operace-s-relacemi-vlastnosti-binárních-homogenních-relací-relace-ekvivalence-a-relace-uspořádání-a-jejich-aplikace)
- [7. Pojem operace a obecný pojem algebra. Algebry s jednou a dvěma binárními operacemi](#7-pojem-operace-a-obecný-pojem-algebra-algebry-s-jednou-a-dvěma-binárními-operacemi)
  - [7.1. Algebry s jednou binární operací](#71-algebry-s-jednou-binární-operací)
  - [7.2. Algebry se dvěma binárními operacemi](#72-algebry-se-dvěma-binárními-operacemi)
- [8. FCA – formální kontext, formální koncept, konceptuální svazy](#8-fca--formální-kontext-formální-koncept-konceptuální-svazy)
- [9. Asociační pravidla, hledání často se opakujících množin položek](#9-asociační-pravidla-hledání-často-se-opakujících-množin-položek)
- [10. Metrické a topologické prostory – metriky a podobnosti. Jejich aplikace](#10-metrické-a-topologické-prostory--metriky-a-podobnosti-jejich-aplikace)
  - [10.1. Metriky](#101-metriky)
  - [10.2. Podobnosti](#102-podobnosti)
  - [10.3. Topologické prostory](#103-topologické-prostory)
- [11. Shlukování. Typy shlukování, metody pro určení kvality shlukování, aplikace shlukování](#11-shlukování-typy-shlukování-metody-pro-určení-kvality-shlukování-aplikace-shlukování)
- [12. Náhodná veličina. Základní typy náhodných veličin. Funkce určující rozdělení náhodných veličin](#12-náhodná-veličina-základní-typy-náhodných-veličin-funkce-určující-rozdělení-náhodných-veličin)
- [13. Vybraná rozdělení diskrétní a spojité náhodné veličiny. Binomické, hypergeometrické, negativně binomické, Poissonovo, exponenciální, Weibullovo, normální rozdělení](#13-vybraná-rozdělení-diskrétní-a-spojité-náhodné-veličiny-binomické-hypergeometrické-negativně-binomické-poissonovo-exponenciální-weibullovo-normální-rozdělení)
- [14. Popisná statistika. Číselné charakteristiky a vizualizace kategoriálních a kvantitativních proměnných](#14-popisná-statistika-číselné-charakteristiky-a-vizualizace-kategoriálních-a-kvantitativních-proměnných)
  - [14.1. Číselné charakteristiky](#141-číselné-charakteristiky)
  - [14.2. Vizualizace](#142-vizualizace)
  - [14.3. Identifikace odlehlých hodnot](#143-identifikace-odlehlých-hodnot)
- [15. Metody statistické indukce. Intervalové odhady. Princip testování hypotéz](#15-metody-statistické-indukce-intervalové-odhady-princip-testování-hypotéz)
  - [15.1. Testování hypotéz](#151-testování-hypotéz)

## 1. Výpočetní složitost algoritmů. Techniky analýzy výpočetní složitosti algoritmů: analýza rekurzivních algoritmů, amortizovaná analýza, složitost algoritmů v průměrném případě

## 2. Matematické modely algoritmů – Turingovy stroje a stroje RAM. Algoritmicky nerozhodnutelné problémy

## 3. Třídy složitosti problémů. Třída PTIME a NPTIME, NP-úplné problémy. Další třídy složitosti (PSPACE, EXPTIME, EXPSPACE, polynomiální hierarchie, NLOGSPACE, LOGSPACE, ...)

## 4. Výpočetní modely pro paralelní a distribuované algoritmy. Výpočetní složitost paralelních algoritmů. Komunikační složitost

## 5. Jazyk predikátové logiky prvního řádu. Práce s kvantifikátory a ekvivalentní transformace formulí

Predikátová logika prvního řádu *(First-Order Logic, FOL)* je rozšíření výrokové logiky.

Existují výroky, které nelze vyjádřit pomocí výrokové logiky. Např.

- `Každý student rozumí FOL.`
- `Každé sudé celé číslo je dělitelné dvěma.`

Jazyk predikátové logiky prvního řádu obsahuje:

|Termy|Formule|
|---|---|
| **Konstantní symboly** `42`, `Bob` | **Atomické formule** `Knows(Bob, FOL)`, `P(x,y)` |
| **Proměnné** `x` | **Logické spojky** $\neg, \lor, \land, \implies, \iff$ |
| **Funkce** ($n$-ární) `Sum(x,3)` | **Kvantifikátory** $\forall, \exist$ |

- Univerzální kvantifikátor $\forall$:
  - $\forall x\, P(x)$ rozumíme $P(A)\land P(B)\land \ldots$
- Existenciální kvantifikátor $\exist$:
  - $\exist x\, P(x)$ rozumíme $P(A)\lor P(B)\lor \ldots$
- Vlastnosti kvantifikátorů:
  - De Morgan: $\neg(\forall x\, P(x)) \Leftrightarrow \exist x\, \neg P(x)$
  - De Morgan: $\neg(\exist x\, P(x)) \Leftrightarrow \forall x\, \neg P(x)$
  - Záleží na pořadí kvantifikátorů. Např. $\forall x\, \exist y\, P(x,y)$ není to stejné jako $\exist y\, \forall x\, P(x,y)$.
  - $\forall x\,\forall y\,P(x,y)\Leftrightarrow \forall y\,\forall x\,P(x,y)$
  - $\exists x\,\exists y\,P(x,y)\Leftrightarrow \exists y\,\exists x\,P(x,y)$
  - $\forall x\,P(x)\land \forall x\,Q(x)\Leftrightarrow \forall x\,(P(x)\land Q(x))$
  - $\exists x\,P(x)\lor \exists x\,Q(x)\Leftrightarrow \exists x\,(P(x)\lor Q(x))$
- **Vázaná proměnná** je taková proměnná, která se vyskytuje vedle kvantifikátoru $(\forall,\exist)$. Proměnné, které nejsou vázané nazýváme **volné**.
- Formuli nazveme **otevřenou** právě tehdy, když obsahuje alespoň jednu volnou proměnnou. V opačném případě nazveme formuli **uzavřenou**.
- **Valuace** je jedno konkrétní přiřazení prvků univerza proměnným.

<details><summary> Příklady </summary>

|Výrok|Formule FOL|
|---|---|
|*Každý* student rozumí FOL.|$\forall x\, \text{Student}(x) \implies \text{Knows}(x, \text{FOL})$|
|*Nějaký* student rozumí FOL.|$\exist x\, \text{Student}(x) \land \text{Knows}(x, \text{FOL})$|
|Každé sudé celé číslo je dělitelné dvěma.| $\forall x\, \text{Even}(x) \implies \text{Div}(x, 2)$|
|Existuje kurz, který absolvoval každý student.| $\exist y\,\text{Course}(y) \land \left[ \forall x\, \text{Student}(x) \implies \text{Took}(x,y)\right ]$|

</details>

## 6. Pojem relace, operace s relacemi, vlastnosti binárních homogenních relací. Relace ekvivalence a relace uspořádání a jejich aplikace

> ($n$-ární) **relace** $R$ je podmnožina kartézského součinu množin $R\subseteq A_1 \times \ldots \times A_n = \set{(a_1,\ldots,a_n)\mid (\forall i = 1,\ldots,n): a_i\in A_i}$.

Dělení podle arity:

- unární $R_1\subseteq A_1$
- binární $R_2 \subseteq A_1 \times A_2$
- ternární
- $n$-ární

Relace je *homogenní*, pokud $A_1 = A_2 = \ldots = A_n$. Jinak je *heterogenní*.

> Binární relace $R\subseteq X,Y$ je **zobrazení** z $X$ do $Y$ právě tehdy, když
>
> $$(\forall x\in X)(\forall y_1,y_2\in Y)\colon [xRy_1 \land xRy_2] \Rightarrow y_1=y_2$$

Operace s relacemi:

> Zobrazení $R$ kartézského součinu množin $A_1,\ldots,A_n$ do množiny $B$ nazveme $n$-ární **operací**.
>
> $$R\colon A_1,\ldots,A_n \to B$$

- unární: $|x|,\frac{1}{x},(-x)$, reflexivní uzávěr $\mathrm{Re}(R)$
- binární: $x\cap y, x+y, \langle x,y\rangle$ (skalární součin vektorů je heterogenní)
- $n$-ární: aritmetický průměr, maximum

> Vlastnosti binárních homogenních relací $(\forall x,y,z):$
>
> | Vlastnost           | Výraz                                                      | poznámka |
> |---------------------|-------------------------------------------------------------|----------|
> | **RE**flexivita     | $xRx$                                                      |  $\mathrm{diag}(\mathbb{A})=\mathbf{1}$        |
> | **IR**eflexivita    | $(x,x)\notin R$                                            |  $\mathrm{diag}(\mathbb{A})=\mathbf{0}$         |
> | **SY**metrie        | $xRy \Rightarrow yRx$                                     |  $\mathbb{A}^\top=\mathbb{A}$        |
> | **AS**ymetrie       | $(x,y) \in R \Rightarrow (y,x) \notin R$                  | žádné smyčky a opačně orientované hrany         |
> | **AN**tisymetrie    | $xRy \wedge yRx \Rightarrow x=y$                          | žádné opačně orientované hrany         |
> | **TR**anzitivita    | $xRy \wedge yRz \Rightarrow xRz$                          | každá cesta délky 2 má i zkratku         |
> | **ÚP**lnost         | $xRy \vee yRx$                                             | po zanedbání orientace $K_n$ se všemi smyčkami          |
> | **SO**uvislost      | $x\neq y \Rightarrow xRy \vee yRx$                        | jako ÚP, ale nemusí mít všechny smyčky         |
>
> **AS**ymetrie $\Rightarrow$ **AN**tisymetrie.

**Relace ekvivalence** je binární homogenní relace, která splňuje vlastnosti **ReSyTr**.

**Relace uspořádání** je binární homogenní relace, která splňuje vlastnosti:

| Uspořádání | Částečné | Úplné |
|:---:|---|---|
| Ostré $\,\,<$ | **IrAsTr** | (+SO) |
| Neostré $\,\,\leq$ | **ReAnTr** | (+ÚP) |

## 7. Pojem operace a obecný pojem algebra. Algebry s jednou a dvěma binárními operacemi

> Binární relace $R\subseteq X,Y$ je **zobrazení** z $X$ do $Y$ právě tehdy, když
>
> $$(\forall x\in X)(\forall y_1,y_2\in Y)\colon [xRy_1 \land xRy_2] \Rightarrow y_1=y_2$$

<!--  -->

> Zobrazení $R$ kartézského součinu množin $A_1,\ldots,A_n$ do množiny $B$ nazveme $n$-ární **operací**.
>
> $$R\colon A_1,\ldots,A_n \to B$$

- unární: $|x|,\frac{1}{x},(-x)$, reflexivní uzávěr $\mathrm{Re}(R)$
- binární: $x\cap y, x+y, \langle x,y\rangle$ (skalární součin vektorů je heterogenní)
- $n$-ární: aritmetický průměr, maximum

> **Algebraický systém** je uspořádaná dvojice nosičů a operací $\left(\set{A_i},\set{f_j}\right)$.

Vlastnosti binárních operací. Buď $\circ$ binární homogenní operace na mže $A$.

| Vlastnost                                          | Výraz                                                                       |
| -------------------------------------------------- | --------------------------------------------------------------------------- |
| **UZ**avřenost                                     | $(\forall a,b\in A)(\exists c \in A): a\circ b = c$                         |
| **AS**ociativita                                   | $(\forall a,b,c\in A):\ a\circ (b\circ c) = (a\circ b)\circ c$              |
| **E**xistence **J**ednotkového (neutrálního) prvku | $(\forall x\in A)(\exists e\in A):\ a\circ e = e\circ a = a$                |
| **E**xistence **N**ulového (agresivního) prvku     | $(\forall x\in A)(\exists n\in A):\ a\circ n = n\circ a = n$                |
| **E**xistence **I**nverzního prvku                 | $(\forall a\in A)(\exists a^{-1}\in A):\ a\circ a^{-1}= a^{-1}\circ  a = e$ |
| **KO**mutativita                                   | $(\forall a,b\in A):\ a\circ b= b\circ  a$                                  |
| **ID**empotentnost                                 | $(\forall a\in A):\ a\circ a= a$                                            |

- **Grupoid** je dvojice $(A,\circ)$, kde $A$ je neprázdná množina a $\circ$ je binární operace **uzavřená** na $A$ **(UZ)**
- **Pologrupa** je asociativní grupoid (**+AS**).
- **Monoid** je pologrupa s jednotkovým prvkem(**+EJ**).
- **Grupa** je monoid s inverzními prvky ke každému prvku (**+IN**).
- **Abelova grupa** je komutativní grupa (**+KO**).

### 7.1. Algebry s jednou binární operací

- *(užzasejeiko**t)*

| Algebra        | Popis                                                                                   | Vlastnost       |
|----------------|------------------------------------------------------------------------------------------|-----------------|
| **Grupoid**     | Dvojice $(A,\circ)$, kde $A$ je neprázdná množina a $\circ$ je binární operace uzavřená na $A$ | **UZ**          |
| **Pologrupa**   | Asociativní grupoid                                                                      | **+ AS**     |
| **Monoid**      | Pologrupa s jednotkovým prvkem                                                           | **+ EJ**|
| **Grupa**       | Monoid s inverzními prvky ke každému prvku                                               | **+ EI** |
| **Abelova grupa** | Komutativní grupa                                                                       | **+ KO** |

- Nechť $(A,\circ)$. Pokud existuje jednotkový/nulový prvek, pak je právě jeden.
- Nechť $(A,\circ)$, $\circ$ je AS, a platí EJ na $A$. Pak $(\forall a \in A): [\exists ! \, a^{-1} \vee \nexists\, a^{-1}]$.
- Nechť $(A,\circ)$ je grupa, pak $(\forall a \in A): \exists ! \, a^{-1}.$
- Pokud v Cayleyho tabulce existuje řádek nebo sloupec s neunikátními hodnotami, pak se nemůže jednat o grupu.
- Neutrální prvek je tam, kde se zkopíruje záhlaví Cayleyho tabulky.

### 7.2. Algebry se dvěma binárními operacemi

- **Dělitelé nuly** jsou $(a\neq0 \wedge b\neq0): a\cdot b = 0$.

| Algebra        | Popis                                                                                   |
|--|--|
| **Okruh** *(Ring)* | Trojice $(R,+,\cdot)$, kde $(R,+)$ je *Abelova grupa,* $(R,\cdot)$ *pologrupa* a platí *distributivní zákony*. |
| **Unitární okruh** | Okruh, kde $(R,\cdot)$ je monoid. |
| **Obor** | Unitární okruh bez *dělitelů nuly*. |
| **Obor integrity** | Komutativní obor. |
| **Těleso** | $(R,+)$ je *Abelova grupa* a $(R\setminus\{0\},\cdot)$ je grupa |
| **Galoisovo těleso** | Konečné těleso. |
| **Pole** | Těleso, kde násobení je komutativní, tj. $(R\setminus\{0\},\cdot)$ je Abelova grupa. |

- Jednotkový prvek aditivní grupy *okruhu* je nulovým prvkem jeho multiplikativního *monoidu*. Tento prvek nazýváme nulovým prvkem okruhu.
- Těleso $\Rightarrow$ obor integrity.
- Konečný obor integrity $\Rightarrow$ konečné pole.

<details><summary> Příklady </summary>

- $(\mathbb{Z}_6,+,\cdot)$ je okruh, ale není to obor integrity, protože ex. dělitelé nuly

    $$\overline{2_6}\cdot\overline{3_6}=\overline{6_6}=\overline{0_6}.$$

- $(\mathbb{Z},+,\cdot)$ je obor integrity, ale není to těleso, protože neex. některé inverze

    $$2\in\mathbb{Z}\quad\wedge\quad 2^{-1}=\dfrac{1}{2}\notin\mathbb{Z}.$$

- $(\mathbb{Z}_p,+,\cdot)$, kde $p$ je prvočíslo je konečné (Galoisovo) těleso.

</details>

## 8. FCA – formální kontext, formální koncept, konceptuální svazy

**Formální konceptuální analýza (FCA)** je metoda, která pracuje s binárními tabulkovými daty - **formálním kontextem** - které popisují relaci mezi objekty a atributy.

**Formální kontext** je uspořádaná trojice $(X,Y,I)$, kde

- $X$ je množina **objektů**,
- $Y$ je množina **atributů**,
- $I\subseteq X\times Y$ je binární **relace**.

Skutečnost, že **objekt** $x\in X$ **má atribut** $y\in Y$, značíme $(x,y)\in I$.

Každý kontext indukuje **šipkové zobrazení** $^\uparrow\colon2^{X}\rightarrow2^{Y}$ a $^\downarrow\colon2^{Y}\rightarrow2^{X}$ definované:

- $A\subseteq X, A^\uparrow=\set{y\in Y \mid (\forall x\in A): (x,y)\in I}$,
- $B\subseteq Y, B^\downarrow=\set{x\in X \mid (\forall y\in B): (x,y)\in I}$.

**Formální koncept** formálního kontextu $(X,Y,I)$ definujeme jako dvojici $(A,B)$, kde

- $X\supseteq \red{A}=B^\downarrow$ nazýváme **extent** a
- $Y\supseteq B=\red{A}^\uparrow$ nazýváme **intent**.

**Konceptuální svaz** je formální kontext $(X,Y,I)$ spolu s relací '$\leq$', s.t.

$$(A_1,B_1)\leq(A_2,B_2)\Leftrightarrow A_1 \subseteq A_2 \Leftrightarrow B_1 \supseteq B_2.$$

<details><summary> Příklad FCA </summary>

Pomocí FCA lze analyzovat hierarchii objektů a atributů, např. vodní plochy:

- Zápisem $x_{a,b,c}$ rozumíme $\set{x_a,x_b,x_c}$.

| "Vodní plochy" | Přírodní $(y_1)$ |  Stojatá $(y_2)$ | Tekoucí $(y_3)$|
|---|---|---|---|
| **Jezero** $(x_1)$ | 1 | 1 | 0 |
| **Rybník** $(x_2)$| 0 | 1 | 0 |
| **Řeka** $(x_3)$| 1 | 0 | 1 |
| **Kanál** $(x_4)$| 0 | 0 | 1 |

| Extent | Intent |
|---|---|
| $e_1=y_1^{\downarrow} = x_{1,3}$ | $i_1=x_{1,3}^{\uparrow} = y_1$ |
| $e_2=y_2^{\downarrow} = x_{1,2}$ | $i_2=x_{1,2}^{\uparrow} = y_2$ |
| $e_3=y_3^{\downarrow} = x_{3,4}$ | $i_3=x_{3,4}^{\uparrow} = y_3$ |
| $e_4= e_1\cap e_2=x_1$ | $i_4=x_{1}^{\uparrow}=y_{1,2}$ |
| $e_5= e_1\cap e_3=x_3$ | $i_5=x_{1}^{\uparrow}=y_{1,2}$ |
| $e_6= e_2\cap e_3=\emptyset$ | $i_6=\emptyset^{\uparrow}=y_{1,2,3}$ |
| $e_7= x_{1,2,3,4}$ | $i_7=x_{1,2,3,4}^{\uparrow}=\emptyset$ |

"Konceptuální svaz vytvářím podle velikosti extentu od spodu nahoru:"

```mermaid
flowchart TB
    e7["$$x_{1,2,3,4} \mid \emptyset$$"] --> e2["$$x_{1,2} \mid y_2$$"]
    e7 --> e3["$$x_{3,4} \mid y_3$$"]
    e7 --> e1["$$x_{1,3} \mid y_1$$"]
    
    e1 --> e5["$$x_{3} \mid y_{1,3}$$"]
    e2 --> e4["$$x_{1} \mid y_{1,2}$$"]
    e1 --> e4
    e3 --> e5
    e4 --> e6["$$\emptyset \mid y_{1,2,3}$$"]
    e5 --> e6
```

</details>

## 9. Asociační pravidla, hledání často se opakujících množin položek

[Asociační pravidla](../azd/azd.md#2-hledání-častých-vzorů-v-datech-základní-principy-metody-varianty-implementace).

## 10. Metrické a topologické prostory – metriky a podobnosti. Jejich aplikace

Aplikace např. ve *strojovém učení* - shlukování, klasifikace, ...

### 10.1. Metriky

>**Metrický prostor** je dvojice $(\mathcal{M},\rho)$, kde $\mathcal{M}\neq\emptyset$ a $\rho\colon\mathcal{M}\times\mathcal{M}\rightarrow \R$ je metrika, která splňuje $\forall x,y,z\in \mathcal{M}$ axiomy:
>
> 1. **Totožnost** $\boxed{\rho(x,y) = 0 \iff x=y.}$
> 2. **Symetrie** $\boxed{\rho(x,y) = \rho(y,x).}$
> 3. **Trojúhelníková nerovnost** $\boxed{\rho(x,z) \leq \rho(x,y)+\rho(y,z).}$

Metrika je *nezáporná*: $2\rho(x,y)=\rho(x,y)+\rho(y,x)\geq \rho(x,x) = 0$.

| Metrika                         | Vzorec                                                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Minkowského** $(p)$ norma     | $\rho_p(x,y)=\left(\sum\limits_{i=1}^n\lvert x_i - y_i \rvert^p\right)^{\frac{1}{p}}, \quad p \in \langle1,+\infty)$ |
| **Manhattanská** $(L_1)$ norma  | $\rho_1(x,y)=\sum\limits_{i=1}^n\lvert x_i - y_i \rvert$                                                             |
| **Eukleidova** $(L_2)$ norma    | $\rho_2(x,y)=\sqrt{\sum\limits_{i=1}^n(x_i - y_i)^2}$                                                                |
| **Čebyševova** $(\infty)$ norma | $\rho_{\infty}(x,y)=\max\limits_{i=1...n} \lbrace \lvert x_i - y_i \rvert \rbrace$                                   |

- $\rho_1 \geq \rho_2 \geq \dots \geq \rho_{\infty}$

| Metrika                        | Vzorec                                                     |
| ------------------------------ | ---------------------------------------------------------- |
| **Hammingova**                 | Počet rozdílných pozic mezi řetězci stejné délky           |
| **Longest Common Subsequence** | Nejmenší počet operací `vkládání` a `mazání`               |
| **Levenshteinova**             | Nejmenší počet operací `vkládání`, `mazání` a `substituce` |

### 10.2. Podobnosti

> **Podobnost** je zobrazení $\text{sim}\colon\mathcal{V}\times\mathcal{V}\rightarrow \R$, kde $\mathcal{V}\neq\emptyset$, které splňuje $\forall x,y\in \mathcal{V}$ axiomy:
>
> 1. **Nezápornost** $\boxed{\text{sim}(x,y) \geq 0.}$
> 2. **Symetrie** $\boxed{\text{sim}(x,y) = \text{sim}(y,x).}$
> 3. **Totožnost** $\boxed{\text{sim}(x,y) \leq \text{sim}(x,x)}$ a $\boxed{\text{sim}(x,y) = \text{sim}(x,x) \iff x=y}$ (bod je nejvíce podobný sám sobě).

| Podobnost                        | Vzorec                                                     |
| ------------------------------ | ---------------------------------------------------------- |
| **Kosinová podobnost**                 | $\text{sim}_C(x,y)=\dfrac{\langle x,y \rangle}{\lVert{x}\rVert\lVert{y}\rVert}$           |
| **Jaccardova podobnost** | $\text{sim}_J(A,B)=\dfrac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$               |

### 10.3. Topologické prostory

> Na množině $X$ je dána **topologie pomocí otevřených množin**, je-li dán systém podmnožin $\mathcal{T} \subseteq 2^X$ takový, že:
>
> - Prázdná a celá množina $\boxed{\emptyset\in \mathcal{T} \wedge X \in \mathcal{T}}$
> - Uzavřenost vůči *průniku dvojic* $\boxed{A,B \in \mathcal{T} \Rightarrow A \cap B\in \mathcal{T}}$
> - Uzavřenost vůči *sjednocení* $\boxed{(\forall A_i \in \mathcal{T}):\,\bigcup\limits_{i} A_i \in \mathcal{T}}$
>
> Pak $(X,\mathcal{T})$ tvoří topologický prostor (TP), kde prvky topologie $\mathcal{T}$ jsou otevřené množiny.

<!--  -->

> Duálně, na množině $X$ je dána **topologie pomocí uzavřených množin**, je-li dán systém podmnožin $\mathcal{T}' \subseteq 2^X$ takový, že:
>
> - Prázdná a celá množina $\boxed{\emptyset\in \mathcal{T}' \wedge X \in \mathcal{T}'}$
> - Uzavřenost vůči *sjednocení dvojic* $\boxed{A,B \in \mathcal{T}' \Rightarrow A \cup B\in \mathcal{T}'}$
> - Uzavřenost vůči *průniku* $\boxed{(\forall A_i \in \mathcal{T}'):\,\bigcap\limits_{i} A_i \in \mathcal{T}'}$
>
> Pak $(X,\mathcal{T}')$ tvoří topologický prostor (TP), kde prvky topologie $\mathcal{T}'$ jsou uzavřené množiny.

- Duální topologii vytvořím $\boxed{\mathcal{T}' = X \setminus \mathcal{T}.}$
- Trivialní topologie $\boxed{\mathcal{T} = \lbrace \emptyset, X \rbrace}$.
- Každý metrický prostor je topologický prostor.

<details><summary> Příklad </summary>

Dvojice $(\lbrace a,b,c \rbrace, \mathcal{T})$, kde $\mathcal{T} = \lbrace \emptyset, \lbrace a,b \rbrace, \lbrace a,c \rbrace,\lbrace a,b,c \rbrace\rbrace$ **není** topologickým prostorem, protože $\lbrace a,b \rbrace \cap \lbrace a,c \rbrace = \lbrace a \rbrace\notin \mathcal{T}$.

</details>

## 11. Shlukování. Typy shlukování, metody pro určení kvality shlukování, aplikace shlukování

[Shlukování](../azd/azd.md#3-shlukovací-metody-shlukování-pomocí-reprezentantů-hierarchické-shlukování-shlukování-na-základě-hustoty-validace-shluků-pokročilé-metody-shlukování-clarans-birch-cure).

## 12. Náhodná veličina. Základní typy náhodných veličin. Funkce určující rozdělení náhodných veličin

> **Náhodný pokus** je experiment, jehož výsledek nelze s jistotou předpovědět.
>
> **Náhodný jev** je tvrzení o výsledku náhodného pokusu, které lze po jeho provedení rozhodnout.
>
> Buď $\Omega=\{\omega_1,\omega_2,\ldots\}$ množina všech možných výsledků *náhodného pokusu*. **Náhodná veličina** $X$ je funkce $X:\Omega\to\mathbb{R}$, která každému elementárnímu jevu $\omega_i$ přiřazuje reálné číslo $X(\omega_i)$.
>
> Tzn. náhodná veličina je číselné vyjádření výsledku náhodného pokusu.

<img src="figures/random-variable.png" alt="random-variable" width="400px">

> Náhodná veličina $X$ má *diskrétní rozdělení pravděpodobnosti* (je diskrétní) právě tehdy, když nabývá *spočetně mnoha hodnot*.
>
> Diskrétní náhodná veličina $X$ s distribuční funkcí $F_X(t)$ je charakterizována *pravděpodobnostní funkcí* $P(X=x_i)$, pro kterou platí
>
> $$F_X(t)=\sum\limits_{x_i<t}P(X=x_i)$$
>
> Distribuční funkce $F_X(t)=P(X\leq t)$ je *neklesající*, *zprava spojitá* a má nejvýše *spočetně mnoho bodů nespojitosti*.

<img src="figures/cdf.drawio.svg" alt="cdf.drawio" width="700px">

> Náhodná veličina $X$ má *spojité rozdělení pravděpodobnosti* (je spojitá) právě tehdy, když má *spojitou distribuční funkci*.
>
> Spojitá náhodná veličina $X$ s distribuční funkcí $F_X(t)$ je charakterizována *hustotou pravděpodobnosti* $f_X(t)$, pro kterou platí
>
> $$F_X(t)=\int\limits_{-\infty}^t f_X(x)dx$$
>
> - $(\forall x\in\mathbb{R})\colon P(X=x)=0$ (žádné skoky)
> - $P(X < x) = P(X \leq x)$
> - $\int\limits_{-\infty}^{+\infty} f_X(x)dx = 1$
> - $F_X(x)$ je vždy spojitá, ale $f_X$ spojitá být nemusí (např. uniformní rozdělení)
> - $(\forall x\text{ kde }f_X(x)\text{ je spojitá})\colon f_X(x)=F_X'(x)$
> - $\mathbb{E}(X)=\int\limits_{-\infty}^{+\infty} xf_X(x)dx$

## 13. Vybraná rozdělení diskrétní a spojité náhodné veličiny. Binomické, hypergeometrické, negativně binomické, Poissonovo, exponenciální, Weibullovo, normální rozdělení

V posloupnosti *bernoulliovských pokusů* jsou NV *vzájemně nezávislé* a mají *alternativní rozdělení*.

$$X\sim \text{A}(p)\iff p_X(x)=\begin{cases}
            p & \text{if } x=1\\
            1-p & \text{if } x=0\\
            0 & \text{otherwise}
        \end{cases}$$

---

**Binomická** NV reprezentuje **počet úspěchů** v posloupnosti $n$ bernoulliovských pokusů s pravděpodobností úspěchu $p$.

$$X\sim \text{Bi}(n,p)\iff p_X(x)=\begin{cases}
    \binom{n}{x}p^x(1-p)^{n-x} & \text{pro } x=0,1,\ldots,n \\
    0 & \text{jinak}
\end{cases}$$

---

**Geometrická** NV představuje počet bernoulliovských pokusů do prvního úspěchu. Nastane $n-1$ neúspěchů, $(1-p)^{n-1}$, a jeden úspěch $p$.

$$
\begin{equation*}
        X\sim Ge(p)\iff p_X(x)=\begin{cases}
            p\cdot (1-p)^{x-1} & x\in\N\\
            0 & \text{otherwise}
        \end{cases}
\end{equation*}
$$

---

**Hypergeometrická** NV reprezentuje počet prvků se sledovanou vlastností ve výběru $n$ prvků. Např. v krabici je 5 zelených a 25 červených kuliček. Náhodně vybereme 10 kuliček. Jaká je pravděpodobnost, že vybereme právě 3 zelené?

$$X\sim \text{H}(N,K,n)\sim\text{H}(30,5,10) \iff p_X(x)=\frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}$$

$$P_X(X=3)=p_X(3)=\dfrac{\binom{5}{3}\binom{30-5}{5-3}}{\binom{30}{10}}$$

---

**Negativně binomická** NV reprezentuje **počet pokusů** potřebných k dosažení $k$ úspěchů v posloupnosti bernoulliovských pokusů s pravděpodobností úspěchu $p$.

$$X\sim \text{NB}(k,p)\iff p_X(x)=\begin{cases}
     \binom{x-1}{k-1}p^k(1-p)^{x-k} & \text{pro } x=k,k+1,\ldots \\
     0 & \text{jinak}
 \end{cases}$$

---

**Poissonova** NV reprezentuje **počet událostí** v daném uzavřeném časovém intervalu $t$ (resp. objemu, ploše,...) v Poissonově procesu s *intenzitou* $\lambda$.

$$X\sim \text{Po}(\lambda t)\iff p_X(x)=\frac{\lambda^x e^{-\lambda t}}{x!}$$

---

**Exponenciální** NV reprezentuje **čas do výskytu 1. události** v Poissonově procesu s *intenzitou* $\lambda$.

$$X\sim \exp(\lambda)\iff f_X(x)=\begin{cases}
    \lambda e^{-\lambda x} & \text{pro } x\geq 0 \\
    0 & \text{pro } x<0
\end{cases}$$

---

**Weibullovo rozdělení** pravděpodobnosti je zobecnění exponenciálního rozdělení. Má dva parametry měřítko $\theta$ a tvar $\beta$. Narozdíl od exponenciálního rozdělení nepředpokládá *konstantní intenzitu* výskytu sledované události.

- využití v analýze spolehlivosti, poruch, přežití atd.
- Pro nezápornou SNV definujeme rizikovou funkci $\mathrm{risk}(t)=\dfrac{f(t)}{1-F(t)}$

---

**Normální rozdělení** je charakterizováno dvěma parametry $\mu$ a $\sigma$:

$$\boxed{X\sim\mathcal{N}(\mu,\sigma^2)}$$

- $\mathbb{E}(X)=\mu, \mathbb{D}(X)=\sigma^2$.
- Normované normální rozdělení $Z\sim\mathcal{N}(0,1)$.
- Buď $X\sim\mathcal{N}(\mu,\sigma^2)$, pak $Z=\dfrac{X-\mu}{\sigma}\sim\mathcal{N}(0,1)$.

---

**Gamma** NV vyjadřuje dobu do výskytu $k$-té událostí v Poissonově procesu.

---

**Uniformní** rozdělení.

$$
\begin{equation*}
    X\sim U(\langle a,b\rangle)\iff f_X(x)=\begin{cases}
        \dfrac{1}{b-a} & x\in\langle a,b\rangle\\
        0 & \text{jinak}
    \end{cases}
\end{equation*}
$$

## 14. Popisná statistika. Číselné charakteristiky a vizualizace kategoriálních a kvantitativních proměnných

Popisná statistika se zabývá popisem a vizualizací dat **bez** provádění závěrů o populaci.

1. **Kategorické** proměnné - barva, pohlaví, název rostliny *(Iris)*
2. **Kvantitativní** proměnné - výška, váha, délka okvětních lístků

### 14.1. Číselné charakteristiky

1. Kvantitativní proměnné
   - **Střední hodnota** - průměrná hodnota
     - Pro DNV $X$ s pravděpodobnostní funkcí $p_X(x)$ je střední hodnota definována jako

        $$\mathbb{E}(X)=\sum\limits_{x\in\mathcal{X}}x\cdot p_X(x)$$

     - Pro SNV $X$ s hustotou pravděpodobnosti $f_X(x)$ je střední hodnota definována jako

        $$\mathbb{E}(X)=\int\limits_{-\infty}^{+\infty}x\cdot f_X(x)dx$$

   - **Medián** - prostřední hodnota
   - **Modus** - nejčastější hodnota, tj. $\max p_X(x)$, resp. $\max f_X(x)$
   - **$p$-kvantil** $x_p\in\mathbb{R}$ je číslo, pro které platí $\mathcal{P}(X\leq x_p)=p$
     - Pokud je $F_X$ rostoucí a spojitá, tak $x_p=F_X^{-1}(p)$
   - **Rozptyl** - průměrná kvadratická odchylka dat kolem střední hodnoty

      $$\mathbb{D}(X) = \mu_2' = \mathbb{E}((X-\mathbb{E}(X))^2)= \mathbb{E}(X^2)-(\mathbb{E}(X))^2$$

   - **Směrodatná odchylka** - odmocnina z rozptylu
   - **Variační rozpětí** - $R=x_{max}-x_{min}$
   - **Mezikvartilové rozpětí** - $\mathrm{IQR}=Q_3-Q_1 = x_{0.75}-x_{0.25}$
   - **Variační koeficient** pro nezáporné NV (čím vyšší, tím více rozptýlený soubor dat)

        $$\gamma(X)=\dfrac{\sigma(X)}{\mathbb{E}(X)}$$

   - **Šikmost**

        $$\mathrm{skew}(X)=\dfrac{\mu_3'}{\sigma^3}=\dfrac{\mathbb{E}((X-\mathbb{E}X)^3)}{\sigma^3}$$

        <img src="../pas/figures/skewness.png" alt="skewness" width="400px">

        $\mathrm{skew}(X)<2$ (negativně zešikmené), $\mathrm{skew}(X)\in\langle-2,2\rangle$, $\mathrm{skew}(X)>2$ (pozitivně zešikmené)

   - **Špičatost**

        $$\mathrm{kurt}(X)=\dfrac{\mu_4'}{\sigma^4}=\dfrac{\mathbb{E}((X-\mathbb{E}X)^4)}{\sigma^4}$$

        <img src="../pas/figures/kurtosis.png" alt="skewness" width="400px">

        $\mathrm{kurt}(X)<3$, $\mathrm{kurt}(X)=3$ a $\mathrm{kurt}(X)>3$

Pokud *šikmost* a *standardní špičatost* $(\mathrm{kurt}(X) - 3)$ leží v intervalu $\langle -2,2\rangle$, tak je rozdělení přibližně normální (empiricky).

1. Kategorické proměnné - *nominální ("třídy"), ordinální (pořadí)*
   - **Četnost/frekvence** - počet výskytů kategorie, tabulka četností
   - **Relativní četnost**
   - **Modus** - nejčastější kategorie

| Pohlaví | Absolutní četnost | Relativní četnost |
|---------|--------------------|--------------------|
| muž     | 66                 | 0.78               |
| žena    | 19                 | 0.22               |
| Celkem: | 85                 | 1.00               |

> **Čebyševova nerovnost**
>
> $(\forall k \in \mathbb{R}^+)\colon \boxed{\mathcal{P}(\mu - k\sigma < X < \mu + k\sigma) \geq 1 - \dfrac{1}{k^2},}$ kde $\mu$ je střední hodnota a $\sigma$ je směrodatná odchylka.
>
> <details><summary> Alternativní formulace Čebyševovy nerovnosti </summary>
>
> $$
 \begin{align*}
     \mathcal{P}(|X-\mu |\geq k\sigma ) &\leq {\frac {1}{k^{2}}}\\
     1 - \mathcal{P}(|X-\mu |< k\sigma ) &\leq {\frac {1}{k^{2}}}\\
     \mathcal{P}(|X-\mu |< k\sigma ) &\geq 1 - {\frac {1}{k^{2}}}\\
     \mathcal{P}(\mu - k\sigma < X < \mu + k\sigma ) &\geq 1 - \dfrac{1}{k^2}\\
 \end{align*}$$
>
> </details>

### 14.2. Vizualizace

1. Kategorické
    - sloupcový graf
2. Kvantitativní
    - histogram - ukazuje rozdělení
    - krabicový graf (boxplot)

<img src="../azd/figures/boxplot.png" alt="boxplot" width="400px">

- Nestačí uvést jen relativní četnost, ale i absolutní četnost. Koláčové grafy jsou většinou nevhodné, je lepší použít sloupcové grafy.
- U histogramu je třeba zvolit vhodnou šířku intervalů. Pokud porovnáváme histogramy, je třeba mít stejnou šířku intervalů.
- Popis os nesmí chybět a neměl by být redundantní.

<img src="figures/visualization.png" alt="visualization" width="650px">

### 14.3. Identifikace odlehlých hodnot

- Metoda vnitřních hradeb $\langle Q_1-1.5\cdot \mathrm{IQR}, Q_3 + 1.5\cdot \mathrm{IQR}\rangle$
- Metoda vnějších hradeb $\langle Q_1-3\cdot \mathrm{IQR}, Q_3 + 3\cdot \mathrm{IQR}\rangle$

## 15. Metody statistické indukce. Intervalové odhady. Princip testování hypotéz

**Statistická indukce** umožňuje stanovit vlastnosti populace na základě pozorování náhodného výběru. Vychází ze *zákonu velkých čísel* a *centrální limitní věty*.

Např. ve výběru 100 lidí bude průměr IQ 110. V jakém rozmezí a s jakou pravděpodobností se nachází průměr IQ populace?

**Výběrové charakteristiky:**

- **Náhodný výběr** $\mathbf{X}=(X_1, X_2, \ldots, X_n)$ je vektor náhodných veličin, které jsou *nezávislé* a mají *stejné rozdělení* (**i.i.d.**).

| Populační parametr       | Výběrová charakteristika          |
|--------------------------|-----------------------------------|
| Střední hodnota $\mu$      | Výběrový průměr $\overline{X}$             |
| Medián $x_{0.5}$          | Výběrový medián $\tilde{X}_{0.5}$           |
| Směrodatná odchylka $\sigma$  | Výběrová směrodatná odchylka $S$  |
| Pravděpodobnost $p$      | Relativní četnost $\hat{p}$           |

**Odhady parametrů populace:**

- bodový odhad
- intervalový odhad

| Popis                                    | Značení                                   | Interval |
|------------------------------------------|-------------------------------------------|--|
| Hladina významnosti                      | $\alpha$                                  | |
| Hladina spolehlivosti                    | $1 - \alpha$                              | |
| Oboustranný odhad                        | $P(T_D < \mu < T_H) = 1 - \alpha$         |$(T_D; T_H)$|
| Levostranný odhad                        | $P(T_D^* < \mu) = 1 - \alpha$             |$(T_D^*,+\infty)$|
| Pravostranný odhad                      | $P(\mu < T_H^*) = 1 - \alpha$             |$(-\infty, T_H^*)$|

### 15.1. Testování hypotéz

- **Statistická hypotéza** je libovolné tvrzení o rozdělení náhodné veličiny. Cílem testování hypotéz je rozhodnout, zda je daná hypotéza v souladu s pozorovanými daty.
- **Statistika** $T(\mathbf{X})$ je funkce náhodných veličin, a tedy je taky náhodnou veličinou (pokud je $T$ borelovská).
- **Pozorovaná hodnota** $t_{obs}(\mathbf{x})\in\mathbb{R}$, kde $\mathbf{x}=(x_1,\ldots,x_n)$ je konkrétní realizace náhodného výběru, je konkrétní realizací výběrové statistiky $T(\mathbf{X})$.
- Před testováním hypotéz odstraníme *odlehlé pozorování*.

> **Klasický test**
>
> 1. Formulujeme **nulovou** a **alternativní** hypotézu.
>      - Nulová hypotéza je tvrzení, které je vždy postaveno jako *nepřítomnost rozdílu* mezi sledovanými skupinami.
>      - Alternativní hypotéza je pak tvrzení, které popírá platnost nulové hypotézy. Přítomnost rozdílu.
> 2. Stanovíme **hladinu významnosti** $\alpha$ (pravděpodobnost chyby I. druhu).
>
> 3. Zvolíme tzv. testovou statistiku , tj. výběrovou charakteristiku, jejíž rozdělení závisí na testovaném parametru $\theta$. (Rozdělení testové statistiky za předpokladu platnosti nulové hypotézy nazýváme nulové rozdělení.)
> 4. Ověříme předpoklady testu.
> 5. Určíme kritický obor $W^*$:
>
> | Tvar alternativní hypotézy $H_A$ | Kritický obor $W^*$                                     |
> |-----------------------------------|---------------------------------------------------------|
> | $\theta < \theta_0$               | $(-\infty, x_{\alpha})$                                 |
> | $\theta > \theta_0$               | $(x_{1-\alpha}, \infty)$                                |
> | $\theta \neq \theta_0$            | $(-\infty, x_{\alpha/2}) \cup (x_{1-\alpha/2}, \infty)$ |
>
> 6. Na základě konkrétní realizace výběru určíme pozorovanou hodnotu $x_{\text{OBS}}$ testové statistiky.
> 7. Rozhodneme o výsledku testu:
>
> | Situace                    | Interpretace                                            |
> |----------------------------|---------------------------------------------------------|
> | $x_{\text{OBS}} \in W^*$   | Na hladině významnosti $\alpha$ zamítáme $H_0$ ve prospěch $H_A$. |
> | $x_{\text{OBS}} \notin W^*$ | Na hladině významnosti $\alpha$ nezamítáme $H_0$.        |
>
> 8. Místo kroků 5-7 v klasickém testu můžeme určit *p-hodnotu* $p$ a porovnat ji s hladinou významnosti $\alpha$:
>
> | Tvar alternativní hypotézy $H_A$  |                      $p$-hodnota                  |
> |-----------------------------------|----------------------------------------------------|
> | $\theta < \theta_0$               | $F_0(x_{\text{OBS}})$               |
> | $\theta > \theta_0$               | $1 - F_0(x_{\text{OBS}})$           |
> | $\theta \neq \theta_0$            | $2 \min\set{F_0(x_{\text{OBS}}), 1 - F_0(x_{\text{OBS}})}$ |
>
> 9. Rozhodnutí o výsledku testu. *"P-value is low, null hypothesis must go."*
>
> | $p$-hodnota                    | Interpretace                                           |
> |-----------------------------|--------------------------------------------------------|
> | $< \alpha$    | Na hladině významnosti $\alpha$ zamítáme $H_0$ ve prospěch $H_A$. |
> | $\geq \alpha$  | Na hladině významnosti $\alpha$ nezamítáme $H_0$.     |

<details><summary> Jednovýběrový t-test střední hodnoty </summary>

1. Odstraním odlehlá pozorování.
2. $H_0\colon \mu = c, c\in\mathbb{R}$
3. $H_1\colon \mu \neq c$
4. $\alpha=0.05$

```r
t.test(
    x,
    mu = 0,
    alternative = "two.sided",
    conf.level = 0.95
)
```

</details>

Korelace neimplikuje kauzalitu.

$$\rho _{X,Y}={\mathrm {cov} (X,Y) \over \sigma _{X}\sigma _{Y}}={\mathbb{E}((X-\mu _{X})(Y-\mu _{Y})) \over \sigma _{X}\sigma _{Y}}$$
