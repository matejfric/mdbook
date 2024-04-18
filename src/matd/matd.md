# MATD

- [1. Exact Pattern Matching](#1-exact-pattern-matching)
  - [1.1. Brute Force](#11-brute-force)
  - [1.2. Vyhledávání pomocí konečného deterministického automatu](#12-vyhledávání-pomocí-konečného-deterministického-automatu)
  - [1.3. Knuth-Morris-Pratt (KMP) Algorithm](#13-knuth-morris-pratt-kmp-algorithm)
  - [1.4. Boyer-Moore-Horspool Algorithm](#14-boyer-moore-horspool-algorithm)
  - [1.5. Aho-Corasick (AC) Algorithm](#15-aho-corasick-ac-algorithm)
- [2. Přibližné vyhledávání řetezců (Exact Pattern Matching)](#2-přibližné-vyhledávání-řetezců-exact-pattern-matching)
  - [2.1. NDFA](#21-ndfa)
- [3. Elias Gamma Coding](#3-elias-gamma-coding)
- [Vector Model](#vector-model)

## 1. Exact Pattern Matching

### 1.1. Brute Force

- $\mathcal{O}(m\cdot n)$
- Posun v textu o jednu pozici.

### 1.2. Vyhledávání pomocí konečného deterministického automatu

>Deterministický konečný automat (DKA/DFA) je pětice $(Q, \Sigma, \delta, q_0, F)$, kde
>
>- $Q$ je neprázdná konečná množina stavů,
>- $\Sigma$ je abeceda (neprázdná konečná množina symbolů),
>- $\delta : Q \times \Sigma \rightarrow Q$ je přechodová funkce,
>- $q_0 \in Q$ je počáteční stav,
>- $F \subseteq Q$ je množina přijímajících stavů.
>
>DFA má pouze konečnou paměť. Např. není schopný vyřešit **parity problem** (kontrola uzavřených závorek).
>
>Derivace regulárního výrazu:
><img src="figures/regex-derivative.png" alt="regex-derivative" width="400px">

Příklad `P="abab"`:

- prefixy: $Q=\{\varepsilon, a, ab, aba, abab\}$
- indexy prefixů: $I=\{0,1,2,3,4\}$

| $Q$           | $I$ | $a$ | $b$ | $\Sigma\setminus\{a,b\}$ |
|---------------|-----|-----|-----|--------------------------|
| $\varepsilon$ | 0   | 1   | 0   | 0                        |
| $a$           | 1   | 1   | 2   | 0                        |
| $ab$          | 2   | 3   | 0   | 0                        |
| $aba$         | 3   | 1   | 4   | 0                        |
| $abab$        | 4   | 3   | 0   | 0                        |

- Sloupce $a$, $b$, $\Sigma\setminus\{a,b\}$ jsou vstupy automatu.

### 1.3. Knuth-Morris-Pratt (KMP) Algorithm

- [KMP | Wiki](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)
- $\mathcal{O}(m+n)$
- Předzpracování vzorku - vytvoření tabulky pro posuny v textu.

### 1.4. Boyer-Moore-Horspool Algorithm

- $\mathcal{O}(m*n)$, ale $\Theta(m+n/m)$
- Předzpracování vzorku - vytvoření tabulky pro posuny v textu. Využívá znaky, které nejsou ve vzorku pro delší skoky. Shoda se kontroluje od konce.
- Např. $T=ab\underline{c}def$, $P=aa\underline{b}$
- Definuju si tabulku pro ASCII znaky, jejich pozici ve vzorku a odpovídající posuny.
- [Handbook of Exact String Matching](.lectures/Handbook_of_Exact_String_Matching_Algorithms.pdf) (str. 119)

1. Předzpracování bez posledního znaku $P$
   - `shift = |p| - i - 1`
2. Vyhledávání od konce $P$

### 1.5. Aho-Corasick (AC) Algorithm

- Efektivní vyhledávání více slov pomocí DFA.
- **Dictionary-matching** algorithm that locates elements of a finite set of strings (the "dictionary") within an input text.
- It matches all strings *simultaneously*.

<img src="figures/ac-algorithm2.png" alt="ac-algorithm" width="400px">
<img src="figures/ac-algorithm.png" alt="ac-algorithm" width="400px">

## 2. Přibližné vyhledávání řetezců (Exact Pattern Matching)

**Hamming**ova vzdálenost - počet pozic, kde se řetězce liší.

**Levenshtein**ova (editační) vzdálenost - nejmenší počet operací vkládání, mazání a substituce.

### 2.1. NDFA

<img src="figures/approx_ndfa.png" alt="approx_ndfa" width="600px">

$T=\text{survey}$

0. `<iterace>: [vstup] -> (<stav>, <pozice v patternu>)`
1. $t_0\colon (1,i)$
2. $t_1\colon s\rightarrow[(2,i+1), (9,i), (8,i)]$

## 3. Elias Gamma Coding

- Každé kladné celé číslo ve dvojkové soustavě začíná jedničkou.

Unární čísla:

$$
\begin{align*}
  u(1) &= 1\\
  u(2) &= 01\\
  u(3) &= 001\\
  &\vdots
\end{align*}
$$

Př. $4 \Rightarrow b(4) =100 \Rightarrow \lvert b(4) \rvert = 3 \Rightarrow u( \lvert b(4) \rvert -1)\Rightarrow \gamma(4)=00100$ (první tři bity jsou $u(3)$, další dva jsou $b(4)$ bez *leading* bitu)

## Vector Model

| TF | Doc1 | Doc2 | ... | DocN |
|--|--|--|--|--|
| term1 | 10 | 0 | ... |  3 |
| term2 | 15 | 0 | ... |  0 |
| term3 | 0 | 8 | ... |  7 |

$$\boxed{\mathrm{idf}(\mathrm{term}) = \log\dfrac{N}{\mathrm{df}(\mathrm{term})},}$$

kde $N$ je počet dokumentů v kolekci a $\mathrm{df}_{\mathrm{term}}$ je počet dokumentů obsahujících $\mathrm{term}$.

Pro dotaz $q$: $\mathrm{Score}(q,d)=\sum\limits_{t\in q} \mathrm{TF}_{d,t} \cdot \mathrm{idf}(t)$
