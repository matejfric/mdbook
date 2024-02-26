# MATD

- [1. Vyhledávání pomocí konečného deterministického automatu](#1-vyhledávání-pomocí-konečného-deterministického-automatu)

## 1. Vyhledávání pomocí konečného deterministického automatu

>Deterministický konečný automat (DKA/DFA) je pětice $(Q, \Sigma, \delta, q_0, F)$, kde
>
>- $Q$ je neprázdná konečná množina stavů,
>- $\Sigma$ je abeceda (neprázdná konečná množina symbolů),
>- $\delta : Q \times \Sigma \rightarrow Q$ je přechodová funkce,
>- $q_0 \in Q$ je počáteční stav,
>- $F \subseteq Q$ je množina přijímajících stavů.

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
