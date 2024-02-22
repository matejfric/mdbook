# Operations Research

- [1. Motivation](#1-motivation)
- [2. Operations Research Lifecycle](#2-operations-research-lifecycle)
- [3. Simplex](#3-simplex)
  - [3.1. Examples](#31-examples)

> "The scientists were asked to do **research on** (military) **operations**" $\Rightarrow$ Operations Research.

## 1. Motivation

Consider a fleet of delivery trucks. How to plan production and routing in the most optimal way (maximize profit)?

<img src="figures/individual-carrier-routing.png" alt="individual-carrier-routing" width="300px">

"All models are wrong, but some are useful."

## 2. Operations Research Lifecycle

<img src="figures/operations-research-lifecycle.png" alt="operations-research-lifecycle" width="600px">

## 3. Simplex

### 3.1. Examples

<details><summary> Example in 2D </summary>

| Product          | Wood | Time | Profit |
|:----------------:|:----:|:----:|:------:|
| Table   ($x_1$)  | 10   |  5   | 180    |
| Wardrobe ($x_2$) | 20   | 4    | 200    |
| (Totals)         | 200  | 80   | ?      |

Maximize:
$$ 180 x_1 + 200 x_2 $$
subject to,
$$
\begin{align*}
    10 x_1 + 20 x_2 &\leq 200\\
    5 x_1 + 4 x_2 &\leq 80\\
    x_1 &\geq 0\\
    x_2 &\geq 0
\end{align*}
$$

Graphical solution using [Desmos](https://www.desmos.com/calculator):

<img src="figures/desmos.png" alt="desmos" width="600px">

- Red point - solution.
- Black points - boundary points.

If we added another column in the table ("material"), a new constraint would appear. If we added a new row ("product"), a new dimension would appear.

</details>

<details><summary> Example in 3D </summary>

| Product          | $m_1$ | $m_2$ | $m_3$ | Profit |
|:----------------:|:----:|:----:|:--------|:------:|
| $p_1$ $(x)$   | 2   |  4   |  2     | 4    |
| $p_2$ $(y)$ | 3   | 0    |  5      | 3    |
| $p_3$ $(z)$ | 2   | 3    |  0      | 6    |
| (Totals)   | 440  | 470   | 430     |?      |

Maximize:
$$ 4x + 3y + 6z $$
subject to,
$$
\begin{align*}
    2x + 3y + 2z &\leq 440\\
    4x + 3z &\leq 470\\
    2x + 5y &\leq 430\\
    x &\geq 0\\
    y &\geq 0\\
    z &\geq 0
\end{align*}
$$

</details>

<details><summary> Solved example in 2D </summary>

| Product          | M  | P | Profit |
|:----------------:|:----:|:----:|:------:|
| A ($x$)  | 1.5   | 2.5   | 50    |
| B ($y$)  | 2.5   | 1.5    | 40    |
| (Totals)         | 300  | 240   | ?      |

Maximize:
$$ 50 x + 40 y $$
subject to,
$$
\begin{align*}
    1.5 x + 2.5 y &\leq 300\\
    2.5 x + 1.5 y &\leq 240\\
    x &\geq 0\\
    y &\geq 0
\end{align*}
$$

$$
\begin{align*}
    3x + 5y &= 600\\
    5x + 3y &= 480\\
    \hline\\
    25y - 9y &= 5\cdot 600 - 3\cdot 480\\
    16y &= 1560\\
    y &= 97.5\\
    \hline\\
    x &= \dfrac{480 - 3y}{5} = 37.5
\end{align*}
$$

<img src="figures/desmos2.png" alt="desmos2" width="300px">

</details>
