# Quantum Computing (QC)

- [1. Felix Bloch - Bloch Sphere](#1-felix-bloch---bloch-sphere)
- [2. Quantum Register](#2-quantum-register)
  - [2.1. Tensor Product](#21-tensor-product)
- [3. Quantum Gates](#3-quantum-gates)
  - [3.1. Pauli Gates](#31-pauli-gates)
  - [3.2. CNOT Gate](#32-cnot-gate)
  - [3.3. Toffoli Gate](#33-toffoli-gate)
- [4. Quantum Circuit](#4-quantum-circuit)
- [5. Quantum Superposition](#5-quantum-superposition)
  - [5.1. Hadamard Gate](#51-hadamard-gate)
  - [5.2. Quantum Entanglement](#52-quantum-entanglement)
- [6. Mathematical Structure](#6-mathematical-structure)
- [7. Axioms of Quantum Mechanics (QM)](#7-axioms-of-quantum-mechanics-qm)
- [8. Multi-System](#8-multi-system)
- [9. No Clone Theorem](#9-no-clone-theorem)

Any **single-qubit state** can be written as

$$
\begin{align*}
  |\psi\rangle &= \alpha|0\rangle + \beta|1\rangle,\\
  |\alpha|^2 + |\beta|^2 &= 1.
\end{align*}
$$

- $|\psi\rangle$ is a qubit
- $\alpha,\beta\in\mathbb{C}$
  - $\alpha=\Re(\alpha) + \Im(\alpha)$
  - $\beta=\Re(\beta) + \Im(\beta)$
  - $\Rightarrow 4$ dimensional space
- $|0\rangle = \begin{pmatrix}1\\0\end{pmatrix}$
- $|1\rangle = \begin{pmatrix}0\\1\end{pmatrix}$

## 1. Felix Bloch - Bloch Sphere

- Sphere has dimension **2**, any point on the sphere can be represented by 2 angles $\theta$ and $\varphi$.
- Any qubit can be represented by a point on the sphere.

$$
\begin{align*}
|\psi\rangle &= \cos(\theta/2) e^{i\varphi_{0}}|0\rangle + \sin(\theta/2) e^{i\varphi_{1}}|1\rangle\\
&= e^{i\varphi_{0}}\left(\cos(\theta/2) |0\rangle + \sin(\theta/2) e^{i(\varphi_{1}-\varphi_{0})}|1\rangle\right)
\end{align*}
$$

<img src="figures/bloch-sphere.png" alt="bloch-sphere" width="350px">

$\vec{s}$ is called the **Bloch vector**.

## 2. Quantum Register

$$
\begin{align*}
  |\psi\rangle &= \sum\limits_{i=0}^{2^n-1}\alpha_i|i\rangle,\\
  \sum\limits_{i=0}^{2^n-1}|\alpha_i| &= 1.
\end{align*}
$$

- To simulate 100 qubits, we would need $2^{100}$ bits.

### 2.1. Tensor Product

$$
|1\rangle \otimes |0\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \otimes
\begin{pmatrix} 1 \\ 0 \end{pmatrix}
=
\begin{pmatrix} 0 \cdot 1 \\ 0 \cdot 0 \\ 1 \cdot 1 \\ 1 \cdot 0 \end{pmatrix}
=
\begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}
$$

$$
|5\rangle_3 = |101\rangle = |1\rangle \otimes |0\rangle \otimes |1\rangle =
\begin{pmatrix} 0 \\ 1 \end{pmatrix} \otimes
\begin{pmatrix} 1 \\ 0 \end{pmatrix} \otimes
\begin{pmatrix} 0 \\ 1 \end{pmatrix} =
\begin{pmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}
$$

- Associative: $(A\otimes B)\otimes C = A\otimes(B\otimes C)$
- Distributive: $A\otimes(B+C) = A\otimes B + A\otimes C$
- **Not** commutative: $A\otimes B \neq B\otimes A$

## 3. Quantum Gates

- Matrices representing quantum gates are unitary.
    $$UU^{\dagger}=U^{\dagger}U=I$$
- The operations are reversible.
- Tensor product of unitary matrices is a unitary matrix.
    $$(U_1\oplus U_2)^{\dagger}(U_1\oplus U_2)=(U_2^{\dagger}\oplus U_1^{\dagger})(U_1\oplus U_2)=U_1^{\dagger}U_1\oplus U_2^{\dagger}U_2=\mathbb{I}\oplus\mathbb{I}=\mathbb{I}$$
- Sum of unitary matrices of the same dimension is a unitary matrix.

### 3.1. Pauli Gates

- **Pauli-X** bit-flip gate
  - $X|0\rangle = |1\rangle$
  - $X|1\rangle = |0\rangle$
  - $X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$

### 3.2. CNOT Gate

- **Controlled-NOT** gate
- $CNOT = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$

### 3.3. Toffoli Gate

- **Toffoli** gate
- $CCNOT = \begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \end{pmatrix}$

## 4. Quantum Circuit

1. Encode the input.
2. Apply quantum gates to the sequence of input qubits.
3. Measure one or more qubits. The measurement is irreversible.

## 5. Quantum Superposition

- **Superposition** is a fundamental principle of quantum mechanics.

### 5.1. Hadamard Gate

- $H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$
- $H|0\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$
- $H|1\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$

### 5.2. Quantum Entanglement

- **Entanglement** is a quantum phenomenon that occurs when pairs or groups of particles interact in such a way that the quantum state of each particle cannot be described independently of the state of the others, even when the particles are separated by a large distance. AKA quantum teleportation.

## 6. Mathematical Structure

- Finite $n$-dimensional Hilbert space $\mathcal{H}\in\mathbb{C}^n$ with inner product $\langle\cdot|\cdot\rangle = \langle\cdot| \,\,\tilde{\cdot}\,\, |\cdot\rangle$.
- **ket** vector $|x\rangle=\begin{pmatrix}
  x_0\\
  \vdots\\
  x_{n-1}
\end{pmatrix}$ and **bra** vector $\langle x| = |x\rangle^\dagger$.
- **Inner product** $\langle x|y\rangle = \sum\limits_{i=0}^{n-1}x_i^*y_i$.
- Tensor product
  - $A\in\mathbb{C}^{m,n}$
  - $B\in\mathbb{C}^{p,q}$
  - $A\otimes B\in\mathbb{C}^{mp,nq}$

$$
A\otimes B = \begin{pmatrix}
  a_{00}B & \cdots & a_{0,n-1}B\\
  \vdots & \ddots & \vdots\\
  a_{m-1,0}B & \cdots & a_{m-1,n-1}B
\end{pmatrix}
$$

## 7. Axioms of Quantum Mechanics (QM)

1. **Pure state** in QM is a normalized vector in a Hilbert space $|\psi\rangle\in\mathcal{H}$ iff $|\psi_0\rangle$ and $|\psi_1\rangle$ are physical states, then $c_0|\psi_0\rangle + c_1|\psi_1\rangle$, where $c_0,c_1\in\mathbb{C}$ and $\sum\limits_{i=0}^1|c_i|^2=1$ is also a physical state.
2. **Measurement**.
3. **Time dependent Schrödinger equation**
    $$i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle = {H}|\psi\rangle,$$
    where $\hbar$ is the Planck constant and $H$ is the Hamiltonian operator corresponding to the system energy.

## 8. Multi-System

- Two registers, two Hilbert spaces $\mathcal{H}_0$ and $\mathcal{H}_1$.
  $$\mathcal{H}=\mathcal{H}_0\otimes\mathcal{H}_1$$
  - $\mathrm{dim}\mathcal{H}= \mathrm{dim}\mathcal{H}_1\cdot\mathrm{dim}\mathcal{H}_1$
  - For separated states: $\mathrm{dim}\mathcal{H}_0 = \mathrm{dim}\mathcal{H}_1 = \mathrm{dim}\mathcal{H}_0+\mathrm{dim}\mathcal{H}_1$

## 9. No Clone Theorem

**No cloning theorem** states that it is impossible to create an identical copy of an arbitrary unknown quantum state.

> Quantum system cannot be cloned by a unitary transformation.
