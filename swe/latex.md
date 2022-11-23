# Latex notes

https://www.overleaf.com/learn/latex/Spacing_in_math_mode#Spaces
https://www.overleaf.com/learn/latex/Matrices
https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols

Aligning equations

```latex
\begin{equation*}
\begin{align*}
    \text{Net Input (z)}
    &= w_1 \, \dot{} \, x_1 + w_2 \, \dot{} \, x_2+ w_3 \, \dot{} \, x_3 + w_0 \\
    &= 0.75 \, \dot{} \, 2 + 0.45 \, \dot{} \, 0.6 + 0.25 \, \dot{} \, 1.2 + 0.5 \, \dot{} \, 1 \\
    &= 1.5 + 0.27 + 0.3 + 0.5 = 2.57
\end{align*}
\end{equation*}

\begin{equation*}
\begin{align*}
\end{align*}
\end{equation*}
```

pmatrix = matrix with parentheses (round brackets)
bmatrix = matrix with brackets (square brackets)
Bmatrix = matrix with braces (curly brackets)
vmatrix = matrix with pipes (determinant)
Vmatrix = matrix with double pipes (norm)

```latex
\begin{equation}
\begin{bmatrix}
  0.9 & 0 \\
  0 & 0.2
\end{bmatrix}
\end{equation}

\begin{equation}
\begin{matrix}
    B_{1,1} \Leftrightarrow (P_{1,2} \lor P_{2,1}) \\
    S_{1,1} \Leftrightarrow (W_{1,2} \lor W_{2,1})
\end{matrix}
\end{equation}
```

Showing curly brace next to an equation with statements

```latex
X =
\begin{cases}
    0    &\text{if} x < \epsilon \\
    1    &\text{otherwise}
\end{cases}
```

Binomial probabilty equation using (n choose x) and big parentheses around a fraction

```latex
  {N \choose x} \left(\frac{1}{2}\right)^x \: \left(\frac{1}{2}\right)^{N-x}
```

Show argmax with "a" centered underneath

\underset{a}{\operatorname{argmax}}

\lceil n / N \rceil
\lfloor n / N \rfloor

\vec{G} vector
\hat{G}

\leftarrow

\implies
\rightarrow →
\Rightarrow

\iff
\Leftrightarrow

\{
\}
\| ||

\cdot
\dot{}

\partial
\nabla ∇

\subset
\subseteq

\infty infinity

\ell cursive ell

\circ (degree)

\le ≤
\ge ≥
\ne ≠

KB \models \alpha

\in ∈

en dash –
em dash —

\pm +/-

\langle
\rangle

\backslash

Symbols (Wikipedia)
α
β
γ
δ
Σ σ
Τ τ
π
ε
Ω
ω
θ
ϕ

¬
∧
∨
≈

= is equal to
\equiv
\approx
\sim

\ldots elipsis
\vdots

\| norm

\dag cross

\mathbb{R} Real numbers
\mathbf{R}

\mathrm{not italic}

\boldsymbol{\epsilon} bold greek letter

P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(A ∪ B) = P(A) + P(B)

\succ ≻ A preferred to B
\nsucc
\sim ∼

\top
\bot ⊥
