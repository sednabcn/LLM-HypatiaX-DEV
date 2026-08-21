# Issue 1 — Nguyen-12 Table Success-Count Arithmetic

**Category:** [STALE — reconciliation only] (partial arithmetic fix) + [OPEN — needs code/data investigation] (per-equation values unverified)

**Location:** `jmlr_paper_main.tex`, `subsec:nguyen12-benchmark-standard-sr-suite` ("Nguyen-12 Benchmark (Standard SR Suite)"), `tab:nguyen12`, l. 2432–2520.

## What is wrong
The table's printed Success row and its own per-equation cells disagree with each other under the table's own stated criterion (Bold: extrap R² ≥ 0.9999):

> Success (R² ≥ 0.9999) — P: 10/12 (83.3%) — H: 11/12 (91.7%) — N: 0/12 (0%)

A hand recount of the twelve bolded per-equation extrapolation cells gives:
- **PySR-only (P): 8/12**, not 10/12 (N-3 and N-7 are not ≥ 0.9999 but are not counted as passes, and no other cell was miscounted — the printed total simply does not equal the number of bolded cells).
- **HypatiaX (H): 7/12**, not 11/12 — two cells (N-3 = 0.9976, N-10 = 0.9997) are typeset in bold despite being below the table's own ≥ 0.9999 threshold, and removing them from the bolded/passing count leaves 7, not 11.

A third, independently conflicting figure appears in the caption itself:

> Strict recovery (≥ 0.9999) yields 4/12 (33.3%); the 91.7% figure uses the 4-decimal rounding convention.

Three different denominators for the same nominal quantity — 10/12 (row), 11/12 (row, also restated in prose at l. 220, 2503), and 4/12 (caption) — appear in the same table environment and do not reconcile against each other or against the twelve visible per-equation cells.

## Content to fix (verbatim from `jmlr_paper_main.tex`, l. 2423–2520)

```latex
\subsection{Nguyen-12 Benchmark (Standard SR Suite)}\label{subsec:nguyen12-benchmark-standard-sr-suite}
\label{sec:nguyen12}

The Nguyen-12 suite~\citep{uy2011semantically} is the standard symbolic-regression
benchmark comprising 12 polynomial and transcendental equations (Nguyen-1 through
Nguyen-12) with up to two input variables.  We evaluate HypatiaX and PySR-only on
the same PCA-directed aggressive extrapolation protocol used for the DeFi benchmark
(40\,\%/60\,\% split), and compare against the neural MLP baseline.

\begin{table}[h]
\centering
\caption{Nguyen-12 benchmark: train and extrapolation $\Rsq$ by equation.
$P$ = PySR-only; $H$ = HypatiaX; $N$ = Neural MLP.
Near-miss criterion: $\Rsq \ge 0.9999$ (4-decimal convention of
\citealt{uy2011semantically}).
Near-miss detail: $\Rsq\in[0.9998,0.9999)$ with perfect extrapolation
($\Rsq=1.0000$ on $2\times$ holdout). Strict recovery ($\ge 0.9999$) yields
4/12 (33.3\%); the 91.7\,\% figure uses the 4-decimal rounding convention.}
\label{tab:nguyen12}
\begin{threeparttable}
\begin{tabular}{llrrrrrr}
\toprule
Eq. & Formula &
  \multicolumn{2}{c}{PySR-only ($P$)} &
  \multicolumn{2}{c}{HypatiaX ($H$)} &
  \multicolumn{2}{c}{Neural MLP ($N$)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
 &  & Train & Extrap & Train & Extrap & Train & Extrap \\
\midrule
N-1  & $x^3 + x^2 + x$
  & 0.9999 & \textbf{1.0000}
  & 0.9999 & \textbf{0.9999}
  & 0.9993 & $-0.784$ \\
N-2  & $x^4 + x^3 + x^2 + x$
  & 0.9999 & \textbf{1.0000}
  & 0.9999 & \textbf{1.0000}
  & 0.9986 & $-0.902$ \\
N-3  & $x^5 + x^4 + x^3 + x^2 + x$
  & 0.9999 & \emph{$-426.2$}
  & 0.9999 & \textbf{0.9976}
  & 0.9986 & $-0.913$ \\
N-4  & $x^6 + x^5 + x^4 + x^3 + x^2 + x$
  & 0.9999 & \emph{$\ll{-}100$}
  & 0.9999 & \emph{$\ll{-}100$}
  & 0.9979 & $-0.828$ \\
N-5  & $\sin(x^2)\cos(x)-1$
  & 0.9999 & \textbf{1.0000}
  & 0.9999 & \textbf{1.0000}
  & 0.9979 & $-5.586$ \\
N-6  & $\sin(x)+\sin(x+x^2)$
  & 0.9999 & \textbf{1.0000}
  & 0.9999 & \textbf{1.0000}
  & 0.9987 & $-12.654$ \\
N-7  & $\ln(x+1)+\ln(x^2+1)$
  & 0.9999 & 0.9762
  & 0.9999 & 0.7316
  & 0.9868 & $0.856$ \\
N-8  & $\sqrt{x}$
  & 0.9999 & \textbf{1.0000}
  & 0.9999 & \textbf{1.0000}$^\dag$
  & 0.9988 & $0.954$ \\
N-9  & $\sin(x)+\sin(y^2)$
  & 0.9999 & \textbf{1.0000}
  & 0.9999 & \textbf{1.0000}
  & 0.9986 & $-6.708$ \\
N-10 & $2\sin(x)\cos(y)$
  & 0.9999 & \textbf{1.0000}
  & 0.9999 & \textbf{0.9997}
  & 0.9995 & $-2.379$ \\
N-11 & $x^y$
  & 0.9999 & \textbf{1.0000}
  & 0.9999 & \textbf{0.9999}
  & 0.9984 & $-0.423$ \\
N-12 & $x^4 - x^3 + \tfrac{1}{2}y^2 - y$
  & 0.9987 & $-1.056$
  & 0.9994 & $-1.054$
  & 0.9985 & $-1.198$ \\
\midrule
\multicolumn{2}{l}{Success ($\Rsq\ge0.9999$)}
  & \multicolumn{2}{c}{10/12 (83.3\%)}
  & \multicolumn{2}{c}{\textbf{11/12 (91.7\%)}}
  & \multicolumn{2}{c}{0/12 (0\%)} \\
MW $H>P$   & \multicolumn{7}{l}{$U=51.0$, $p=0.893$ — not significant} \\
MW $P>N$   & \multicolumn{7}{l}{$U=113.0$, $p=0.0097$ — significant} \\
\bottomrule
\end{tabular}
\begin{tablenotes}\small
  \item \textbf{Bold}: extrap $\Rsq \ge 0.9999$. \emph{Italic}: $\Rsq < 0$.
    $P$ = PySR-only; $H$ = HypatiaX; $N$ = Neural MLP.
  \item[$\dag$] N-8 ($\sqrt{x}$): HypatiaX routed to \texttt{hybrid\_llm\_only} mode.
    The LLM directly proposed $x^{0.5}$, which passed the trust gate with
    train $\Rsq = 0.9999$ and achieved exact extrapolation ($\Rsq = 1.000$)
    without invoking PySR. Wall-clock time: 7.4\,s vs.\ 301\,s for PySR-only ---
    the only equation where the LLM warm-start fully bypassed evolutionary search.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

**Hand recount (bolded cells) from the table above:**
- PySR-only (P) bolded extrap cells: N-1, N-2, N-5, N-6, N-8, N-9, N-10, N-11 = **8/12**
- HypatiaX (H) bolded extrap cells: N-1, N-2, N-3(0.9976 — below threshold), N-5, N-6, N-8, N-9, N-10(0.9997 — below threshold), N-11 = 9 bolded, but only **7/12** are actually ≥ 0.9999 once N-3 and N-10 are excluded
- Printed row says 10/12 and 11/12 respectively; caption separately says 4/12 (33.3%)

## Fix required
The Success row is fixable by hand re-tally against the table's own printed cells. However, per the fix-list's caution, the per-equation R² values themselves have not been independently re-verified against the raw result JSON, so a complete fix should regenerate both the per-equation cells and the summary row from source, not merely re-tally the printed numbers.
