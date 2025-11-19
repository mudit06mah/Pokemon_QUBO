# QUBO-Based Pokémon Team Builder (Gen 1)

This project builds an optimal Pokémon team for Generation 1 using QUBO (Quadratic Unconstrained Binary Optimization) and solves it locally with `dimod` and `dwave-neal`.

## Overview

Instead of choosing Pokémon directly, the model selects type-combination counts (e.g., Fire, Water, Fire/Flying) and ensures the final team satisfies key battle constraints. All constraints are translated into QUBO penalties and solved as an optimization problem.

## Core Idea

A QUBO minimizes an energy function:

```
E(x) = xᵀQx + cᵀx + offset
```

Binary variables represent Pokémon type-combo counts, encoded using binary expansion.

## Constraints

1. **Team Size = 6** — Enforces exactly 6 Pokémon on the team using penalty term P₁(∑ᵢ cᵢ - 6)²
2. **Weakness Control** — No more than 2 Pokémon weak to any attack type. Slack variables convert inequality constraints to equality.
3. **Resistance Coverage** — At least 1 Pokémon resistant to every attack type.

All constraints use quadratic penalty terms of the form:

```
P * (sum(coeff_i * x_i) – target)²
```

## Data

- Gen 1 Pokémon stats and typings
- Gen 1 type chart (downloadable from Kaggle)

## How to Run

```bash
pip install pandas numpy dimod dwave-neal
python team_builder.py
```

Edit penalty weights in the script (`P1`, `P2`, `P3`) if needed to adjust constraint strictness.

## Output

- Best low-energy binary solution
- Reconstructed team type-combo counts
- Constraint evaluations (weakness/resistance/team size)
- Final optimal team structure (by type layout)

## Notes

- This is a feasibility-first model—it prioritizes meeting constraints over team quality.
- You can add objectives (e.g., maximize Base Stat Total) to refine team quality further.
- Solves locally—no quantum hardware needed.
