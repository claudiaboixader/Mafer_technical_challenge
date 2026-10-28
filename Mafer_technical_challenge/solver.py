from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from sklearn.linear_model import ElasticNet

import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize
from sklearn.linear_model import Lasso

def _normalize(name: str) -> str:
    return name.strip().lower()


@dataclass
class MaterialDatabase:
    matrix: np.ndarray  # shape: (compounds, materials)
    compound_names: list[str]
    material_names: list[str]
    _compound_idx: dict[str, int] = field(repr=False, default_factory=dict)
    _material_idx: dict[str, int] = field(repr=False, default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> MaterialDatabase:
        csv_path = Path(path) if path else Path(__file__).parent / "rm_compound_amount.csv"
        df = pd.read_csv(csv_path)
        df.columns = ["material", "compound", "amount"]
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["key"] = df["compound"].map(_normalize)
        display = df.drop_duplicates("key").set_index("key")["compound"]
        pivot = (
            df.groupby(["key", "material"], sort=False)["amount"]
            .sum()
            .reset_index()
            .pivot(index="key", columns="material", values="amount")
            .fillna(0.0)
        )
        keys = list(pivot.index)
        materials = list(pivot.columns)
        return cls(
            matrix=pivot.to_numpy(dtype=np.float64),
            compound_names=[str(display.loc[k]) for k in keys],
            material_names=materials,
            _compound_idx={k: i for i, k in enumerate(keys)},
            _material_idx={_normalize(m): i for i, m in enumerate(materials)},
        )

    def formula_to_compounds(self, formula: list[dict]) -> list[dict]:
        """Forward: materials -> compounds."""
        w = np.zeros(self.matrix.shape[0])
        for item in formula:
            idx = self._material_idx.get(_normalize(item["material"]))
            if idx is not None:
                w += self.matrix[:, idx] * (item["percentage"] / 100.0)
        return sorted(
            [
                {"compound": self.compound_names[i], "percentage": round(float(v), 6)}
                for i, v in enumerate(w)
                if v >= 1e-4
            ],
            key=lambda r: r["percentage"],
            reverse=True,
        )

def get_solver_function(solver_name: str):
    """Maps string argument to the actual solver function."""
    solvers = {
        "hybrid": solve_hybrid,        # Two-step Lasso + NNLS
        "nnls": solve,     # Pure Non-Negative Least Squares
        "genetic": solve_genetic_algorithm  # Genetic Algorithm Solver
    }
    if solver_name not in solvers:
        raise ValueError(f"Unknown solver '{solver_name}'. Choose from: {list(solvers.keys())}")
    return solvers[solver_name]

def solve(db: MaterialDatabase, compounds: list[dict], threshold: float = 0.01) -> list[dict]:
    """Inverse: compounds -> materials (NNLS). Returns list of {material, percentage}."""
    rows, values = [], []
    for c in compounds:
        idx = db._compound_idx.get(_normalize(c["compound"]))
        if idx is not None:
            rows.append(idx)
            values.append(c["percentage"])
    if not rows:
        return []

    x, _ = nnls(db.matrix[rows, :], np.array(values, dtype=np.float64))
    total = x.sum()
    if total == 0:
        return []
    pct = 100.0 * x / total

    result = [
        {"material": db.material_names[i], "percentage": round(float(pct[i]), 4)}
        for i in range(len(db.material_names))
        if pct[i] > threshold
    ]
    result.sort(key=lambda m: m["percentage"], reverse=True)
    return result

def solve_hybrid(
    db: MaterialDatabase, 
    compounds: list[dict], 
    threshold: float = 0.05, 
    alpha: float = 0.001
) -> list[dict]:
    """
    Two-Step Hybrid Solver.
    Step 1: Feature selection using Non-Negative Lasso (filters noise and false positives).
    Step 2: Unbiased refinement using standard NNLS strictly on the selected materials.
    """
    rows, values = [], []
    for c in compounds:
        idx = db._compound_idx.get(_normalize(c["compound"]))
        if idx is not None:
            rows.append(idx)
            values.append(c["percentage"])
            
    if not rows:
        return []

    A_sub = db.matrix[rows, :]
    b_sub = np.array(values, dtype=np.float64)

    # --- STEP 1: Feature Selection with Non-Negative Lasso ---
    try:
        lasso = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=5000)
        lasso.fit(A_sub, b_sub)
        active_indices = np.where(lasso.coef_ > 1e-3)[0]
    except Exception:
        active_indices = np.arange(db.matrix.shape[1])

    if len(active_indices) == 0:
        active_indices = np.arange(db.matrix.shape[1])

    # --- STEP 2: Precise Refinement with NNLS (Zero Shrinkage Bias) ---
    A_filtered = A_sub[:, active_indices]
    x_filtered, _ = nnls(A_filtered, b_sub)

    x = np.zeros(db.matrix.shape[1])
    x[active_indices] = x_filtered

    total = x.sum()
    if total == 0:
        return []
        
    pct = 100.0 * x / total

    result = [
        {"material": db.material_names[i], "percentage": round(float(pct[i]), 4)}
        for i in range(len(db.material_names))
        if pct[i] > threshold
    ]
    result.sort(key=lambda m: m["percentage"], reverse=True)
    return result

def solve_genetic_algorithm(
    db: MaterialDatabase, 
    compounds: list[dict], 
    threshold: float = 0.05, 
    generations: int = 45, 
    pop_size: int = 40
) -> list[dict]:
    """
    Combinatorial Genetic Algorithm with AIC (Akaike Information Criterion) Fitness.
    Mathematically prevents overfitting to GC-MS noise and compound dropouts.
    """
    rows, values = [], []
    for c in compounds:
        comp_name = c.get("compound") or c.get("name") or c.get("substance")
        if not comp_name:
            continue
        idx = db._compound_idx.get(_normalize(comp_name))
        if idx is not None:
            rows.append(idx)
            values.append(c.get("percentage", 0.0))
            
    if not rows:
        return []

    A_full = db.matrix[rows, :]
    b_sub = np.array(values, dtype=np.float64)
    n_materials = A_full.shape[1]
    n_compounds = len(rows)
    max_allowed_ingredients = 20  # Strict ceiling for clean formulas

    # Seed initial population mask
    try:
        x_nnls, _ = nnls(A_full, b_sub)
        initial_mask = (x_nnls > 0.002).astype(int)
        if np.sum(initial_mask) > max_allowed_ingredients:
            top_indices = np.argsort(x_nnls)[-max_allowed_ingredients:]
            initial_mask = np.zeros(n_materials, dtype=int)
            initial_mask[top_indices] = 1
    except Exception:
        initial_mask = np.zeros(n_materials, dtype=int)
        initial_mask[np.random.choice(n_materials, min(10, n_materials), replace=False)] = 1

    np.random.seed(42)
    population = np.random.randint(0, 2, size=(pop_size, n_materials))
    population[0] = initial_mask

    # Properly initialized as x_best
    x_best = np.zeros(n_materials)
    best_score = float('inf')

    for gen in range(generations):
        fitnesses = []
        evaluated_weights = []

        for mask in population:
            active_indices = np.where(mask == 1)[0]
            
            if len(active_indices) == 0 or len(active_indices) > max_allowed_ingredients:
                fitnesses.append(float('inf'))
                evaluated_weights.append(np.zeros(n_materials))
                continue
            
            A_sub = A_full[:, active_indices]
            try:
                x_sub, _ = nnls(A_sub, b_sub)
                
                # Prune out any active ingredients whose NNLS weight is virtually zero (noise chasers)
                significant_sub_idx = np.where(x_sub > 1e-3)[0]
                if len(significant_sub_idx) == 0:
                    fitnesses.append(float('inf'))
                    evaluated_weights.append(np.zeros(n_materials))
                    continue
                    
                final_active_indices = active_indices[significant_sub_idx]
                A_clean = A_full[:, final_active_indices]
                x_clean, _ = nnls(A_clean, b_sub)
                
                x_full = np.zeros(n_materials)
                x_full[final_active_indices] = x_clean
                
                # --- AIC (Akaike Information Criterion) Scoring ---
                pred = A_full @ x_full
                residuals = pred - b_sub
                rss = np.sum(residuals ** 2)
                
                if rss == 0:
                    rss = 1e-12  # Prevent log(0)
                    
                k = len(final_active_indices)
                aic_score = n_compounds * np.log(rss / n_compounds) + (2 * k) + (2 * k * (k + 1) / max(1, n_compounds - k - 1))
                
                score = aic_score
            except Exception:
                score = float('inf')
                x_full = np.zeros(n_materials)

            fitnesses.append(score)
            evaluated_weights.append(x_full)
            
            if score < best_score:
                best_score = score
                x_best = x_full  # Updated consistently using x_best

        fitnesses = np.array(fitnesses)
        sorted_idx = np.argsort(fitnesses)
        
        elite_count = max(3, pop_size // 4)
        elites = population[sorted_idx[:elite_count]]

        new_pop = list(elites)
        while len(new_pop) < pop_size:
            p1 = population[np.random.choice(sorted_idx[:pop_size//2])]
            p2 = population[np.random.choice(sorted_idx[:pop_size//2])]
            
            crossover_mask = np.random.rand(n_materials) > 0.5
            child = np.where(crossover_mask, p1, p2)
            
            mutation_mask = np.random.rand(n_materials) < 0.03
            child = np.bitwise_xor(child, mutation_mask.astype(int))
            
            if np.sum(child) > max_allowed_ingredients:
                active = np.where(child == 1)[0]
                drop = np.random.choice(active, size=len(active) - max_allowed_ingredients, replace=False)
                child[drop] = 0

            new_pop.append(child)

        population = np.array(new_pop)

    total = x_best.sum()
    if total == 0:
        return []
        
    pcts = 100.0 * (x_best / total)
    result = [
        {"material": db.material_names[i], "percentage": round(float(pcts[i]), 4)}
        for i in range(n_materials)
        if pcts[i] > threshold
    ]
    result.sort(key=lambda m: m["percentage"], reverse=True)
    return result