from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from solver import MaterialDatabase, solve, solve_genetic_algorithm, solve_hybrid, solve_genetic_algorithm
from generate_data import generate_random_formula, add_gcms_noise


def calculate_error(original: list[dict], reconstructed: list[dict]) -> float:
    """
    Calculates the Ingredient F1-Score (0.0 to 1.0) between two formulas.
    Higher values indicate better structural accuracy and fewer false-positive ingredients.
    """
    orig_dict = {item["material"]: item["percentage"] for item in original}
    recon_dict = {item["material"]: item["percentage"] for item in reconstructed}
    
    # Active presence threshold
    threshold = 0.05
    orig_set = {m for m, p in orig_dict.items() if p > threshold}
    recon_set = {m for m, p in recon_dict.items() if p > threshold}
    
    true_positives = len(orig_set & recon_set)
    false_positives = len(recon_set - orig_set)  # Wrong/phantom ingredients added by noise
    false_negatives = len(orig_set - recon_set)  # Missing ingredients
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return float(f1_score)*100

def compare_solvers_report(title: str, original: list[dict], compounds: list[dict], db: MaterialDatabase) -> None:
    """
    Executes NNLS, genetic and Hybrid solvers on the same profile and prints 
    a unified alphabetic table comparing all of them alongside their MAE scores.
    """
    # 1. Run all three solvers
    res_nnls = solve(db, compounds)
    res_hybrid = solve_hybrid(db, compounds)
    res_genetic = solve_genetic_algorithm(db, compounds)  # Optional: Include if implemented

    # 2. Calculate individual MAEs
    mae_nnls = calculate_error(original, res_nnls)
    mae_hybrid = calculate_error(original, res_hybrid)
    mae_genetic = calculate_error(original, res_genetic)  # Optional: Include if implemented

    # 3. Print header and MAE metrics summary
    print(f"\n{'='*92}")
    print(f" {title}")
    print(f"{'='*92}")
    print(f"F1-Score Summary:")
    print(f" • Pure NNLS:   {mae_nnls:.4f}%")
    print(f" • Hybrid (Lasso+NNLS): {mae_hybrid:.4f}%")
    print(f" • Genetic Algorithm: {mae_genetic:.4f}%\n")

    # 4. Set up table headers
    print(f"{'Material Name':<32} | {'Orig %':<8} | {'NNLS %':<8} | {'Hybrid %':<8} | {'Genetic %':<8}")
    print("─" * 90)

    # 5. Build lookup dictionaries for each result
    orig_dict = {item["material"]: item["percentage"] for item in original}
    nnls_dict = {item["material"]: item["percentage"] for item in res_nnls}
    hybrid_dict = {item["material"]: item["percentage"] for item in res_hybrid}
    genetic_dict = {item["material"]: item["percentage"] for item in res_genetic}  # Optional: Include if implemented

    # Gather all unique materials across all models and sort alphabetically
    all_materials = sorted(list(
        set(orig_dict.keys()) | set(nnls_dict.keys()) | set(hybrid_dict.keys()) | set(genetic_dict.keys())
    ))

    # 6. Print rows side by side
    for mat in all_materials:
        o = orig_dict.get(mat)
        n = nnls_dict.get(mat)
        h = hybrid_dict.get(mat)
        g = genetic_dict.get(mat)  # Optional: Include if implemented

        o_str = f"{o:>6.2f}%" if o is not None else "  ---  "
        n_str = f"{n:>6.2f}%" if n is not None else "  ---  "
        h_str = f"{h:>6.2f}%" if h is not None else "  ---  "
        g_str = f"{g:>6.2f}%" if g is not None else "  ---  "

        print(f"{mat:<32} | {o_str}  | {n_str} | {h_str}  | {g_str}")

    print("─" * 90)

def print_comparison(title: str, original: list[dict], reconstructed: list[dict], mae: float) -> None:
    """Imprimeix una taula comparativa ordenada alfabèticament per facilitar la validació."""
    print(f"\n{'='*75}")
    print(f" {title}")
    print(f"{'='*75}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}%\n")
    
    print(f"{'Material Name':<38} {'Orig %':<10} | {'Recon %':<10}")
    print("─" * 64)
    
    orig_dict = {item["material"]: item["percentage"] for item in original}
    recon_dict = {item["material"]: item["percentage"] for item in reconstructed}
    
    # Ordenar alfabèticament
    all_materials = sorted(list(set(orig_dict.keys()).union(set(recon_dict.keys()))))
    
    for mat in all_materials:
        orig_val = orig_dict.get(mat)
        recon_val = recon_dict.get(mat)
        
        orig_str = f"{orig_val:>6.2f}%" if orig_val is not None else f"{'---':>7}"
        recon_str = f"{recon_val:>6.2f}%" if recon_val is not None else f"{'---':>7}"
        
        print(f"{mat:<38} {orig_str}   |   {recon_str}")
        
    print("─" * 64)

def run_all_tests(here: Path, db: MaterialDatabase, solver_func=solve, solver_name: str = "hybrid") -> None:
    print(f"=== RUNNING TEST SUITE USING SOLVER: {solver_name.upper()} ===")
    
    # Test 1: Predefined Formula
    formula_path = here / "formula_example.json"
    if formula_path.exists():
        with open(formula_path) as f:
            data = json.load(f)
        original_json = data["materials"] if isinstance(data, dict) and "materials" in data else data
        compounds_json = db.formula_to_compounds(original_json)
        result_json = solver_func(db, compounds_json)
        mae_json = calculate_error(original_json, result_json)
        print_comparison("TEST 1: Predefined Formula (formula_example.json)", original_json, result_json, mae_json)

    # Test 2: Synthetic Formula (Zero Noise)
    original_synth = generate_random_formula(db, 5, 10)
    compounds_synth = db.formula_to_compounds(original_synth)
    result_synth = solver_func(db, compounds_synth)
    mae_synth = calculate_error(original_synth, result_synth)
    print_comparison("TEST 2: Synthetic Formula (Zero Noise / Ideal Check)", original_synth, result_synth, mae_synth)

    # Test 3: Synthetic Formula (With 3% Lab Noise)
    noisy_compounds = add_gcms_noise(compounds_synth, noise_level=0.03, dropout_prob=0.02)
    result_noisy = solver_func(db, noisy_compounds)
    mae_noisy = calculate_error(original_synth, result_noisy)
    print_comparison("TEST 3: Synthetic Formula (Robustness Check with 3% Noise)", original_synth, result_noisy, mae_noisy)

    print("\n🎉 All test suites passed successfully!")