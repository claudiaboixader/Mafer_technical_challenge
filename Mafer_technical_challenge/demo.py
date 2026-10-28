from __future__ import annotations

import argparse
import json
from pathlib import Path
from tests import run_all_tests, compare_solvers_report
from generate_data import generate_synthetic_data
from solver import MaterialDatabase, solve, get_solver_function


def run_demo(here: Path, db: MaterialDatabase, solver_func=solve, solver_name: str = "hybrid") -> None:
    """Runs a single interactive demo prediction."""
    print(f"=== RUNNING SINGLE PREDICTION DEMO USING SOLVER: {solver_name.upper()} ===")
    
    formula_path = here / "formula_example.json"
    if not formula_path.exists():
        print("[Error] formula_example.json not found.")
        return

    with open(formula_path) as f:
        data = json.load(f)
    original = data["materials"] if isinstance(data, dict) and "materials" in data else data
    
    compounds = db.formula_to_compounds(original)
    # Get chosen solver function
    result = solver_func(db, compounds)
    
    print(f"\n{'Reconstructed Material':<40} {'Percentage %':<10}")
    print("─" * 52)
    for m in result:
        print(f"{m['material']:<40} {m['percentage']:>8.4f}%")
    print("─" * 52)

def run_comparison(here: Path, db: MaterialDatabase) -> None:
    """Runs the multi-solver comparison report on the expert formula."""
    formula_path = here / "formula_example.json"
    if not formula_path.exists():
        print("[Error] formula_example.json not found.")
        return

    with open(formula_path) as f:
        data = json.load(f)
    original = data["materials"] if isinstance(data, dict) and "materials" in data else data
    compounds = db.formula_to_compounds(original)
    
    compare_solvers_report("MULTI-SOLVER BENCHMARK", original, compounds, db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Essential Oil Formula Reconstruction Solver (GC-MS)")
    parser.add_argument("--test", action="store_true", help="Run the comprehensive test suite with error metrics")
    parser.add_argument("--demo", action="store_true", help="Run a single demonstration reconstruction")
    parser.add_argument("--compare", action="store_true", help="Run side-by-side solver comparison table")
    parser.add_argument("--generate_data", action="store_true", help="Generate synthetic formulas and compounds for testing")
    
    parser.add_argument(
        "--solver", 
        type=str, 
        default="nnls", 
        choices=["hybrid", "nnls", "genetic"], 
        help="Choose optimization solver: hybrid, nnls, or genetic"
    )
    
    args = parser.parse_args()
    here = Path(__file__).parent
    db = MaterialDatabase.from_csv(here / "rm_compound_amount.csv")
    
    if args.test:
        # run tests with the selected solver. Default solver = nnls
        run_all_tests(here, db, solver_func=get_solver_function(args.solver), solver_name=args.solver)
    elif args.demo:
        # run given demo with example formula. Default solver = nnls
        run_demo(here, db, solver_func=get_solver_function(args.solver), solver_name=args.solver)
    elif args.compare:
        # return a table with all solvers side-by-side for the example formula
        run_comparison(here, db)
    elif args.generate_data:
        # generate synthetic formulas and compounds for testing
        generate_synthetic_data(here, db)
    else:
        # Default behavior if no action flag is passed: run tests with chosen solver
        run_all_tests(here, db, solver_func=get_solver_function(args.solver), solver_name=args.solver)


if __name__ == "__main__":
    main()