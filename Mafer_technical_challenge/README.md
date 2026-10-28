# Essential Oil Formula Reconstruction Solver

Reconstruct the composition of an essential oil mixture from GC-MS (Gas Chromatography–Mass Spectrometry) compound percentage data. Given a mixture's detected chemical compounds and their relative percentages, the solver estimates which reference materials (essential oils) are present and their respective proportions.

The project includes multiple optimization approaches, a reference compound database, a demonstration formula, a test suite, and synthetic data generation for evaluating solver performance.

## Contents

| File | Description |
|---|---|
| `demo.py` | Main entry point for running tests, demonstrations, solver comparisons, and synthetic data generation |
| `solver.py` | Material database, formula-to-compound conversion, and solver implementations |
| `tests.py` | Test suite and solver performance evaluation |
| `generate_data.py` | Generates synthetic formulas and compound data for testing |
| `rm_compound_amount.csv` | Reference database containing 342 materials and 1,641 compounds |
| `formula_example.json` | Example formula containing 22 materials, based on insights from a professional analyst-perfumist |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Install dependencies

Ensure Python is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Run the application

Run commands from the project directory:

```bash
python demo.py
```

By default, the application runs the test suite using the **NNLS solver**.

## Usage

The application supports four main operations: testing, demonstration, solver comparison, and synthetic data generation.

### Run the test suite

Run the comprehensive tests using the default NNLS solver:

```bash
python demo.py --test
```

Choose a specific solver:

```bash
python demo.py --test --solver nnls
python demo.py --test --solver hybrid
python demo.py --test --solver genetic
```

The test suite evaluates reconstruction performance using the project's test cases and error metrics.

### Run a single prediction demo

Reconstruct the example formula and print the estimated material percentages:

```bash
python demo.py
```

Choose a solver for the demonstration:

```bash
python demo.py --demo --solver nnls
python demo.py --demo --solver hybrid
python demo.py --demo --solver genetic
```

The demo reads `formula_example.json`, converts the input materials into their corresponding compound representation using the reference database, and runs the selected solver. The results are printed as a table of reconstructed materials and their estimated percentages.

### Compare all solvers

Run a side-by-side comparison of the available solvers on the example formula:

```bash
python demo.py --compare
```

This generates a multi-solver benchmark report using the same input formula and compound representation, allowing the approaches to be compared under consistent conditions.

The `--solver` argument does not change which solvers are included in this comparison.

### Generate synthetic data

Generate synthetic formulas and compound data for testing:

```bash
python demo.py --generate_data
```

Synthetic data can be used to develop additional test cases and evaluate reconstruction performance under controlled conditions.

## Available solvers

The `--solver` argument accepts three options:

| Solver | Description |
|---|---|
| `nnls` | Non-Negative Least Squares optimization |
| `hybrid` | Hybrid optimization approach (Lasso + NNLS) |
| `genetic` | Genetic algorithm-based optimization |

The default solver is `nnls`. The selected solver is used by the test and demo commands, as well as when running the application without an action flag.

## Interactive Application and Visualization

The project also includes an interactive application/interface that makes it easier to explore and visualize the reconstruction process and its results.

The interface provides a more accessible way to understand the reconstructed essential oil mixture, inspect estimated material proportions, and explore solver outputs without relying exclusively on command-line results.

It complements the command-line tools by making the analysis easier to interpret and present, particularly when comparing optimization methods or preparing a demonstration for clients.

Launch it with:

```bash
streamlit run app.py
```