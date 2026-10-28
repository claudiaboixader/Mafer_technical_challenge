# app.py
from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from solver import MaterialDatabase, solve, solve_hybrid, solve_genetic_algorithm
from generate_data import add_gcms_noise
from tests import calculate_error

# Page Configuration
st.set_page_config(
    page_title="Essential Oil Reverse Engineering Tool",
    page_icon="🧪",
    layout="wide"
)

# Load Database (Cached so it only loads once for speed)
@st.cache_resource
def get_database():
    here = Path(__file__).parent
    db_path = here / "rm_compound_amount.csv"
    return MaterialDatabase.from_csv(db_path)

db = get_database()

# Title and Business Framing Header
st.title("🧪 Essential Oil Reverse-Engineering Solver")
st.markdown(
    "Reconstruct complex natural raw material formulas "
    "from molecular GC-MS chromatography data in seconds using advanced optimization "
    "(**NNLS, Hybrid (Lasso + NNLS) & Genetic Algorithm**)."
)

# --- SIDEBAR NAVIGATION & CONFIGURATION ---
st.sidebar.header("🧭 Navigation")
app_mode = st.sidebar.radio(
    "Select Mode",
    options=["🔍 GC-MS Compound Solver", "📊 Noise Test"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Solver Configuration")
solver_choice = st.sidebar.selectbox(
    "Select Optimization Algorithm",
    options=["hybrid", "genetic_algorithm", "nnls"],
    format_func=lambda x: {
        "nnls": "Pure Non-Negative Least Squares (NNLS)",
        "hybrid": "Two-Step Hybrid (Lasso + NNLS)",
        "genetic_algorithm": "Genetic Algorithm",
    }[x]
)

threshold = st.sidebar.slider("Minimum % Threshold", 
                            min_value=0.0, 
                            max_value=0.1,
                            value=0.05, 
                            step=0.001,
                            format="%.3f",
                            help="Cleans up the final output table by hiding trace percentages below this cutoff.")

alpha = 0.05
if solver_choice in ["hybrid", "elastic_net"]:
    alpha = st.sidebar.slider("Regularization Penalty (alpha)", 
                            min_value=0.001, 
                            max_value=0.1, 
                            value=0.05, 
                            step=0.001, 
                            format="%.3f",
                            help="Controls overall regularization strength.")

# --- CONDITIONAL SIDEBAR CONTROLS (Only visible in Noise Test Mode) ---
noise_level = 0.0
dropout_prob = 0.0

if app_mode == "📊 Noise Test":
    st.sidebar.markdown("---")
    st.sidebar.header("🔬 Noise Test")
    noise_level = st.sidebar.slider("GC-MS Gaussian Noise Level", 
                                    min_value=0.0, 
                                    max_value=0.10, 
                                    value=0.0, 
                                    step=0.001, 
                                    format="%.3f",
                                    help="Simulates real-world analytical instrument noise and calibration drift.")
    dropout_prob = st.sidebar.slider("Compound Dropout Probability", 
                                     min_value=0.0, 
                                     max_value=0.05, 
                                     value=0.0, 
                                     step=0.001, 
                                     format="%.3f",
                                     help="Simulates trace chemical compounds falling below instrument detection thresholds.")

# Map solver selection
def run_solver(db, compounds):
    if solver_choice == "nnls":
        return solve(db, compounds, threshold=threshold)
    elif solver_choice == "hybrid":
        return solve_hybrid(db, compounds, threshold=threshold, alpha=alpha)
    elif solver_choice == "genetic_algorithm":
        return solve_genetic_algorithm(db, compounds, threshold=threshold)


# --- MAIN APP VIEWS ---

if app_mode == "🔍 GC-MS Compound Solver":
    st.header("Step 1: Upload GC-MS Compound Profile")
    st.markdown("Upload a JSON file containing measured compounds and their percentages, or use the default example.")
    
    uploaded_file = st.file_uploader("Upload Compound JSON file", type=["json"])

    compounds_data = None
    if uploaded_file is not None:
        try:
            content = json.load(uploaded_file)
            compounds_data = content.get("compounds", content)
            st.success("Successfully loaded uploaded compound data!")
        except Exception as e:
            st.error(f"Error parsing JSON file: {e}")
    else:
        here = Path(__file__).parent
        ex_path = here / "formula_example.json"
        if ex_path.exists():
            with open(ex_path) as f:
                ex_data = json.load(f)
            orig = ex_data["materials"] if isinstance(ex_data, dict) and "materials" in ex_data else ex_data
            compounds_data = db.formula_to_compounds(orig)
            st.info("ℹ️ No file uploaded. Using default sample compound profile derived from `formula_example.json`.")

    if compounds_data:
        result = run_solver(db, compounds_data)
            
        if not result:
            st.warning("⚠️ No materials met the threshold criteria with current parameter settings. Try lowering the threshold or alpha.")
        else:
            st.subheader("🎯 Reconstructed Raw Material Formula")
            
            total_pct = sum(item["percentage"] for item in result)
            col1, col2 = st.columns(2)
            col1.metric("Materials Identified", len(result))
            col2.metric("Total Sum", f"{total_pct:.2f}%")
            
            df_result = pd.DataFrame(result)
            st.dataframe(
                df_result.rename(columns={"material": "Raw Material", "percentage": "Percentage (%)"}), 
                use_container_width=True,
                height=350
            )
            
            # One-Click Export Buttons
            st.markdown("### 📥 Export Results")
            col_ex1, col_ex2 = st.columns(2)
            
            json_str = json.dumps(result, indent=4)
            col_ex1.download_button(
                label="Download as JSON",
                data=json_str,
                file_name="reconstructed_formula.json",
                mime="application/json"
            )
            
            csv_data = df_result.to_csv(index=False).encode('utf-8')
            col_ex2.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="reconstructed_formula.csv",
                mime="text/csv"
            )


else:  # Noise Test Mode
    st.header("📊 Forward & Reverse Round-Trip Stress Test")
    st.markdown("Upload a reference formula JSON, apply real-time lab noise/dropouts in the sidebar, and evaluate solver resilience.")
    
    formula_file = st.file_uploader("Upload Reference Formula JSON", type=["json"], key="formula_upload")
    
    formula_data = None
    if formula_file is not None:
        try:
            f_content = json.load(formula_file)
            formula_data = f_content.get("materials", f_content)
            st.success("Formula uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        here = Path(__file__).parent
        ex_path = here / "formula_example.json"
        if ex_path.exists():
            with open(ex_path) as f:
                f_content = json.load(f)
            formula_data = f_content.get("materials", f_content)
            st.info("ℹ️ Using default `formula_example.json` for simulation.")

    if formula_data:
        # Lock random seed for deterministic simulation results
        np.random.seed(42)
        
        # Forward pass to compounds
        simulated_compounds = db.formula_to_compounds(formula_data)
        
        # Apply optional noise and dropouts from sidebar sliders
        if noise_level > 0.0 or dropout_prob > 0.0:
            simulated_compounds = add_gcms_noise(simulated_compounds, noise_level=noise_level, dropout_prob=dropout_prob)
            st.warning(f"⚠️ Active Stress Test: Gaussian Noise = {noise_level*100:.1f}%, Dropout Probability = {dropout_prob*100:.1f}%")

        # Inverse pass back to formula
        reconstructed_formula = run_solver(db, simulated_compounds)
        
        # Calculate F1-Score
        f1_score = calculate_error(formula_data, reconstructed_formula)
        
        # Required dictionaries & sets for TVD, downstream tables, and comparison chart
        orig_dict = {item["material"]: item["percentage"] for item in formula_data}
        recon_dict = {item["material"]: item["percentage"] for item in reconstructed_formula}
        all_mats = set(orig_dict.keys()) | set(recon_dict.keys())
        
        # Calculate Total Variation Distance (TVD)
        tvd = 0.5 * sum(abs(orig_dict.get(m, 0.0) - recon_dict.get(m, 0.0)) for m in all_mats)
        
        # Metric display (Side-by-side metrics)
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(
            "Ingredient F1-Score", 
            f"{f1_score:.4f}", 
            help="Range 0.0 to 1.0 (Higher is better). Measures structural ingredient accuracy."
        )
        m_col2.metric(
            "Total Variation Distance (TVD)", 
            f"{tvd:.2f}%", 
            help="Lower is better. Accumulates compositional errors and phantom ingredients."
        )
        
        # Side by side display tables
        col_a, col_b = st.columns(2)
        df_orig = pd.DataFrame(formula_data)
        df_recon = pd.DataFrame(reconstructed_formula)
        
        with col_a:
            st.subheader("Original Reference Formula")
            st.dataframe(df_orig, use_container_width=True, height=300)
        with col_b:
            st.subheader("Reconstructed Result")
            st.dataframe(df_recon, use_container_width=True, height=300)

        # Visual Comparison Chart
        st.subheader("📈 Visual Comparison (Original vs Reconstructed)")
        comparison_df = pd.DataFrame(index=list(all_mats))
        comparison_df["Original %"] = comparison_df.index.map(lambda m: orig_dict.get(m, 0.0))
        comparison_df["Reconstructed %"] = comparison_df.index.map(lambda m: recon_dict.get(m, 0.0))
        st.bar_chart(comparison_df)