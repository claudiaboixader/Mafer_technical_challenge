from pathlib import Path
import random
import numpy as np
from solver import MaterialDatabase, solve
import json

def generate_random_formula(db: MaterialDatabase, min_materials: int = 5, max_materials: int = 20) -> list[dict]:
    """Genera una fórmula sintètica aleatòria i esparsa"""
    num_mats = random.randint(min_materials, max_materials)
    selected_mats = random.sample(db.material_names, num_mats)
    
    # Pesos aleatoris (distribució Dirichlet o Uniforme normalitzada)
    weights = np.random.dirichlet(np.ones(num_mats)) * 100.0
    
    formula = [
        {"material": mat, "percentage": round(float(w), 4)}
        for mat, w in zip(selected_mats, weights)
    ]
    return formula


def add_gcms_noise(
    compounds: list[dict], 
    noise_level: float = 0.05, 
    dropout_prob: float = 0.05,
    min_detection_limit: float = 0.005
) -> list[dict]:
    """
    Simula errors de mesura reals de GC-MS:
    - noise_level: Soroll gaussià multiplicatiu (ex: 5% de desviació en l'àrea del pic).
    - dropout_prob: Probabilitat que un compost minoritari desaparegui (sota el límit de detecció).
    - min_detection_limit: Components per sota d'aquest % s'eliminen (soroll de fons).
    """
    np.random.seed(42)
    noisy_compounds = []
    for c in compounds:
        pct = c["percentage"]
        
        # 1. Aplicar límit de detecció / dropout
        if pct < min_detection_limit or random.random() < dropout_prob:
            continue
            
        # 2. Soroll gaussià multiplicatiu (variabilitat instrumental)
        noise_factor = np.random.normal(1.0, noise_level)
        noisy_pct = max(0.0, pct * noise_factor)
        
        noisy_compounds.append({
            "compound": c["compound"],
            "percentage": noisy_pct
        })
        
    # Re-normalitzar perquè sumi 100% (o l'escala equivalent del cromatograma)
    total_pct = sum(item["percentage"] for item in noisy_compounds)
    if total_pct > 0:
        for item in noisy_compounds:
            item["percentage"] = round((item["percentage"] / total_pct) * 100.0, 4)
            
    return noisy_compounds

def generate_synthetic_data(here: Path, db: MaterialDatabase):
    print(f"=== GENERATING SYNTHETIC DATA ===")

    # Synthetic Formula (Zero Noise)
    original_synth = generate_random_formula(db=db, min_materials=5, max_materials=20)
    save_synth_path = here / "synthetic_formula.json"
    with open(save_synth_path, "w") as f:
        json.dump({"materials": original_synth}, f, indent=4)
    print("Synthetic formula saved to synthetic_formula.json")
    
    compounds_synth = db.formula_to_compounds(original_synth)
    save_path = here / "synthetic_compounds.json"
    with open(save_path, "w") as f:
        json.dump({"compounds": compounds_synth}, f, indent=4)
    print("Synthetic compounds saved to synthetic_compounds.json")

    # Synthetic Compounds (With 3% Lab Noise)
    noisy_compounds = add_gcms_noise(compounds_synth, noise_level=0.03, dropout_prob=0.02)
    save_noisy_path = here / "synthetic_compounds_noisy.json"
    with open(save_noisy_path, "w") as f:
        json.dump({"compounds": noisy_compounds}, f, indent=4)
    print("Noisy synthetic compounds saved to synthetic_compounds_noisy.json")

def evaluate_solver(db: MaterialDatabase, num_tests: int = 50):
    """Executa un banc de proves amb dades sintètiques sorolloses per avaluar el solver."""
    errors = []
    success_count = 0
    
    print(f"--- Executant banc de proves de {num_tests} iteracions amb soroll ---")
    
    for i in range(num_tests):
        # 1. Crear fórmula original
        original_formula = generate_random_formula(db)
        
        # 2. Convertir a compostos (GC-MS ideal)
        ideal_compounds = db.formula_to_compounds(original_formula)
        
        # 3. Afegir soroll real de laboratori (5% error + dropouts)
        noisy_compounds = add_gcms_noise(ideal_compounds, noise_level=0.05, dropout_prob=0.03)
        
        # 4. Intentar resoldre
        reconstructed = solve(db, noisy_compounds, threshold=0.1)
        
        # 5. Calcular Error Mitjà Absolut (MAE) entre original i reconstruït
        orig_dict = {item["material"]: item["percentage"] for item in original_formula}
        recon_dict = {item["material"]: item["percentage"] for item in reconstructed}
        
        all_mats = set(orig_dict.keys()).union(set(recon_dict.keys()))
        abs_diffs = [abs(orig_dict.get(m, 0.0) - recon_dict.get(m, 0.0)) for m in all_mats]
        mae = np.mean(abs_diffs)
        errors.append(mae)
        
        if mae < 2.0:  # Considerem èxit si l'error mitjà per ingredient és inferior a 2 punts percentuals
            success_count += 1
            
    print(f"Resultats del test:")
    print(f" - Error Mitjà Absolut (MAE) global: {np.mean(errors):.4f}%")
    print(f" - Taxa d'èxit (MAE < 2%): {(success_count / num_tests) * 100:.1f}%")