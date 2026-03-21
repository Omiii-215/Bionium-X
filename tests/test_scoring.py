import pytest
from src.scoring.biosignature import calculate_biosignature_score

def test_high_chemical_disequilibrium():
    # Simultaneous presence of heavily reactive gases
    probas = {'O2': 0.95, 'CH4': 0.88, 'H2O': 0.6, 'O3': 0.1, 'CO2': 0.5}
    score, conf = calculate_biosignature_score(probas)
    
    assert score > 0.8, "Simultaneous O2 and CH4 should trigger a high habitability score (> 0.8)"
    assert conf == "HIGH", "Confidence level must mark High for disequilibrium"

def test_abiotic_equilibrium():
    # Only standard background gases; no life predictors
    probas = {'O2': 0.1, 'CH4': 0.05, 'H2O': 0.1, 'O3': 0.0, 'CO2': 0.9}
    score, conf = calculate_biosignature_score(probas)
    
    assert score < 0.4, "Abiotic atmosphere should result in a low structural habitability score"
    assert conf in ["LOW", "MARGINAL"], "Confidence level should remain Low/Marginal for abiotic profiles"
