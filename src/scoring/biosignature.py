def calculate_biosignature_score(probas):
    """
    Calculate the heuristic Biosignature Score based on molecule probabilities.
    Input probas: dict mapping molecule name to probability [0.0 - 1.0]
    e.g. {'O2': 0.81, 'CH4': 0.75, 'O3': 0.1, 'H2O': 0.9, 'CO2': 0.5}

    Returns a score from 0.0 to 1.0 and a confidence category.
    """
    p_O2 = probas.get('O2', 0.0)
    p_CH4 = probas.get('CH4', 0.0)
    p_O3 = probas.get('O3', 0.0)
    p_H2O = probas.get('H2O', 0.0)

    # Base score is an average of the crucial life markers (O2, CH4, H2O)
    base_score = (p_O2 + p_CH4 + p_H2O) / 3.0

    # Bonus for Earth-like thermodynamic disequilibrium
    # (co-existence of O2/O3 and CH4)
    # They rapidly destroy each other, so co-presence means active
    # replenishment (life)
    disequilibrium_bonus = 0.0
    if (p_O2 > 0.5 or p_O3 > 0.5) and p_CH4 > 0.5:
        # Strong bonus
        disequilibrium_bonus = 0.3 * min(max(p_O2, p_O3), p_CH4)

    # Penalty if water is absent
    water_penalty = 0.0
    if p_H2O < 0.2:
        water_penalty = -0.2

    final_score = base_score + disequilibrium_bonus + water_penalty

    # Cap between 0 and 1
    final_score = max(0.0, min(1.0, final_score))

    # Determine confidence Category
    if final_score > 0.75:
        confidence = "HIGH"
    elif final_score > 0.4:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    return final_score, confidence
