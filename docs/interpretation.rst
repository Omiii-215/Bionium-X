Interpretation Guide
====================

AI Model Confidence
-------------------
Underlying classification metrics utilize probabilistic floats derived via typical Softmax normalization across 5 target arrays. Values above `0.5` denote nominal detection confidence with `> 0.8` interpreted as a rigorous structural confirmed signature within the data stream.

Chemical Disequilibrium
-----------------------
If `O₂ Prob > 0.8` AND `CH₄ Prob > 0.8`, Bionium-X issues a **Simultaneous Biosignature Alert**. In standard planetary science, atmospheric oxygen and methane react rapidly, depleting each other. Identifying both concurrently serves as a near-absolute prerequisite for continuous organic atmospheric replacement.

Habitability Scoring
--------------------
AI probabilistic scores undergo severe constraint normalization based strictly upon:

- **Temperature Constraints**: The Goldilocks boundary heavily aggressively penalizes temperatures outside `250K - 320K`. Target bodies experiencing `>400K` inherently generate a 90% viability penalty regardless of atmospheric constituents.
- **Physical Size Defaults**: Scores are scaled down for radii `> 2.5` Earths to mitigate false positives potentially spawned from Gas Giant/Ice Giant readings lacking solid state surface interfaces for carbon-based evolution.
