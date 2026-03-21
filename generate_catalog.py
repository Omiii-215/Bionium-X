import json
import random

# Core famous planets with specific data
core_planets = {
    "TRAPPIST-1e": {"star": "M-dwarf", "rad": 0.92, "temp": 251, "mol": {'O2': 0.1, 'CO2': 0.8, 'H2O': 0.2}},
    "TRAPPIST-1d": {"star": "M-dwarf", "rad": 0.78, "temp": 288, "mol": {'CO2': 0.6, 'H2O': 0.5}},
    "TRAPPIST-1f": {"star": "M-dwarf", "rad": 1.04, "temp": 219, "mol": {'CO2': 0.9, 'CH4': 0.05}},
    "K2-18b": {"star": "M-dwarf", "rad": 2.61, "temp": 265, "mol": {'CH4': 0.6, 'H2O': 0.4, 'CO2': 0.2}},
    "LHS 1140 b": {"star": "M-dwarf", "rad": 1.73, "temp": 226, "mol": {'CO2': 0.5, 'H2O': 0.5}},
    "Kepler-186f": {"star": "M-dwarf", "rad": 1.17, "temp": 188, "mol": {'CO2': 0.9}},
    "Kepler-452b": {"star": "G-type", "rad": 1.63, "temp": 265, "mol": {'CO2': 0.4, 'H2O': 0.3, 'O2': 0.1, 'CH4': 0.02}},
    "Proxima Centauri b": {"star": "M-dwarf", "rad": 1.07, "temp": 234, "mol": {'O2': 0.05, 'CO2': 0.7, 'CH4': 0.05}},
    "WASP-39b (Hot Jupiter)": {"star": "G-type", "rad": 1.27, "temp": 1166, "mol": {'CO2': 0.8, 'H2O': 0.6}},
    "Venus (Reference)": {"star": "G-type", "rad": 0.95, "temp": 737, "mol": {'CO2': 0.95}},
    "Earth (Reference)": {"star": "G-type", "rad": 1.00, "temp": 288, "mol": {'O2': 0.21, 'H2O': 0.05, 'CO2': 0.0004, 'CH4': 0.00002, 'O3': 0.00001}}
}

catalog = core_planets.copy()

random.seed(42)
stars = ["M-dwarf", "K-type", "G-type", "F-type"]

# Generate Kepler planets
for i in range(1, 80):
    name = f"Kepler-{i+200}b"
    rad = round(random.uniform(0.5, 3.5), 2)
    temp = random.randint(150, 1200)
    star = random.choice(stars)
    
    # Plausible molecules
    mol = {}
    if temp > 800:
        if random.random() > 0.3: mol['H2O'] = random.uniform(0.1, 0.8)
        if random.random() > 0.2: mol['CO2'] = random.uniform(0.1, 0.5)
    elif rad < 1.6: # Rocky
        mol['CO2'] = random.uniform(0.1, 0.9)
        if random.random() > 0.5: mol['H2O'] = random.uniform(0.01, 0.4)
        if 200 < temp < 320 and random.random() > 0.8: # Rare habitability
            mol['O2'] = random.uniform(0.05, 0.2)
            mol['CH4'] = random.uniform(0.01, 0.1)
    else: # Gas giant / Neptune
        mol['CH4'] = random.uniform(0.1, 0.8)
        mol['H2O'] = random.uniform(0.1, 0.5)
        if random.random() > 0.5: mol['CO2'] = random.uniform(0.05, 0.3)
        
    # clean up dict so no empty mol
    if not mol: mol['CO2'] = 0.1
    
    catalog[name] = {"star": star, "rad": rad, "temp": temp, "mol": mol}

# Generate TOI planets
for i in range(1, 50):
    name = f"TOI-{i+700}d"
    rad = round(random.uniform(0.8, 2.5), 2)
    temp = random.randint(200, 600)
    star = random.choices(["M-dwarf", "K-type"], weights=[0.7, 0.3])[0]
    
    mol = {}
    mol['CO2'] = random.uniform(0.1, 0.9)
    if random.random() > 0.4: mol['H2O'] = random.uniform(0.05, 0.6)
    if temp < 300 and random.random() > 0.7: mol['CH4'] = random.uniform(0.01, 0.3)
    if not mol: mol['CO2'] = 0.5
    
    catalog[name] = {"star": star, "rad": rad, "temp": temp, "mol": mol}

# Sort catalog alphabetically but keep Core Planets first
sorted_catalog = {k: catalog[k] for k in list(core_planets.keys()) + sorted(list(set(catalog.keys()) - set(core_planets.keys())))}

with open('data/exoplanet_catalog.json', 'w') as f:
    json.dump(sorted_catalog, f, indent=2)

print(f"Generated {len(sorted_catalog)} exoplanets.")
