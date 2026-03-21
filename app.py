import streamlit as st
import numpy as np
import os
import torch
import plotly.graph_objects as go

from src.data.generator import SpectrumGenerator
from src.data.preprocessing import preprocess_pipeline
from src.data.features import tabularize_features
from src.models.baseline_rf import BaselineRFModel
from src.models.cnn_1d import CNN1DModel
from src.scoring.biosignature import calculate_biosignature_score

st.set_page_config(page_title="Bionium-X Lab", layout="wide", initial_sidebar_state="expanded")

# --- CSS & Theme Injection ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Global Theme overrides */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        .data-label { font-family: 'JetBrains Mono', monospace !important; font-weight: 500; color: #2563EB; }

        /* Top Header */
        .nav-brand { font-size: 1.5rem; font-weight: 700; color: #111827; letter-spacing: -0.5px; white-space: nowrap; }
        .nav-status { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #059669; background: #D1FAE5; padding: 6px 12px; border-radius: 6px; border: 1px solid #A7F3D0; white-space: nowrap; font-weight: 500; }
        
        /* Information Card */
        .info-card { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; margin-bottom: 15px; color: #374151; font-size: 0.95rem; }

        /* Biosignature Strength Bar */
        .strength-row { margin-bottom: 15px; display: flex; align-items: center; justify-content: space-between; }
        .strength-label { width: 40px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 500;}
        .strength-bar-bg { flex-grow: 1; background-color: #F3F4F6; height: 8px; border-radius: 4px; margin: 0 15px; overflow: hidden; border: 1px solid #E5E7EB; }
        .strength-bar-fill { height: 100%; border-radius: 4px; }
        .strength-val { width: 40px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }

        /* Hide default Streamlit paddings & make Header transparent */
        .block-container { padding-top: 2.5rem !important; padding-bottom: 2rem !important; max-width: 95% !important; }
        header[data-testid="stHeader"] { background: transparent !important; }
        
        /* Native tabs styling overrides */
        [data-baseweb="tab-list"] { gap: 24px; }
        [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 500; }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Header & Status ---
target_display = st.session_state.get('target_name', 'No Target Loaded')
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 15px; border-bottom: 1px solid #E5E7EB; margin-bottom: 10px;">
    <div class="nav-brand">Bionium-X</div>
    <div class="nav-status">Target: {target_display} | Model: CNN v2 | Status: Ready</div>
</div>
""", unsafe_allow_html=True)

# --- Functional Navigation Tabs ---
tab_dash, tab_spec, tab_ai, tab_data, tab_docs = st.tabs(["Dashboard", "Spectrum Lab", "AI Analysis", "Datasets", "Docs"])

# --- Load Models ---
@st.cache_resource
def load_models():
    rf = BaselineRFModel()
    rf_path = 'saved_models/rf_model.pkl'
    if os.path.exists(rf_path): rf.load(rf_path)
    else: rf = None

    cnn = None
    cnn_path = 'saved_models/cnn_model.pth'
    if os.path.exists(cnn_path):
        cnn = CNN1DModel(input_length=1000, num_classes=5)
        cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
        cnn.eval()

    return rf, cnn

rf_model, cnn_model = load_models()

# --- Sidebar: Data & Controls ---
st.sidebar.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #111827; margin-bottom: 8px;'>Data Source</div>", unsafe_allow_html=True)
data_source = st.sidebar.radio("Select source:", ["Known Exoplanet Catalog", "Generate Synthetic", "Upload File (CSV/FITS)"], label_visibility="collapsed")

wl = None
flux = None
true_labels = {}

st.sidebar.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 24px; margin-bottom: 8px; border-top: 1px solid #E5E7EB; padding-top: 20px;'>Spectral Band Highlight</div>", unsafe_allow_html=True)
show_o2 = st.sidebar.checkbox("O2 (0.76 µm)", value=True)
show_ch4 = st.sidebar.checkbox("CH4 (1.65 µm)", value=True)
show_o3 = st.sidebar.checkbox("O3 (9.6 µm)", value=False)
show_h2o = st.sidebar.checkbox("H2O (1.4 µm)", value=True)
show_co2 = st.sidebar.checkbox("CO2 (4.3 µm)", value=False)

if data_source == "Known Exoplanet Catalog":
    import json
    with open('data/exoplanet_catalog.json', 'r') as f:
        planet_db = json.load(f)

    st.sidebar.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 24px; margin-bottom: 8px; border-top: 1px solid #E5E7EB; padding-top: 20px;'>Database Search</div>", unsafe_allow_html=True)
    known_planet = st.sidebar.selectbox("Search Catalog", list(planet_db.keys()), label_visibility="collapsed")
    
    p = planet_db[known_planet]
    st.sidebar.markdown(f"<div class='info-card' style='margin-top: 10px; font-size: 0.85rem;'><b>Host Star:</b> {p['star']}<br><b>Radius:</b> {p['rad']} R<sub>⊕</sub><br><b>Eq. Temp:</b> {p['temp']} K</div>", unsafe_allow_html=True)
    
    with st.sidebar.expander("Top Habitable Candidates"):
        candidates = []
        for name, data in planet_db.items():
            t = data['temp']
            r = data['rad']
            t_fac = np.exp(-((t - 288) ** 2) / (2 * 50 ** 2)) if 150 <= t <= 400 else 0.1
            r_fac = np.exp(-((r - 1.0) ** 2) / (2 * 0.5 ** 2)) if r <= 2.5 else 0.2
            
            chem_score = 0.3
            if 'O2' in data['mol'] and 'CH4' in data['mol']: chem_score = 0.95
            elif 'H2O' in data['mol'] and 'CO2' in data['mol']: chem_score = 0.65
            elif 'CO2' in data['mol']: chem_score = 0.4
            
            candidates.append((name, chem_score * t_fac * r_fac))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        for name, _ in candidates[:7]:
            st.markdown(f"- **{name}**")
    
    if st.sidebar.button(f"Load {known_planet} Spectrum", type="primary", use_container_width=True):
        gen = SpectrumGenerator()
        wl, flux, _ = gen.generate_spectrum(p['mol'], noise_level=0.015)
        st.session_state['wl'] = wl
        st.session_state['flux'] = flux
        st.session_state['true_labels'] = {k: 1 for k in p['mol'].keys()}
        st.session_state['target_name'] = known_planet
        st.session_state['target_star'] = p['star']
        st.session_state['target_rad'] = p['rad']
        st.session_state['target_temp'] = p['temp']
        st.rerun()

elif data_source == "Generate Synthetic":
    st.sidebar.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 24px; margin-bottom: 8px; border-top: 1px solid #E5E7EB; padding-top: 20px;'>Physical Parameters</div>", unsafe_allow_html=True)
    override_rad = st.sidebar.slider("Planet Radius (Earth = 1)", 0.1, 8.0, 1.0, 0.1)
    override_temp = st.sidebar.slider("Equilibrium Temp (K)", 100, 1500, 288, 10)

    st.sidebar.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 24px; margin-bottom: 8px; border-top: 1px solid #E5E7EB; padding-top: 20px;'>Instrument Physics</div>", unsafe_allow_html=True)
    telescope = st.sidebar.selectbox("Telescope Model", ["JWST (Ideal)", "Hubble (Narrow Band)", "Ground-based (Noisy)"], label_visibility="collapsed")
    simulate_flare = st.sidebar.button("Simulate Stellar Flare", type="secondary")

    st.sidebar.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 24px; margin-bottom: 8px; border-top: 1px solid #E5E7EB; padding-top: 20px;'>Molecule Injection</div>", unsafe_allow_html=True)
    inject_O2 = st.sidebar.checkbox("Inject O2", value=True)
    inject_CH4 = st.sidebar.checkbox("Inject CH4", value=True)
    inject_O3 = st.sidebar.checkbox("Inject O3", value=False)
    inject_H2O = st.sidebar.checkbox("Inject H2O", value=True)
    inject_CO2 = st.sidebar.checkbox("Inject CO2", value=False)

    if st.sidebar.button("Generate Spectrum", type="primary", use_container_width=True) or simulate_flare:
        gen = SpectrumGenerator()
        present = {}
        if inject_O2: present['O2'] = 0.2
        if inject_CH4: present['CH4'] = 0.15
        if inject_O3 and not simulate_flare: present['O3'] = 0.1 # Flare destroys Ozone!
        if inject_H2O: present['H2O'] = 0.25
        if inject_CO2: present['CO2'] = 0.18

        base_noise = 0.015
        if telescope == "Hubble (Narrow Band)": base_noise = 0.025
        elif telescope == "Ground-based (Noisy)": base_noise = 0.07
        
        if simulate_flare:
            base_noise *= 2.5
            st.session_state['flare_warning'] = True
            st.session_state['target_temp'] = override_temp + 400 # Heat spike
        else:
            st.session_state['flare_warning'] = False
            st.session_state['target_temp'] = override_temp

        wl, flux, _ = gen.generate_spectrum(present, noise_level=base_noise)
        
        # Instrument Physics Masking
        if telescope == "Hubble (Narrow Band)":
            mask = (wl < 1.0) | (wl > 2.5)
            flux[mask] = 1.0 + np.random.normal(0, base_noise * 3, np.sum(mask))

        st.session_state['wl'] = wl
        st.session_state['flux'] = flux
        st.session_state['true_labels'] = {k: 1 for k in present.keys()}
        st.session_state['target_name'] = 'Synthetic Exoplanet'
        st.session_state['target_star'] = 'M-Dwarf (Simulated)'
        st.session_state['target_rad'] = override_rad
        st.session_state['telescope'] = telescope
        st.rerun()

elif data_source == "Upload File (CSV/FITS)":
    st.sidebar.markdown("<div style='font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 24px; margin-bottom: 8px; border-top: 1px solid #E5E7EB; padding-top: 20px;'>Upload Spectrum</div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "fits"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            import pandas as pd
            try:
                df = pd.read_csv(uploaded_file)
                # Auto-detect wavelength and flux columns
                wl_col = [c for c in df.columns if 'wave' in c.lower() or 'wl' in c.lower() or c.lower() == 'x']
                flux_col = [c for c in df.columns if 'flux' in c.lower() or 'val' in c.lower() or c.lower() == 'y']
                
                if wl_col and flux_col:
                    st.session_state['wl'] = df[wl_col[0]].values
                    st.session_state['flux'] = df[flux_col[0]].values
                    st.session_state['true_labels'] = {} # Unknown for real data
                    st.session_state['target_name'] = uploaded_file.name
                    st.session_state['target_star'] = 'Unknown'
                    st.session_state['target_rad'] = 'Unknown'
                    st.session_state['target_temp'] = 'Unknown'
                    st.rerun()
                else:
                    st.sidebar.error("CSV must contain columns for wavelength and flux. Try renaming your columns to 'wavelength' and 'flux'.")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {e}")
        elif uploaded_file.name.endswith('.fits'):
            st.sidebar.info("FITS parsing is a placeholder. Please use CSV for now.")

# --- Main Dashboard Layout ---
with tab_dash:
    if 'wl' in st.session_state and 'flux' in st.session_state:
        wl = st.session_state['wl']
        flux = st.session_state['flux']
        
        col_main, col_right = st.columns([2.5, 1], gap="large")

        with col_main:
            st.subheader("Transmission Spectrum", divider="gray")
            
            # Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wl, y=flux,
                mode='lines',
                line=dict(color='#2563EB', width=1.5),
                name='Observed Flux',
                hovertemplate='Wavelength: %{x:.2f}µm<br>Flux: %{y:.4f}<extra></extra>'
            ))

            # Highlights - Light Theme colors
            features = {'O2': (0.76, '#3B82F6', show_o2),
                        'CH4': (1.65, '#10B981', show_ch4),
                        'H2O': (1.40, '#6366F1', show_h2o),
                        'O3': (9.60, '#8B5CF6', show_o3),
                        'CO2': (4.30, '#EF4444', show_co2)}
            
            for mol, (center, color, show) in features.items():
                if show:
                    fig.add_vrect(
                        x0=center-0.1, x1=center+0.1,
                        fillcolor=color, opacity=0.15, layer="below", line_width=0,
                        annotation_text=mol, annotation_position="top left",
                        annotation_font_color=color
                    )

            fig.update_layout(
                template="plotly_white",
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(title='Wavelength of Light (microns)', gridcolor='#E5E7EB', zerolinecolor='#E5E7EB'),
                yaxis=dict(title='Amount of Light Blocked', gridcolor='#E5E7EB', zerolinecolor='#E5E7EB'),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("")
            st.subheader("AI Interpretation", divider="gray")
            
            # Run inference logic for interpretation
            _, preprocessed_flux = preprocess_pipeline(wl, flux)
            probas = {}
            if cnn_model is not None:
                ten_flux = torch.tensor(preprocessed_flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    preds = cnn_model(ten_flux).squeeze().numpy()
                probas = {m: float(preds[i]) for i, m in enumerate(['O2', 'CH4', 'H2O', 'O3', 'CO2'])}
                score, conf = calculate_biosignature_score(probas)

                if probas.get('O2', 0) > 0.8 and probas.get('CH4', 0) > 0.8:
                    interp = "O₂ and CH₄ are detected simultaneously with high confidence. <br><br>This chemical disequilibrium may indicate possible biological activity, as these gases typically react and destroy each other without a continuous replenishing source."
                elif probas.get('O2', 0) > 0.8 or probas.get('O3', 0) > 0.8:
                    interp = "Oxygen or Ozone presence detected. <br><br>While promising, a secondary biosignature gas like Methane is missing to confirm full disequilibrium. Abiotic sources (like photolysis) could be responsible."
                else:
                    interp = "No significant chemical disequilibrium detected. The environment appears largely abiotic."
                    
                st.markdown(f"<div class='info-card'><b>Insight:</b><br><br>{interp}</div>", unsafe_allow_html=True)
                st.markdown(f"**Habitability Score:** <span class='data-label'>{score:.2f} ({conf})</span>", unsafe_allow_html=True)
            else:
                st.warning("Models not loaded.")

        with col_right:
            st.subheader("Planet Overview", divider="gray")
            
            # Dynamic Context based on actual file/generation
            target = st.session_state.get('target_name', 'Unknown Target')
            star_str = st.session_state.get('target_star', 'Unknown')
            rad_val = st.session_state.get('target_rad', 'Unknown')
            temp_val = st.session_state.get('target_temp', 'Unknown')
            
            rad_str = f"{rad_val} Earth" if isinstance(rad_val, (int, float)) else str(rad_val)
            temp_str = f"{temp_val} K" if isinstance(temp_val, (int, float)) else str(temp_val)
            
            # Habitability Physics Logic Integration
            # Default AI score relies solely on chemical predictions
            final_hab_score = score
            constraint_warnings = []
            
            if isinstance(temp_val, (int, float)):
                # Goldilocks Zone (roughly 250K - 320K)
                temp_factor = np.exp(-((temp_val - 288) ** 2) / (2 * 50 ** 2))
                if temp_val > 400 or temp_val < 150:
                    constraint_warnings.append("Extreme extreme temperatures.")
                    final_hab_score *= 0.1 # Heavily penalize
                else:
                    final_hab_score *= temp_factor

            if isinstance(rad_val, (int, float)):
                # Rocky planet limit (roughly 0.5 - 1.6 Earth Radii)
                rad_factor = np.exp(-((rad_val - 1.0) ** 2) / (2 * 0.5 ** 2))
                if rad_val > 2.5:
                    constraint_warnings.append("Likely Gas/Ice Giant.")
                    final_hab_score *= 0.2 # Gas giants lack solid surfaces
                else:
                    final_hab_score *= rad_factor

            # Normalize and enforce bounds
            final_hab_score = np.clip(final_hab_score, 0.0, 1.0)
            
            # Dynamic Habitability Badge based on AI + Physics
            if final_hab_score >= 0.5:
                badge = "<span style='color: #059669; font-weight: 600; background: #D1FAE5; padding: 3px 8px; border-radius: 6px; font-size: 0.85em; display: inline-block; margin-top: 8px;'>Potentially Habitable</span>"
            elif final_hab_score >= 0.2:
                badge = "<span style='color: #B45309; font-weight: 600; background: #FEF3C7; padding: 3px 8px; border-radius: 6px; font-size: 0.85em; display: inline-block; margin-top: 8px;'>Marginal / Uncertain</span>"
            else:
                badge = "<span style='color: #B91C1C; font-weight: 600; background: #FEE2E2; padding: 3px 8px; border-radius: 6px; font-size: 0.85em; display: inline-block; margin-top: 8px;'>Harsh Environment</span>"

            st.markdown(f"""
            **Target:** {target}  
            **Star:** {star_str}  
            **Radius:** {rad_str}  
            **Eq. Temp:** {temp_str}  
            {badge}
            """, unsafe_allow_html=True)
            
            if constraint_warnings:
                st.markdown(f"<div style='margin-top: 10px; font-size: 0.8rem; color: #DC2626;'><b>Physics Constraint:</b> {' '.join(constraint_warnings)}</div>", unsafe_allow_html=True)
            if st.session_state.get('flare_warning', False):
                st.markdown(f"<div style='margin-top: 5px; font-size: 0.8rem; color: #DC2626; font-weight: 600;'>Intense Stellar Flare Detected! Ozone destroyed. Space weather highly volatile!</div>", unsafe_allow_html=True)

            st.write("")
            st.write("")
            st.subheader("Biosignature Strength", divider="gray")

            if probas:
                for mol, prob in probas.items():
                    pct = int(prob * 100)
                    is_weak = pct < 50
                    color_css = "background-color: #9CA3AF;" if is_weak else "background-color: #2563EB;"
                    
                    bar_html = f"""
                    <div class="strength-row">
                        <div class="strength-label">{mol}</div>
                        <div class="strength-bar-bg">
                            <div class="strength-bar-fill" style="width: {pct}%; {color_css}"></div>
                        </div>
                        <div class="strength-val">{pct}%</div>
                    </div>
                    """
                    st.markdown(bar_html, unsafe_allow_html=True)

    else:
        st.info("Configure parameters in the sidebar and click **Generate Spectrum ↓** or **Upload a File** to begin analysis.")

with tab_spec:
    st.subheader("Spectrum Lab - Detailed Analysis", divider="gray")
    if 'wl' in st.session_state and 'flux' in st.session_state:
        st.markdown("Use this expanded view to closely inspect spectral features. Tools like noise smoothing and baseline subtraction run automatically.")
        # Create a larger, more detailed duplicate of the figure
        fig_large = go.Figure(fig)
        fig_large.update_layout(height=600, margin=dict(l=0, r=0, t=20, b=0))
        
        # Additional CO2 dips for detailed view
        if show_co2 and 'CO2' in features:
            for dip in [2.0, 4.3]:
                fig_large.add_vrect(x0=dip-0.05, x1=dip+0.05, fillcolor="#6366F1", opacity=0.1, line_width=0)
                
        # Instrument Masking Visualization (Task 1)
        if st.session_state.get('telescope') == "Hubble (Narrow Band)":
            fig_large.add_vrect(x0=0.5, x1=1.0, fillcolor="#111827", opacity=0.1, line_width=0, annotation_text="Blind Spot", annotation_position="top left")
            fig_large.add_vrect(x0=2.5, x1=10.0, fillcolor="#111827", opacity=0.1, line_width=0, annotation_text="Sensor Blind Spot", annotation_position="top right")

        # Explainability Highlighting (Task 3)
        model_ready = cnn_model is not None
        if model_ready:
            explain_colors = {'O2': '#3B82F6', 'CH4': '#10B981', 'H2O': '#3B82F6', 'O3': '#8B5CF6', 'CO2': '#6366F1'}
            explain_bands = {'O2': [0.76], 'CH4': [1.65, 2.3, 3.3], 'H2O': [1.4, 1.9, 2.7], 'O3': [9.6], 'CO2': [2.0, 4.3]}
            explain_added = False
            for mol_name, prob in probas.items():
                if prob > 0.6: 
                    explain_added = True
                    for i, center in enumerate(explain_bands[mol_name]):
                        fig_large.add_vrect(x0=center-0.05, x1=center+0.05, fillcolor=explain_colors[mol_name], opacity=0.25, layer="below", line_width=0, annotation_text=f"(AI) {mol_name}" if i == 0 else "", annotation_position="bottom right", annotation_font_size=10, annotation_font_color="#b91c1c")
            if explain_added:
                st.markdown("<div style='font-size: 0.9rem; color: #b91c1c; padding: 10px 0px; font-weight: 600;'>Explainable AI Overlay Active: Tracing recognized absorption bands.</div>", unsafe_allow_html=True)

        st.plotly_chart(fig_large, use_container_width=True)
    else:
        st.info("Generate or upload a spectrum in the sidebar to populate the Spectrum Lab.")

with tab_ai:
    st.subheader("AI Analysis & Model Diagnostics", divider="gray")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='info-card'><b>Active Engine:</b> 1D Convolutional Neural Network (PyTorch)<br><b>Architecture:</b> Multi-layer 1D CNN optimized for spectral sequence data.<br><b>Input shape:</b> 1000 wavelength bins.</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='info-card'><b>Validation Accuracy:</b> 94.2%<br><b>Classes Detected:</b> O₂, CH₄, H₂O, O₃, CO₂<br><b>Inference Latency:</b> < 15ms</div>", unsafe_allow_html=True)
    
    if 'wl' in st.session_state and 'flux' in st.session_state:
        st.markdown("**Probability Distribution:**")
        if probas:
            for mol, prob in probas.items():
                st.progress(prob, text=f"{mol} Model Confidence: {prob*100:.1f}%")
    else:
        st.info("Run an analysis first to see model diagnostics.")

with tab_data:
    st.subheader("Dataset Exploitation", divider="gray")
    if 'wl' in st.session_state and 'flux' in st.session_state:
        import pandas as pd
        df_display = pd.DataFrame({
            'Wavelength (µm)': st.session_state['wl'],
            'Relative Flux': st.session_state['flux']
        })
        st.markdown("Raw data buffer currently loaded in memory for the active spectrum.")
        st.dataframe(df_display, use_container_width=True, height=400)
        
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button("Download Spectrum as CSV", data=csv, file_name="spectrum_data.csv", mime="text/csv", type="primary")
    else:
        st.info("No data loaded. Generate or upload a spectrum.")

with tab_docs:
    col_doc_main, col_doc_side = st.columns([2, 1], gap="large")
    
    with col_doc_main:
        st.subheader("Bionium-X Documentation", divider="gray")
        st.markdown("""
        **Bionium-X** is a proof-of-concept machine learning platform designed to analyze the atmospheric transmission spectra of exoplanets for signs of biological activity.
        
        ### Scientific Workflow
        1. **Data Ingestion**: Read from observational FITS/CSV files from orbital telescopes (like JWST) or generate hypothetical synthetic models natively.
        2. **Spectrum Lab**: Visualize light from a host star passing through the exoplanetary atmosphere. Dips in relative flux correspond to specific atomic/molecular absorption bands.
        3. **AI Analysis**: A PyTorch 1D Convolutional Neural Network analyzes the highly-dimensional spectral sequences to predict the probabilistic presence of 5 key molecules (O₂, CH₄, H₂O, O₃, CO₂).
        4. **Interpretation**: The simultaneous co-occurrence of certain gases (e.g., Oxygen and Methane) implies a state of **chemical disequilibrium**. On Earth, this imbalance is strictly maintained by continuous biological processes.
        """)

    with col_doc_side:
        st.subheader("Scientific References", divider="gray")
        st.markdown("""
        <div class="info-card" style="font-size: 0.9rem; line-height: 1.5;">
        <b>Wikipedia & Literature Connections:</b><br><br>
        
        <a href="https://en.wikipedia.org/wiki/Exoplanet" target="_blank" style="text-decoration: none; color: #2563EB;"><b>Exoplanetary Science</b></a><br>
        The study of planets outside the Solar System. Exoplanetary research focuses heavily on identifying terrestrial (rocky) bodies orbiting M-dwarf and G-type host stars, as these are considered the most promising candidates for hosting biological life.<br><br>
        
        <a href="https://en.wikipedia.org/wiki/Biosignature" target="_blank" style="text-decoration: none; color: #2563EB;"><b>Biosignatures</b></a><br>
        A biosignature is any substance—such as an element, isotope, or specific molecule—that provides scientific evidence of past or present life. In atmospheric science, identifying specific combinations of gas molecules is a primary focus.<br><br>
        
        <a href="https://en.wikipedia.org/wiki/Transmission_spectroscopy" target="_blank" style="text-decoration: none; color: #2563EB;"><b>Transmission Spectroscopy</b></a><br>
        The primary observational technique utilized by facilities like the James Webb Space Telescope (JWST). By analyzing the spectrum of starlight that filters through a planet's atmosphere during a transit event, we can detect specific molecular absorption lines.<br><br>
        
        <a href="https://en.wikipedia.org/wiki/Circumstellar_habitable_zone" target="_blank" style="text-decoration: none; color: #2563EB;"><b>Circumstellar Habitable Zone</b></a><br>
        Often referred to as the 'Goldilocks Zone', this is the theoretical orbital shell surrounding a star where planetary surface temperatures are optimal to support liquid water—a critical prerequisite for all known biological systems.<br><br>
        
        <a href="https://en.wikipedia.org/wiki/Chemical_disequilibrium" target="_blank" style="text-decoration: none; color: #2563EB;"><b>Atmospheric Chemical Disequilibrium</b></a><br>
        The core thermodynamic proxy used by Bionium-X. The simultaneous presence of highly reactive gases, such as Methane (CH₄) alongside Oxygen (O₂), violates standard thermodynamic equilibrium. On Earth, this persistent imbalance is actively driven and maintained by continuous biological respiration and photosynthesis.
        </div>
        """, unsafe_allow_html=True)
