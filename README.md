# Bionium-X

An AI-based system for detecting biosignature molecules in exoplanet atmospheric spectra.

## Overview
Astrophysics researchers use transmission and emission spectra from telescopes like JWST and Hubble to analyze exoplanet atmospheres. This project uses machine learning to process these noisy signals and build models capable of automatically detecting biosignatures like Oxygen (O₂), Methane (CH₄), Ozone (O₃), Water (H₂O), and Carbon Dioxide (CO₂).

## Features
- **Synthetic Spectrum Generator**: Generates noisy mock exoplanet spectra with specific molecular absorption features.
- **Data Ingestion & Preprocessing**: Loads data (CSV, FITS, HDF5), filters noise, standardizes features.
- **Machine Learning Models**:
  - Baseline Random Forest Classifier
  - 1D Convolutional Neural Network
  - Spectral Transformer
- **Biosignature Scoring**: Estimates the probability of biological processes based on multi-molecule detections.
- **Visualization**: Streamlit web application for interactive exploration of spectra and model results.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Web App

```bash
streamlit run app.py
```
