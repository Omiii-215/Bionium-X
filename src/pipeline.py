import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.data.generator import SpectrumGenerator
from src.data.preprocessing import preprocess_pipeline
from src.data.features import tabularize_features
from src.models.baseline_rf import BaselineRFModel
from src.models.cnn_1d import CNN1DModel


def run_pipeline(n_samples=500, save_dir='saved_models'):
    print("--- Bionium-X Pipeline ---")
    print(f"1. Generating {n_samples} synthetic spectra...")
    gen = SpectrumGenerator(wl_min=0.5, wl_max=10.0, num_points=1000)
    wavelengths, df_flux, df_labels = gen.generate_dataset(n_samples=n_samples)

    print("2. Preprocessing data...")
    flux_arr = df_flux.values
    labels_arr = df_labels.values
    preprocessed_flux = []
    for f in flux_arr:
        _, pf = preprocess_pipeline(wavelengths, f)
        preprocessed_flux.append(pf)
    preprocessed_flux = np.array(preprocessed_flux)

    print("3. Extracting tabular features for Random Forest...")
    df_features = tabularize_features(wavelengths, preprocessed_flux)

    # Split datasets
    print("4. Splitting datasets...")
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        df_features, df_labels, test_size=0.2, random_state=42)
    X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
        preprocessed_flux, labels_arr, test_size=0.2, random_state=42)

    # --- Random Forest ---
    print("\n--- Training Random Forest Baseline ---")
    rf = BaselineRFModel()
    rf.train(X_train_rf, y_train_rf)
    metrics_rf = rf.evaluate(X_test_rf, y_test_rf)
    print("RF Metrics:")
    for k, v in metrics_rf.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    rf.save(os.path.join(save_dir, 'rf_model.pkl'))

    # --- 1D CNN ---
    print("\n--- Training 1D CNN ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = CNN1DModel(
        input_length=preprocessed_flux.shape[1],
        num_classes=5).to(device)

    # Prepare DataLoader
    batch_size = 32
    train_dataset = TensorDataset(
        torch.tensor(
            X_train_dl, dtype=torch.float32).unsqueeze(1), torch.tensor(
            y_train_dl, dtype=torch.float32))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    cnn.train()
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            probas = cnn(batch_x)
            loss = criterion(probas, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.4f}")

    # Evaluate CNN (simple precision/recall via threshold)
    cnn.eval()
    with torch.no_grad():
        X_test_ten = torch.tensor(
            X_test_dl, dtype=torch.float32).unsqueeze(1).to(device)
        test_probas = cnn(X_test_ten).cpu().numpy()

    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score
    )
    y_pred_dl = (test_probas > 0.5).astype(int)
    print("CNN Metrics:")
    p_score = precision_score(y_test_dl, y_pred_dl, average='macro', zero_division=0)
    print(f"  precision: {p_score:.4f}")

    r_score = recall_score(y_test_dl, y_pred_dl, average='macro', zero_division=0)
    print(f"  recall: {r_score:.4f}")

    f_score = f1_score(y_test_dl, y_pred_dl, average='macro', zero_division=0)
    print(f"  f1: {f_score:.4f}")

    roc = roc_auc_score(y_test_dl, test_probas, average='macro')
    print(f"  roc_auc: {roc:.4f}")

    torch.save(cnn.state_dict(), os.path.join(save_dir, 'cnn_model.pth'))

    print("\nPipeline completed successfully! Models saved to", save_dir)


if __name__ == "__main__":
    run_pipeline()
