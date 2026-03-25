import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np

st.set_page_config(page_title="Brisbane Water Quality Detector", layout="wide")
st.title("🌊 Brisbane Water Quality Anomaly Detector")
st.markdown("**Research Demo** — 3 Unsupervised Models (Z-Score + Isolation Forest + Autoencoder)")

# Load models
scaler = joblib.load('scaler.pkl')
iso_model = joblib.load('isolation_forest.pkl')

class Autoencoder(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, 16))
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(),
                                     nn.Linear(32, 64), nn.ReLU(),
                                     nn.Linear(64, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae_model = Autoencoder()
ae_model.load_state_dict(torch.load('autoencoder.pth', weights_only=True, map_location='cpu'))
ae_model.eval()

features = ['Average Water Speed', 'Average Water Direction', 'Chlorophyll',
            'Temperature', 'Dissolved Oxygen', 'Dissolved Oxygen (%Saturation)',
            'pH', 'Salinity', 'Specific Conductance', 'Turbidity']

st.header("Enter New Sensor Reading")

input_data = {}
for feat in features:
    input_data[feat] = st.number_input(feat, value=5.0, format="%.4f")

if st.button("🔍 Run All 3 Models"):
    df_input = pd.DataFrame([input_data])
    scaled = scaler.transform(df_input)
    
    # Z-Score
    z_anomaly = 1 if np.abs(scaled).max() > 3 else 0
    
    # Isolation Forest
    iso_anomaly = 1 if iso_model.predict(scaled)[0] == -1 else 0
    
    # Autoencoder
    with torch.no_grad():
        recon = ae_model(torch.tensor(scaled, dtype=torch.float32))
        error = ((recon - torch.tensor(scaled, dtype=torch.float32))**2).mean().item()
    ae_anomaly = 1 if error > 0.8 else 0   # ← Change 0.8 to your actual threshold if needed
    
    st.subheader("Model Predictions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Z-Score", "🟥 Poor/Anomaly" if z_anomaly else "🟩 Good")
    col2.metric("Isolation Forest", "🟥 Poor/Anomaly" if iso_anomaly else "🟩 Good")
    col3.metric("Autoencoder", "🟥 Poor/Anomaly" if ae_anomaly else "🟩 Good")
    
    votes = z_anomaly + iso_anomaly + ae_anomaly
    if votes >= 2:
        st.error("🚨 HIGH CONFIDENCE ANOMALY — Poor Water Quality Detected")
    else:
        st.success("✅ Normal / Good Water Quality")
