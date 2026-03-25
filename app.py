import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np

st.set_page_config(page_title="Brisbane Water Quality Detector", layout="wide")
st.title("🌊 Brisbane Water Quality Anomaly Detector")
st.markdown("**Research Demo** — 3 Unsupervised Models + Feature Explanation")

# Load models
scaler = joblib.load('scaler.pkl')
iso_model = joblib.load('isolation_forest.pkl')
rf_surrogate = joblib.load('surrogate_rf.pkl')   # ← New: Surrogate Random Forest

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

if st.button("🔍 Detect Anomaly & Explain"):
    df_input = pd.DataFrame([input_data])
    scaled = scaler.transform(df_input)
    
    # 1. Z-Score
    z_anomaly = 1 if np.abs(scaled).max() > 3 else 0
    
    # 2. Isolation Forest
    iso_anomaly = 1 if iso_model.predict(scaled)[0] == -1 else 0
    
    # 3. Autoencoder
    with torch.no_grad():
        recon = ae_model(torch.tensor(scaled, dtype=torch.float32))
        error = ((recon - torch.tensor(scaled, dtype=torch.float32))**2).mean().item()
    ae_anomaly = 1 if error > 0.8 else 0   # Update threshold if needed
    
    # Show predictions
    st.subheader("Anomaly Detection Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Z-Score", "🟥 Poor" if z_anomaly else "🟩 Good")
    col2.metric("Isolation Forest", "🟥 Poor" if iso_anomaly else "🟩 Good")
    col3.metric("Autoencoder", "🟥 Poor" if ae_anomaly else "🟩 Good")
    
    # Majority vote
    votes = z_anomaly + iso_anomaly + ae_anomaly
    if votes >= 2:
        st.error("🚨 HIGH CONFIDENCE ANOMALY — Poor Water Quality")
    else:
        st.success("✅ Normal Water Quality")
    
    # 4. Feature Explanation using Surrogate Random Forest
    if votes >= 2:   # Only explain if anomaly is detected
        st.subheader("🔍 Why is this Anomaly? (Feature Importance)")
        importance = pd.Series(rf_surrogate.feature_importances_, index=features).sort_values(ascending=False)
        
        expl_df = pd.DataFrame({
            'Feature': importance.index,
            'Importance': importance.values.round(4)
        })
        st.dataframe(expl_df.style.format({"Importance": "{:.4f}"}))
        
        st.info("Top features contributing to this anomaly: " + ", ".join(importance.head(3).index.tolist()))
