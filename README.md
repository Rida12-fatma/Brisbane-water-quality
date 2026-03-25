# 🌊 Brisbane Water Quality Anomaly Detection

**Research Project** — Pure Unsupervised Anomaly Detection using Time-Series Environmental Sensor Data

### Overview
This project implements **unsupervised anomaly detection** on the Brisbane Water Quality dataset to identify poor water quality conditions without any ground-truth labels.

Three models are used:
- **Z-Score** (simple statistical baseline)
- **Isolation Forest** (ensemble-based anomaly detection)
- **Autoencoder** (deep learning reconstruction error-based detection)

Additionally, robust statistical methods (IQR and Sliding Window with MAD) are included for comparison.

### Key Features
- Purely unsupervised (no labels used during training)
- IQR capping for outlier robustness
- Timestamp handling and temporal feature engineering
- Domain validation: Anomalies show significantly lower Dissolved Oxygen and higher Turbidity
- Interactive web demo with all 3 models running in real-time
- Cross-model agreement for higher confidence predictions

### Models Prediction Logic
Since this is **unsupervised learning**, the models do **not** predict traditional class labels. Instead:
- They learn the "normal" pattern of water quality from the data.
- Any reading that deviates significantly from the learned normal pattern is flagged as **Anomaly = Poor Water Quality**.
- Final decision uses majority voting across the 3 models for reliability.

### Repository Contents
- `app.py` → Streamlit web application (live demo)
- `scaler.pkl` → Fitted StandardScaler
- `isolation_forest.pkl` → Trained Isolation Forest model
- `autoencoder.pth` → Trained PyTorch Autoencoder model
- `requirements.txt` → Python dependencies
- `sample_data.csv` → Sample data for testing (optional)
- Jupyter Notebook (full analysis, preprocessing, and model training)

### How to Run the Web Demo Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/brisbane-water-quality-anomaly.git
cd brisbane-water-quality-anomaly

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
