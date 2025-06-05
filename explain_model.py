import pandas as pd
import tensorflow as tf
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load data and model
df = pd.read_csv('cattle_health_with_anomalies.csv')
features = ['temperature', 'heart_rate', 'activity', 'rumination', 'milk_yield']
X = df[features]

# Load scaler and model
scaler = joblib.load('scaler.save')
X_scaled = scaler.transform(X)
model = tf.keras.models.load_model('cattle_health_model.h5')

# Use a subset for SHAP
background = X_scaled[:100]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_scaled[:10])

# Fix shape mismatch
shap_values = np.array(shap_values).squeeze()  # Remove singleton dimensions
X_plot = X.iloc[:10].values  # Ensure (10, 5) shape

# Plot SHAP summary
shap.summary_plot(
    shap_values, 
    X_plot,
    feature_names=features,
    plot_type="dot",
    show=True
)
plt.show()
