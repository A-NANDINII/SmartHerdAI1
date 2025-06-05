import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import numpy as np

# Load cattle health data
df = pd.read_csv('cattle_health_data.csv')
features = ['temperature', 'heart_rate', 'activity', 'rumination', 'milk_yield']

# Standardize features for fair distance calculation
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Calculate the "herd centroid" (mean of all cows' health features)
herd_centroid = np.mean(X, axis=0)

# Calculate anomaly score for each record (distance from centroid)
df['anomaly_score'] = [euclidean(x, herd_centroid) for x in X]

# Mark top 5% as anomalies (outliers in the herd)
threshold = df['anomaly_score'].quantile(0.95)
df['herd_anomaly'] = (df['anomaly_score'] > threshold).astype(int)

# Save the new data with anomaly labels
df.to_csv('cattle_health_with_anomalies.csv', index=False)
print("Anomaly detection complete. Results saved to cattle_health_with_anomalies.csv")
