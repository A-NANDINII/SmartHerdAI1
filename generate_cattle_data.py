import pandas as pd
import numpy as np

np.random.seed(42)

n_cows = 200
n_days = 30

data = []
for cow in range(1, n_cows+1):
    for day in range(1, n_days+1):
        temp = np.random.normal(38.5, 0.7)  # Normal cattle temp
        hr = np.random.normal(65, 10)
        activity = np.random.normal(50, 15)
        rumination = np.random.normal(400, 60)
        milk = np.random.normal(20, 5)
        # Simulate disease: 10% chance
        disease = 1 if np.random.rand() < 0.1 else 0
        # If sick, change features
        if disease:
            temp += np.random.uniform(0.5, 2.0)
            hr += np.random.uniform(10, 30)
            activity -= np.random.uniform(10, 30)
            rumination -= np.random.uniform(50, 150)
            milk -= np.random.uniform(5, 10)
        data.append([cow, day, temp, hr, activity, rumination, milk, disease])

df = pd.DataFrame(data, columns=[
    'cow_id', 'day', 'temperature', 'heart_rate', 'activity', 'rumination', 'milk_yield', 'disease'
])
df.to_csv('cattle_health_data.csv', index=False)
print("Dataset generated: cattle_health_data.csv")
