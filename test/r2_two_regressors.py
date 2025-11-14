import numpy as np
from sklearn.metrics import r2_score

# Simulate data
n_samples = 1000
X = np.random.uniform(1000, 3000, n_samples)  # Square footage
boundary = 2000

# True regions
true_region = X < boundary

# Classifier predictions (85% accurate near boundary)
distance_to_boundary = np.abs(X - boundary)
misclass_prob = np.exp(-distance_to_boundary/200) * 0.15
predicted_region = true_region.copy()
predicted_region[np.random.random(n_samples) < misclass_prob] = ~true_region[np.random.random(n_samples) < misclass_prob]

# Make predictions
y_pred = np.zeros(n_samples)
for i in range(n_samples):
    if predicted_region[i]:  # Region A (small houses)
        y_pred[i] = regressor_A.predict(X[i])
    else:  # Region B (large houses)
        y_pred[i] = regressor_B.predict(X[i])

# Calculate total R²
total_r2 = r2_score(y_true, y_pred)