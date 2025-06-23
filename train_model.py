import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

DATA_DIR = 'data'
X = []
y = []

# Load data
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith('.npy'):
                data = np.load(os.path.join(label_path, file))
                if data.shape == (63,):  # Ensure correct shape
                    X.append(data)
                    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}, Labels: {len(y)}")
print(f"Unique labels: {set(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f" Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open('sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(" Model saved as sign_model.pkl")
