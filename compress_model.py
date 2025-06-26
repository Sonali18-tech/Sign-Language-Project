
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 21:36:07 2025

@author: capl2
"""

import pickle
import gzip
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# --- Replace this section with your actual model loading and training ---
# For demonstration, let's create a dummy model
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X, y)
# --- End of replacement section ---

# Save with gzip compression
# This will create 'sign_model.pkl.gz' in the same directory as this script
with gzip.open('sign_model.pkl.gz', 'wb') as f:
    pickle.dump(model, f)


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 21:36:07 2025

@author: capl2
"""

import pickle
import gzip
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# --- Replace this section with your actual model loading and training ---
# For demonstration, let's create a dummy model
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X, y)
# --- End of replacement section ---

# Save with gzip compression
# This will create 'sign_model.pkl.gz' in the same directory as this script
with gzip.open('sign_model.pkl.gz', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as sign_model.pkl.gz")