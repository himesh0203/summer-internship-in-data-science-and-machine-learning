# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create sample dataset (instead of CSV)
np.random.seed(42)

data = pd.DataFrame({
    'age': np.random.randint(18, 60, 200),
    'income': np.random.randint(20000, 100000, 200),
    'spending_score': np.random.randint(1, 100, 200),
    'purchase': np.random.randint(0, 2, 200)  # Target (0 or 1)
})

# Features & target
X = data.drop("purchase", axis=1)
y = data["purchase"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Segmentation (K-Means)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['segment'] = kmeans.fit_predict(scaler.fit_transform(X))

# Prediction (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nSample Data with Segments:\n", data.head())



output - 

Model Accuracy: 50.0 %

Sample Data with Segments:
   age  income  spending_score  purchase  segment
0   56   82497              92         0        2
1   46   55987              71         1        1
2   32   87819              20         0        0
