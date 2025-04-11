import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
print("Loading dataset...")
df = pd.read_csv("data/KDDTrain+.csv", header=None)

# Define column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]
df.columns = columns

# Map label to binary: normal = 0, attack = 1
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# One-hot encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Separate features and target
X = df_encoded.drop('label', axis=1)
y = df_encoded['label']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model, scaler, and feature names
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Training complete. Model saved.")
