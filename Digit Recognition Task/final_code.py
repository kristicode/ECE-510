import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the provided datasets
features_path = "/home/kristi/Desktop/Research progress/Winter 2025 term/JoSIM-master/merged.csv"
labels_path = "//home/kristi/Desktop/Research progress/Winter 2025 term/JoSIM-master/labels.csv"

# Read the files
features_df = pd.read_csv(features_path)
labels_df = pd.read_csv(labels_path)

# Display the first few rows of each dataset
features_df.head(), labels_df.head()
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Clean and structure the features dataset
features_df.columns = features_df.iloc[0]  # Set proper column names
features_df = features_df[1:].reset_index(drop=True)  # Remove the duplicated header row

# Convert to numerical format
features_df = features_df.apply(pd.to_numeric, errors='coerce')

# Extract features (X) and labels (y)
X = features_df.iloc[:, 1:].values  # Exclude the first column if it's an index
y = labels_df.iloc[:, 1].values     # Assuming second column holds the labels

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Ridge Regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predict outputs
y_pred = ridge.predict(X_test)

# Return model coefficients and sample predictions
ridge.coef_, y_test[:10], y_pred[:10]
# Check row counts
num_features = X.shape[0]
num_labels = y.shape[0]

num_features, num_labels

# Trim the labels to match the feature count
y = y[:num_features]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Ridge Regression model
ridge.fit(X_train, y_train)

# Predict outputs
y_pred = ridge.predict(X_test)

# Return model coefficients and sample predictions
ridge.coef_, y_test[:10], y_pred[:10]

# Round predictions to nearest integer (0-9 classification)
y_pred_rounded = np.round(y_pred).astype(int)
y_pred_rounded = np.clip(y_pred_rounded, 0, 9)  # Ensure valid range



# Print results
print("Sample True Labels:    ", y_test[:10])
print("Sample Predicted Labels:" , y_pred[:10])

y_pred_rounded = np.round(y_pred).astype(int)
y_pred_rounded = np.clip(y_pred_rounded, 0, 9)  # Ensure valid range




