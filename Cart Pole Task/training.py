import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load the feature CSV ---
data = pd.read_csv('/home/kristi/Desktop/Research_progress/Winter_2025/JoSIM-master/build/cartpole_1.csv')

# --- Step 2: Split features and labels ---
# Assume the label column is named 'label' (update if different)
X = data.drop(columns=['label']).values  # Features
y = data['label'].values                 # Labels

# --- Step 3: Split into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train logistic regression ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Step 5: Evaluate model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Step 6: Confusion matrix plot ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
