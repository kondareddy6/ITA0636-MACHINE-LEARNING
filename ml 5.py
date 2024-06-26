import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to a DataFrame for better visualization
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y

print("First five rows of the Iris dataset:")
print(df.head())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Implementing the Expectation Maximization algorithm using GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

# Predict the cluster for each data point
y_gmm = gmm.predict(X_scaled)

# Map the GMM cluster labels to the original labels for better evaluation
# This mapping step is specific to this dataset and required due to arbitrary cluster labeling by GMM
def map_cluster_to_class(y_gmm, y):
    mapping = {}
    for i in range(3):
        mask = (y_gmm == i)
        target_class = np.argmax(np.bincount(y[mask]))
        mapping[i] = target_class
    return np.vectorize(mapping.get)(y_gmm)

y_gmm_mapped = map_cluster_to_class(y_gmm, y)

# Evaluate the model
accuracy = accuracy_score(y, y_gmm_mapped)
print("\nAccuracy of the EM algorithm using GMM:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y, y_gmm_mapped))

print("\nConfusion Matrix:")
cm = confusion_matrix(y, y_gmm_mapped)
print(cm)

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
