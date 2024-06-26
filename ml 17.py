import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier()
}

# DataFrame to store results
results = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Training Time (s)', 'Prediction Time (s)'])

# Train and evaluate each model
for algo_name, model in models.items():
start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train

start_pred = time.time()
y_pred = model.predict(X_test)
end_pred = time.time()
pred_time = end_pred - start_pred

    accuracy = accuracy_score(y_test, y_pred)

    results = results.append({
        'Algorithm': algo_name,
        'Accuracy': accuracy,
        'Training Time (s)': train_time,
        'Prediction Time (s)': pred_time
    }, ignore_index=True)

# Display results
print("Performance Comparison of Classification Algorithms on Iris Dataset:\n")
print(results)
