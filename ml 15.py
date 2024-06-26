# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

# a) Read the breast cancer dataset using the Pandas module
df = pd.read_csv('breast_cancer.csv')

# b) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# c) Basic statistical computations on the data set
print("\nBasic statistical computations:")
print(df.describe())

# d) Print the columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# e) Detect and handle null values in the dataset
print("\nNull values in the dataset:")
print(df.isnull().sum())

# Replace null values with mode value
for column in df.columns:
    if df[column].isnull().any():
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

print("\nNull values after replacement (if any):")
print(df.isnull().sum())

# f) Split the data into test and train
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = GaussianNB()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# g) Evaluate the performance of the model using confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
