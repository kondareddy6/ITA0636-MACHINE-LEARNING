# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Define the dataset
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': ['1980', '1970', '1990', '1980', '1980'],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# a) Read the dataset (already in DataFrame)

# b) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# c) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# d) Columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# e) Detect and replace null values with mode value
if df.isnull().sum().any():
    for column in df.columns:
        if df[column].isnull().any():
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)

print("\nNull values after replacement (if any):")
print(df.isnull().sum())

# f) Explore the dataset using heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

# Since this dataset is categorical and 'Example Type' is our target variable,
# convert categorical features to numerical using one-hot encoding
df_encoded = pd.get_dummies(df.drop(columns=['Example Type']))
y = df['Example Type'].apply(lambda x: 1 if x == 'Positive' else 0)  # Encode target variable

# g) Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# h) Fit into the model Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# i) Predict the model
y_pred = model.predict(X_test)

# j) Find the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", accuracy)
