import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# a) Read the Mobile price dataset using the Pandas module
df = pd.read_csv('mobile_price_data.csv')  # replace 'mobile_price_data.csv' with your file path

# b) Print the 1st five rows
print("First five rows of the dataset:")
print(df.head())

# c) Basic statistical computations on the data set or distribution of data
print("\nBasic statistical computations:")
print(df.describe())

# d) The columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# e) Detects null values in the dataset. If there is any null values replaced it with mode value
print("\nChecking for null values:")
print(df.isnull().sum())

if df.isnull().sum().sum() > 0:
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
    print("\nNull values replaced with mode value.")

print("\nNull values after replacement (if any):")
print(df.isnull().sum())

# f) Explore the data set using heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

# g) Split the data into test and train
X = df.drop('price_range', axis=1)  # Features
y = df['price_range']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# h) Fit into the model Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# i) Predict the model
y_pred = model.predict(X_test)

# j) Find the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:")
print(accuracy)
