# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# a) Read the house dataset using the Pandas module
df = pd.read_csv('houses.csv')

# b) Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# c) Basic statistical computations on the data set or distribution of data
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

print("\nNull values after replacement:")
print(df.isnull().sum())

# f) Explore the dataset using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

# g) Split the data into training and testing sets
# Assume 'price' is the target variable
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# h) Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# i) Predict the price of houses in the test set
y_pred = model.predict(X_test)

# j) Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("R-squared:", r2)

# Predict the price of a new house (example house with values for each feature)
# Example: Assuming new_house is a DataFrame with the same columns as X
new_house = pd.DataFrame({
    'feature1': [value1],  # Replace value1, value2, ... with actual feature values
    'feature2': [value2],
    # Add other features as per your dataset
})

predicted_price = model.predict(new_house)
print("\nPredicted price for the new house:", predicted_price[0])
