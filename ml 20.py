# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming 'data' is your DataFrame with columns 'sales', 'advertising', and 'quarter'

# a) Print the first five rows
print(data.head())

# b) Basic statistical computations
print(data.describe())

# c) Columns and their data types
print(data.dtypes)

# d) Explore the data using scatter plot
import matplotlib.pyplot as plt
plt.scatter(data['advertising'], data['sales'])
plt.xlabel('Advertising Expenditure')
plt.ylabel('Sales')
plt.title('Relationship between Advertising Expenditure and Sales')
plt.show()

# e) Handle null values by replacing with mode
data = data.fillna(data.mode().iloc[0])

# f) Split data into train and test sets (assuming 80-20 split)
X = data[['advertising']]  # Feature
y = data['sales']          # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# g) Train and predict using linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
