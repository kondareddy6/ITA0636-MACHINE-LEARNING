-import pandas as pd
# Creating the dataset as a DataFrame
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Enjoy Sport': ['Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Find-S Algorithm Implementation
def find_s_algorithm(df):
    # Initialize the hypothesis with the first positive example
    hypothesis = ['0'] * (len(df.columns) - 1)
    for i in range(len(df)):
        if df['Enjoy Sport'][i] == 'Yes':
            if hypothesis == ['0'] * (len(df.columns) - 1):
                hypothesis = df.iloc[i, :-1].tolist()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != df.iloc[i, j]:
                        hypothesis[j] = '?'
    return hypothesis

# Applying the Find-S algorithm
hypothesis = find_s_algorithm(df)
print("The most specific hypothesis is:", hypothesis)
