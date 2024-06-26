import pandas as pd
# Define the dataset
data = {
    'Size': ['Big', 'Small', 'Small', 'Big', 'Small'],
    'Color': ['Red', 'Red', 'Red', 'Blue', 'Blue'],
    'Shape': ['Circle', 'Triangle', 'Circle', 'Circle', 'Circle'],
    'Class': ['No', 'No', 'Yes', 'No', 'Yes']
}
# Create a DataFrame
df = pd.DataFrame(data)

# Find-S Algorithm
def find_s_algorithm(df):
    # Initialize the most specific hypothesis
    hypothesis = None

    # Iterate through the dataset
    for i in range(len(df)):
        if df['Class'][i] == 'Yes':
            if hypothesis is None:
                hypothesis = df.iloc[i][:-1].tolist()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != df.iloc[i][j]:
                        hypothesis[j] = '?'

    return hypothesis

# Execute the Find-S algorithm
hypothesis = find_s_algorithm(df)

# Output the final hypothesis
print("The most specific hypothesis is:", hypothesis)
