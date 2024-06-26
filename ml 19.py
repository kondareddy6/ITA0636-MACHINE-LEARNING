import pandas as pd

# Define the dataset
data = {
    'Example': [1, 2, 3, 4],
    'Citations': ['Some', 'Many', 'Many', 'Many'],
    'Size': ['Small', 'Big', 'Medium', 'Small'],
    'In Library': ['No', 'No', 'No', 'No'],
    'Price': ['Affordable', 'Expensive', 'Expensive', 'Affordable'],
    'Editions': ['Few', 'Many', 'Few', 'Many'],
    'Buy': ['No', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Initialize the hypothesis
hypothesis = None

# Find-S algorithm
for index, row in df.iterrows():
    if row['Buy'] == 'Yes':
        if hypothesis is None:
            hypothesis = row.drop('Example')
        else:
            for attr in hypothesis.index:
                if hypothesis[attr] != row[attr]:
                    hypothesis[attr] = '?'

# Output the hypothesis
print("The most specific hypothesis is:")
print(hypothesis)
