import pandas as pd

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

# Separate positive examples
positive_examples = df[df['Example Type'] == 'Positive'].drop(columns=['Example Type'])

# Initialize the hypothesis
hypothesis = ['?' for _ in range(len(positive_examples.columns))]

# Apply Find-S algorithm
for index, row in positive_examples.iterrows():
    if all(feature == '?' for feature in hypothesis):
        hypothesis = row.values
    else:
        for i, value in enumerate(row):
            if hypothesis[i] != value:
                hypothesis[i] = '?'

print("The most specific hypothesis is:")
print(hypothesis)
