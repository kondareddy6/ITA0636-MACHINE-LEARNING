import pandas as pd

# Load data from CSV
data = pd.DataFrame({
    "Citations": ["Some", "Many", "Many", "Many"],
    "Size": ["Small", "Big", "Medium", "Small"],
    "In Library": ["No", "No", "No", "No"],
    "Price": ["Affordable", "Expensive", "Expensive", "Affordable"],
    "Editions": ["Few", "Many", "Few", "Many"],
    "Buy": ["No", "Yes", "Yes", "Yes"]
})

# Initialize S and G
def initialize_hypotheses(attributes):
    S = ['0'] * len(attributes)
    G = [['?'] * len(attributes)]
    return S, G

# Check if a hypothesis is consistent with an example
def is_consistent(hypothesis, example):
    return all(h == '?' or h == e for h, e in zip(hypothesis, example))

# Candidate-Elimination algorithm
def candidate_elimination(data):
    attributes = data.columns[:-1]
    S, G = initialize_hypotheses(attributes)
    
    for i, row in data.iterrows():
        example = row[:-1].values
        if row[-1] == "Yes":
            # Update S
            for j in range(len(S)):
                if S[j] == '0':
                    S[j] = example[j]
                elif S[j] != example[j]:
                    S[j] = '?'
            
            # Remove from G any hypothesis inconsistent with example
            G = [g for g in G if is_consistent(g, example)]
        
        else:  # row[-1] == "No"
            # Remove from S any hypothesis inconsistent with example
            if is_consistent(S, example):
                S = ['0'] * len(S)
            
            # For each hypothesis in G that is consistent with the negative example
            G_new = []
            for g in G:
                if not is_consistent(g, example):
                    continue
                for j in range(len(g)):
                    if g[j] == '?':
                        for value in data[attributes[j]].unique():
                            if value != example[j]:
                                new_hypothesis = g[:]
                                new_hypothesis[j] = value
                                if is_consistent(S, new_hypothesis):
                                    G_new.append(new_hypothesis)
            G = G_new
    
    return S, G

S, G = candidate_elimination(data)
print("Final Specific Hypothesis S:", S)
print("Final General Hypothesis G:", G)
