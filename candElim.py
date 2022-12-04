#candidate elimination algorithm
import numpy as np
import pandas as pd

df = pd.read_csv('a1.csv')
concepts = np.array(df)[:,:-1]
target = np.array(df)[:,-1]

def candidateElimination(concepts, target):

    specific_hypothesis = concepts[0].copy()
    general_hypothesis = [["?" for i in range(len(specific_hypothesis))] for i in range(len(specific_hypothesis))]

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_hypothesis)):
                if h[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                    general_hypothesis[x][x] = '?'

        if target[i] == "no":
            for x in range(len(specific_hypothesis)):
                if h[x] != specific_hypothesis[x]:
                    general_hypothesis[x][x] = specific_hypothesis[x]
                else:
                    general_hypothesis[x][x] = '?'

    indices = [i for i, val in enumerate(general_hypothesis) if val == ['?', '?', '?', '?', '?', '?']]

    for i in indices:
        general_hypothesis.remove(['?', '?', '?', '?', '?', '?'])
    return specific_hypothesis, general_hypothesis

specific, general = candidateElimination(concepts, target)

print("Specific Hypothesis is:", specific,  sep="\n")
print("General Hypothesis is:", general, sep="\n")
