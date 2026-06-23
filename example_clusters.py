"""
A tiny, self-contained demonstration of relaxation labeling.

Four objects fall into two natural clusters: {0,1} are alike and {2,3} are alike.
We give the algorithm two labels (A and B), a small per-object *prior* (the nudge that
breaks the initial symmetry -- relaxation labeling needs a seed, otherwise a perfectly
uniform start has no gradient to descend), and a *coupling* that says: similar objects
should agree. Starting from those weak hints, the algorithm firms up into the consistent
labeling on its own -- discovering the two clusters.

Run:  python example_clusters.py
"""
import numpy as np
from relax import RelaxationLabeling

nObj, nLab = 4, 2

# pairwise similarity (off-diagonal coupling): {0,1} alike, {2,3} alike
sim = np.array([
    [0.0, 0.9, 0.0, 0.0],
    [0.9, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.9],
    [0.0, 0.0, 0.9, 0.0],
])
# weak per-object prior over [A, B] -- just enough to break the tie
prior = np.array([
    [0.6, 0.1],   # object 0 leans A
    [0.3, 0.2],   # object 1 faint lean A
    [0.2, 0.3],   # object 2 faint lean B
    [0.1, 0.6],   # object 3 leans B
])

C = np.zeros((nObj, nLab, nObj, nLab))
for i in range(nObj):
    for j in range(nLab):
        C[i, j, i, j] = prior[i, j]                      # self-term: the prior seed
        for k in range(nObj):
            if k == i:
                continue
            for l in range(nLab):
                C[i, j, k, l] = sim[i, k] * (1.0 if j == l else 0.0)  # agree-if-similar

rl = RelaxationLabeling(C, save=False, iterations=20)
labels = rl.objectToLabelMapping.ravel().astype(int)
print("\nObject -> label:", labels.tolist())
print("Final strengths:\n", np.round(rl.strength, 3))
assert labels.tolist() == [0, 0, 1, 1], "expected the two clusters {0,1} and {2,3}"
print("\nOK: relaxation labeling recovered the clusters {0,1}=A and {2,3}=B.")
