# Relaxation Labeling

A small, clean Python implementation of **relaxation labeling** — the identify-by-constraints
algorithm of Hummel & Zucker, *"On the foundations of relaxation labeling processes"* (1983).

If you can describe a problem as:

1. a set of **objects** that need identifying,
2. a set of **labels** you might assign to them, and
3. a **compatibility** — a number saying how well any label on one object agrees with any
   label on another object,

…then relaxation labeling will iterate the whole assignment toward a globally consistent
labeling. It's a beautifully general idea: the same engine segments images, parses scenes,
matches shapes — and, in the companion project below, reads the roles of chess pieces,
grows terrain from a battle, and infers a piece of music's time signature.

> 🎥 Demos & tutorials: **[youtube.com/@mannyglover](https://www.youtube.com/@mannyglover)**

## Quick start

```bash
pip install numpy
python example_clusters.py
```

For development, install the test extra and run the suite:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

`example_clusters.py` is a 30-line worked example: four objects in two hidden clusters, two
labels, weak per-object priors. The algorithm firms the guess up into the consistent
labeling and recovers the clusters:

```
Object -> label: [0, 0, 1, 1]
OK: relaxation labeling recovered the clusters {0,1}=A and {2,3}=B.
```

## The API

```python
from relax import RelaxationLabeling
rl = RelaxationLabeling(compatibility, save=False, iterations=50)
labels = rl.objectToLabelMapping   # object -> chosen label
strength = rl.strength             # final per-(object,label) confidence
```

`compatibility` is a NumPy array:
- **order-2** (pairwise): shape `(numObjects, numLabels, numObjects, numLabels)`, where
  `C[i, j, k, l]` is the support object *i* taking label *j* gets from object *k* taking
  label *l*.
- **order-3** (triples): shape `(nObj, nLab, nObj, nLab, nObj, nLab)`.

The iteration runs in the constructor. Degenerate rows with no compatibility gradient remain
valid probability distributions rather than producing `NaN`; a symmetric row remains
uniform until compatibility or a prior breaks the tie.

### One practical lesson worth knowing
Relaxation labeling needs a **seed**. From a perfectly uniform, symmetric start there is no
gradient to descend, and degenerate normalization can collapse everything to one label (or
divide by zero). Encode what you already know as a small per-object **prior** in the
compatibility's self-term — as `example_clusters.py` does — and keep that prior present as a
gentle bias. The coalition terms then *refine* the answer instead of erasing it.

## What's in here

| File | What it is |
|---|---|
| `relax.py` | The algorithm — the `RelaxationLabeling` class (support/strength iteration). |
| `example_clusters.py` | The runnable worked example above. |
| `triangles_gt_squares.py` | A small trinary-logic companion (`Trool`: false / **ish** / true) — the "uncertain middle" that the boundary work builds on. |

## Applied showcase

See **[whimsy-chess](https://github.com/mannyglover/whimsy-chess)** — the same algorithm,
ported to the browser, labeling chess-piece *roles*, segmenting the board into *terrain*,
and inferring *rhythm* from a game. It's the living demonstration of what this core enables.

## License

- **Code:** [AGPL-3.0](LICENSE) — free to use, study, share, and modify; distributing it or
  running it as a modified service means sharing your changes back. A **commercial license**
  is also available — see [`NOTICE`](NOTICE).

Built by **Manny Glover** (R. Michael Glover). Contributions welcome under a short CLA that
keeps the dual-license possible. Collaborations and teaching uses especially welcome.
