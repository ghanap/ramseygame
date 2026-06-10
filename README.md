# Ramsey Graph Search

Stochastic parallel search for Ramsey graph colorings with MLflow experiment tracking and live Plotly visualization.

## What is Ramsey Theory?
Ramsey Theory is a branch of combinatorics that studies the conditions under which order must appear within chaos. The Ramsey number R(s, t) is the minimum number of vertices n such that every edge-coloring (with red and blue) of the complete graph on n vertices contains either a red clique of size s or a blue clique of size t. Finding these colorings is computationally explosive, making heuristic searches necessary.

## What this Search Does
This project runs a **greedy repair search** to find valid edge colorings for a given n, s, and t. It attempts to minimize the number of "forbidden subgraphs" (monochromatic cliques). A perfect score of 0 means a valid Ramsey coloring was found.

## Architecture
- **Parallel Workers**: Multiple worker threads run independent stochastic repair searches in parallel to explore different regions of the state space.
- **Elite Archive**: A shared, thread-safe archive stores the best graphs discovered across all threads, avoiding redundant work.
- **Coordinator**: Logs the scores, parameters, and JSON graph structures directly into **MLflow** for experiment tracking.
- **Live Visualization**: Uses Plotly to render a real-time, circular graph layout that updates dynamically as the workers find better colorings.

## The `autoresearch` Folder
The `autoresearch` folder contains additional scripts for deep analysis of the generated graphs:
- `analysis.ipynb`: Analyzes the distribution and symmetry of the elite archive graphs.
- `train.py` & `prepare.py`: Machine learning pipelines to learn from the graph structures.

## Mathematical Connections
*Note: This computational search for symmetry and invariants directly complements my ongoing Study Oriented Project on Burnside's Lemma and Group Actions with Prof. Sushil Bhunia, applying computational group theory to graph symmetry.*

## How to Run

```bash
# Install dependencies
pip install mlflow plotly networkx numpy

# Run the coordinator (default: R(3,3) on n=6 with 4 workers)
python ramseygame.py --s 3 --t 3 --n 6 --workers 4
```
