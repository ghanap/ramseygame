"""
Ramsey search with Plotly live visualisation (no buffering, interactive).
Workers run greedy repair search in parallel threads.
Coordinator logs results to MLflow and updates the Plotly figure.
"""

import mlflow
import time
import hashlib
import numpy as np
import networkx as nx
import threading
import queue
import random
from itertools import combinations
import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------
# Elite Archive
# ------------------------------------------------------------
class EliteArchive:
    def __init__(self, filename="archive.json"):
        self.filename = filename
        self.graphs = []
        self.lock = threading.Lock()
        self.load()

    def load(self):
        import json
        try:
            with open(self.filename, 'r') as f:
                self.graphs = json.load(f)
        except FileNotFoundError:
            self.graphs = []

    def save(self):
        import json
        with self.lock:
            with open(self.filename, 'w') as f:
                json.dump(self.graphs, f, indent=2)

    def add(self, graph_matrix, score, worker_id):
        graph_list = graph_matrix.tolist() if hasattr(graph_matrix, 'tolist') else graph_matrix
        h = hashlib.md5(str(graph_list).encode()).hexdigest()
        with self.lock:
            existing = next((g for g in self.graphs if hashlib.md5(str(g['graph']).encode()).hexdigest() == h), None)
            if existing:
                if score > existing['score']:
                    existing['score'] = score
                    existing['timestamp'] = time.time()
                    existing['worker_id'] = worker_id
            else:
                self.graphs.append({
                    "graph": graph_list,
                    "score": score,
                    "timestamp": time.time(),
                    "worker_id": worker_id
                })
            self.save()

    def get_best(self, n=1):
        sorted_graphs = sorted(self.graphs, key=lambda x: x['score'], reverse=True)
        return sorted_graphs[:n]

# ------------------------------------------------------------
# Greedy repair search worker
# ------------------------------------------------------------
def count_forbidden(board, s, t):
    n = len(board)
    cnt = 0
    for nodes in combinations(range(n), s):
        if all(board[u][v] == 0 for u, v in combinations(nodes, 2)):
            cnt += 1
    for nodes in combinations(range(n), t):
        if all(board[u][v] == 1 for u, v in combinations(nodes, 2)):
            cnt += 1
    return cnt

def repair_search(n, s, t, max_iter=500):
    # random initial full coloring
    board = [[random.choice([0, 1]) for _ in range(n)] for __ in range(n)]
    for i in range(n):
        board[i][i] = -1
    # symmetrize
    for i in range(n):
        for j in range(i+1, n):
            board[j][i] = board[i][j]

    best_board = [row[:] for row in board]
    best_score = count_forbidden(board, s, t)

    for _ in range(max_iter):
        i, j = random.sample(range(n), 2)
        if i == j:
            continue
        board[i][j] = 1 - board[i][j]
        board[j][i] = board[i][j]
        new_score = count_forbidden(board, s, t)
        if new_score <= best_score:
            best_score = new_score
            best_board = [row[:] for row in board]
        else:
            board[i][j] = 1 - board[i][j]
            board[j][i] = board[i][j]

    return best_board, -best_score   # score = - (forbidden count) → higher better, 0 perfect

# ------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------
class WorkerThread(threading.Thread):
    def __init__(self, worker_id, job_queue, result_queue, s, t, n):
        super().__init__()
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.s = s
        self.t = t
        self.n = n
        self.daemon = True

    def run(self):
        print(f"Worker {self.worker_id} started.")
        while True:
            try:
                job = self.job_queue.get(timeout=2)
            except queue.Empty:
                continue
            params = job['params']
            best_graph, score = repair_search(params['n'], params['s'], params['t'], params.get('max_iter', 500))
            result = {
                "job_id": job['job_id'],
                "worker_id": self.worker_id,
                "graph": best_graph,
                "score": score
            }
            self.result_queue.put(result)
            print(f"Worker {self.worker_id} finished job {job['job_id']} with score {score}")

# ------------------------------------------------------------
# Coordinator with Plotly live updates
# ------------------------------------------------------------
class Coordinator:
    def __init__(self, s=3, t=3, n=6, num_workers=4):
        self.s = s
        self.t = t
        self.n = n
        self.num_workers = num_workers
        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.archive = EliteArchive()
        self.job_counter = 0
        self.best_score = -float('inf')
        self.best_graph = None

        # Start worker threads
        self.workers = []
        for i in range(num_workers):
            w = WorkerThread(f"W{i}", self.job_queue, self.result_queue, s, t, n)
            w.start()
            self.workers.append(w)

        # Precompute layout for the complete graph (circular)
        self.G = nx.complete_graph(n)
        self.pos = nx.circular_layout(self.G)

        # Create empty Plotly figure (will be updated later)
        self.fig = go.Figure()
        self.fig.update_layout(
            title=f"Ramsey R({s},{t}) search – n={n}",
            width=700,
            height=700,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True
        )
        # Open the browser window (non-blocking)
        pio.renderers.default = 'browser'
        self.fig.show(renderer="browser")

    def update_visualization(self):
        """Fetch best graph and redraw Plotly figure."""
        best = self.archive.get_best(1)
        if not best:
            return
        graph_data = best[0]['graph']
        n = len(graph_data)

        # Build edge line coordinates for red and blue
        red_x, red_y = [], []
        blue_x, blue_y = [], []
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = self.pos[i]
                xj, yj = self.pos[j]
                if graph_data[i][j] == 0:       # red edge
                    red_x.extend([xi, xj, None])
                    red_y.extend([yi, yj, None])
                elif graph_data[i][j] == 1:     # blue edge
                    blue_x.extend([xi, xj, None])
                    blue_y.extend([yi, yj, None])

        # Node positions and labels
        node_x = [self.pos[i][0] for i in range(n)]
        node_y = [self.pos[i][1] for i in range(n)]
        node_labels = [str(i) for i in range(n)]

        # Clear old traces and create new ones
        self.fig.data = []
        # Edges
        self.fig.add_trace(go.Scatter(x=red_x, y=red_y, mode='lines', line=dict(color='red', width=2), name='Red edge'))
        self.fig.add_trace(go.Scatter(x=blue_x, y=blue_y, mode='lines', line=dict(color='blue', width=2), name='Blue edge'))
        # Nodes
        self.fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                      marker=dict(size=25, color='lightgray', line=dict(color='black', width=1)),
                                      text=node_labels, textposition='middle center', name='Vertex'))
        # Update title with score
        self.fig.update_layout(title=f"R({self.s},{self.t}) n={self.n}  Best score = {best[0]['score']}")
        # Redraw
        self.fig.show(renderer="browser")

    def start(self):
        print(f"Coordinator started for R({self.s},{self.t}) on n={self.n}")
        mlflow.set_experiment(f"RamseyPlotly_R{self.s}_{self.t}_n{self.n}")

        last_update = time.time()
        while True:
            # Push a new job every 10 seconds
            job = {
                "job_id": self.job_counter,
                "params": {"n": self.n, "s": self.s, "t": self.t, "max_iter": 500},
                "timestamp": time.time()
            }
            self.job_queue.put(job)
            self.job_counter += 1

            # Collect results (non-blocking)
            try:
                res = self.result_queue.get(timeout=5)
                score = res['score']
                graph = np.array(res['graph'])
                self.archive.add(graph, score, res['worker_id'])
                with mlflow.start_run(run_name=f"worker_{res['worker_id']}_job_{res['job_id']}"):
                    mlflow.log_param("job_id", res['job_id'])
                    mlflow.log_metric("score", score)
                    mlflow.log_text(str(graph.tolist()), "graph.json")
                print(f"Job {res['job_id']} from worker {res['worker_id']}: score = {score}")
                if score > self.best_score:
                    self.best_score = score
                    self.best_graph = graph
                    self.update_visualization()
            except queue.Empty:
                print("No results yet...")

            # Periodically refresh plot anyway (every 2 seconds)
            if time.time() - last_update > 2:
                self.update_visualization()
                last_update = time.time()

            time.sleep(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=3)
    parser.add_argument("--t", type=int, default=3)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    c = Coordinator(s=args.s, t=args.t, n=args.n, num_workers=args.workers)
    c.start()