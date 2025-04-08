import numpy as np
from scipy.spatial import ConvexHull

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problemInstance import vehicle_capacity as VEHICLE_CAPACITY, customers as CUSTOMERS

# ---------------------------------------------------------------------------
# ------------------------  Helpers genéricos  ------------------------------
# ---------------------------------------------------------------------------

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def compute_distance_matrix(points):
    n = len(points)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dmat[i, j] = euclidean_distance(points[i], points[j])
    return dmat


# ------------------------  Parámetros (métricas)  --------------------------

def convex_average_hull_area(points, cluster):
    """Parámetro 1."""
    if len(points) < 3:
        return 0.0
    hull = ConvexHull(points)
    return hull.area / len(cluster)


def convex_hull_area(points):
    """Parámetro 2."""
    if len(points) < 3:
        return 0.0
    hull = ConvexHull(points)
    return hull.area


def convex_average_demand_hull_area(points, cluster, customers):
    """Parámetro 3."""
    if len(points) < 3:
        return 0.0
    hull = ConvexHull(points)
    total_demand = sum(customers[i]["demand"] for i in cluster)
    return hull.area / total_demand


def mean_distance_from_centroid(points):
    """Parámetro 4."""
    if len(points) < 2:
        return 0.0
    centroid = np.mean(points, axis=0)
    return np.mean([euclidean_distance(pt, centroid) for pt in points])


def mean_distance_from_centroid_avg_demand(points, cluster, customers):
    """Parámetro 5."""
    if len(points) < 2:
        return 0.0
    centroid = np.mean(points, axis=0)
    mean_dist = np.mean([euclidean_distance(pt, centroid) for pt in points])
    total_demand = sum(customers[i]["demand"] for i in cluster)
    return mean_dist / total_demand


# ------------------------  Agrupamiento  -----------------------------------

def _cluster_iteration(seed_node, customers, capacity, dmat):

    unclustered = set(range(1, len(customers)))     # ignoramos depósito (0)
    clusters = []

    while unclustered:
        current = seed_node if seed_node in unclustered else next(iter(unclustered))
        cluster = [current]
        load = customers[current]["demand"]
        unclustered.remove(current)

        # nodos restantes ordenados por distancia al nodo raíz
        dist_list = sorted((dmat[current, j], j) for j in unclustered)
        for _, j in dist_list:
            dem = customers[j]["demand"]
            if load + dem <= capacity:
                cluster.append(j)
                load += dem
                unclustered.remove(j)

        clusters.append(cluster)

    return clusters


# ------------------------  Evaluación de métricas  -------------------------

def evaluate_metrics(clusters, customers):
    coords = [c["coord"] for c in customers]

    p1 = p2 = p3 = p4 = p5 = 0.0
    for cl in clusters:
        pts = [coords[i] for i in cl]

        p1 += convex_average_hull_area(pts, cl)
        p2 += convex_hull_area(pts)
        p3 += convex_average_demand_hull_area(pts, cl, customers)
        p4 += mean_distance_from_centroid(pts)
        p5 += mean_distance_from_centroid_avg_demand(pts, cl, customers)

    return p1, p2, p3, p4, p5


# ------------------------  Fase 1 principal  -------------------------------

def phase1_top_sets(customers, capacity):
    """
    Devuelve una lista con ≤ 5 conjuntos de clústeres.
    Cada conjunto es el arg min de uno de los cinco parámetros.
    """
    coords = [c["coord"] for c in customers]
    dmat = compute_distance_matrix(coords)

    all_clusterings = []

    # 1. generar N clusterings
    for seed in range(1, len(customers)):
        clusters = _cluster_iteration(seed, customers, capacity, dmat)
        metrics = evaluate_metrics(clusters, customers)
        all_clusterings.append((clusters, metrics))

    # 2. encontrar mínimo por parámetro
    winners = {}
    for k in range(5):
        idx_min = min(range(len(all_clusterings)),
                      key=lambda i: all_clusterings[i][1][k])
        winners[idx_min] = all_clusterings[idx_min][0]      # dict evita duplicados

    return list(winners.values())


# ---------------------------------------------------------------------------
# Correr
# ---------------------------------------------------------------------------
def run_phase1():

    print("\n --- Fase 1 - Agrupamiento --- \n")
    top_sets = phase1_top_sets(CUSTOMERS, VEHICLE_CAPACITY)

    print(f"Se obtuvieron {len(top_sets)} conjuntos ganadores (máx. 5):\n")
    
    for s, clusters in enumerate(top_sets, 1):
        p1, p2, p3, p4, p5 = evaluate_metrics(clusters, CUSTOMERS)

        print(f"  Conjunto {s}: {len(clusters)} clústeres  |  P1={p1:.2f}  P2={p2:.2f}  P3={p3:.2f}  P4={p4:.2f}  P5={p5:.2f}")
        for i, c in enumerate(clusters, 1):
            print(f"    Ruta{i}: {c}")
    
        print()

    return top_sets