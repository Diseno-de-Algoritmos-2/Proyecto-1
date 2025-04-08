# ---------------------------------------------------------------------------
import numpy as np
from scipy.spatial import ConvexHull

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problemInstance import vehicle_capacity as VEHICLE_CAPACITY, customers as CUSTOMERS

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def euclidean_distance(a, b):
    """Distancia Euclídea entre dos puntos 2-D."""
    return np.linalg.norm(np.array(a) - np.array(b))

def compute_distance_matrix(points):
    """Matriz completa de distancias Euclídeas."""
    n = len(points)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dmat[i][j] = euclidean_distance(points[i], points[j])
    return dmat


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def convex_average_hull_area(points, cluster):
    """Área del casco convexo dividida por el número de nodos del clúster.
    *Nota*: en 2-D, `ConvexHull.area` devuelve el *perímetro*; usa `volume`
    si deseas el área geométrica.
    """
    if len(points) < 3:
        return 0.0
    pts = np.array(points)
    hull = ConvexHull(pts)
    return hull.area / len(cluster)

# ---------------------------------------------------------------------------
# Fase 1 – agrupamiento
# ---------------------------------------------------------------------------

def _cluster_iteration(seed_node, customers, capacity, dmat):
    """Crea clústeres iniciando con `seed_node` y respetando la capacidad."""
    unclustered = set(range(1, len(customers)))  # ignoramos depósito
    clusters = []

    while unclustered:
        current = seed_node if seed_node in unclustered else next(iter(unclustered))
        cluster = [current]
        load = customers[current]["demand"]
        unclustered.remove(current)

        # Ordena nodos restantes por distancia al nodo raíz actual
        dist_list = sorted((dmat[current][j], j) for j in unclustered)
        for _, j in dist_list:
            if load + customers[j]["demand"] <= capacity:
                cluster.append(j)
                load += customers[j]["demand"]
                unclustered.remove(j)

        clusters.append(cluster)

    return clusters


def _clustering_metric(clusters, customers):
    """Σ(area casco convexo / |cluster|)"""
    total = 0.0
    for cluster in clusters:
        pts = [customers[i]["coord"] for i in cluster]
        total += convex_average_hull_area(pts, cluster)
    return total


def phase1_best_clustering(customers, capacity):
    """Devuelve la mejor partición encontrada 
        (N iteraciones, una por cliente)."""
    coords = [c["coord"] for c in customers]
    dmat = compute_distance_matrix(coords)

    best_clusters = []
    best_metric = float("inf")
    unclustered = set(range(1, len(customers)))  # ignoramos depósito

    # Realizamos tantas iteraciones como clientes no agrupados (nodos del grafo sin el depósito).
    for seed_node in unclustered:
        clusters = _cluster_iteration(seed_node, customers, capacity, dmat)
        metric = _clustering_metric(clusters, customers)
        if metric < best_metric:
            best_metric = metric
            best_clusters = clusters
            print(f"  --> Mejor agrupamiento: {seed_node} (métrica = {round(best_metric, 2)})")

    return best_clusters


# ---------------------------------------------------------------------------
# Correr
# ---------------------------------------------------------------------------
def run_phase1():

    print("\n --- Fase 1 - Agrupamiento --- \n")
    clusters = phase1_best_clustering(CUSTOMERS, VEHICLE_CAPACITY)

    print(f"\nSe formaron {len(clusters)} clústeres:\n")

    for idx, cl in enumerate(clusters, 1):
        load = sum(CUSTOMERS[i]["demand"] for i in cl)
        print(f"  Clúster {idx:>2}: {cl}, Capacidad = {load}/{VEHICLE_CAPACITY}")

    print(f"\nMétrica de agrupamiento: {round(_clustering_metric(clusters, CUSTOMERS), 2)}\n")

    return clusters