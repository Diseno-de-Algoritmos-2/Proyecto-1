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

"""
In the beginning of the first phase, a cost/distance matrix is
calculated which contains the distance among all the customer nodes and the depot. 
"""


def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def compute_distance_matrix(points):
    n = len(points)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dmat[i][j] = euclidean_distance(points[i], points[j])
    return dmat


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
The convex hull of a set X of points in
the Euclidean plane is defined as the smallest convex set that contains X. Initially, a convex hull is formed by
considering the locations of customer nodes of a particular cluster and then area of that convex hull is
calculated. Subsequently, the area is divided by number of customer nodes in that cluster. In a similar way, that
value is evaluated for all the clusters of one particular iterative set of clusters and calculate the summation. This
summation is used as the parameter to select the best set of clusters. 
"""


def convex_average_hull_area(points, cluster):
    if len(points) < 3:
        return 0.0
    pts = np.array(points)
    hull = ConvexHull(pts)
    return hull.area / len(cluster)


def _clustering_metric(clusters, customers):
    """Σ(area casco convexo / |cluster|)"""
    total = 0.0
    for cluster in clusters:
        pts = [customers[i]["coord"] for i in cluster]
        total += convex_average_hull_area(pts, cluster)
    return total


# ---------------------------------------------------------------------------
# Fase 1 – agrupamiento
# ---------------------------------------------------------------------------


def _cluster_iteration(seed_node, customers, capacity, dmat):

    unclustered = set(range(1, len(customers)))  # ignoramos depósito
    clusters = []

    """
    At the beginning of n
    th iteration, distances from nth customer node to all the other customer nodes are
    obtained from the distance matrix and a distance list is formed by arranging those distances in ascending order. 

    This procedure is repeated
    until all the customer nodes are clustered and subsequently the next ((n+1)th) iteration commences. 
    """
    while unclustered:

        """
        The first cluster of the nth iteration is started with nth customer node by setting the total demand of the current
        cluster (TDCC) to the demand of the nth customer node and nth customer node is marked as a clustered node.
        """
        v = seed_node if seed_node in unclustered else min(unclustered)
        cluster = [v]
        load = customers[v]["demand"]
        unclustered.remove(v)

        """
        In each iteration, all customers to be
        served are clustered according to a repeatedly updating distance list of non-clustered customer nodes without
        exceeding the vehicle capacity.

        After that, if inserting first node (i) from the top of the distance list does not exceed the TDCC, the node i is
        added to the current cluster, removed from the distance list and marked as a clustered node. Then, the demand of
        node i is added (TDCC = TDCC + di) to TDCC. Accordingly, customer nodes are inserted to the first cluster
        from the top of the list until vehicle capacity constraint reached.
        """

        while True:

            # rebuild list each time, as required
            dist_list = sorted((dmat[v, u], u) for u in unclustered)
            if not dist_list:
                break
            _, i = dist_list[0]

            if load + customers[i]["demand"] <= capacity:  # fits -> put inside
                cluster.append(i)
                load += customers[i]["demand"]
                unclustered.remove(i)
                v = i
            else:  # doesn’t fit -> start new cluster
                break

        clusters.append(cluster)

        """
        Afterwards, next customer node (i) from the distance list is selected and the second cluster starts with
        that selected customer node.
        """

    return clusters


def phase1_best_clustering(customers, capacity):

    coords = [c["coord"] for c in customers]
    dmat = compute_distance_matrix(coords)

    best_clusters = []
    best_metric = float("inf")
    unclustered = set(range(1, len(customers)))  # ignoramos depósito

    """
    The first phase is an
    iterative procedure which creates N (number of customers) number of sets of clusters.
    """
    # Realizamos tantas iteraciones como clientes no agrupados (nodos del grafo sin el depósito).
    for seed_node in unclustered:

        clusters = _cluster_iteration(seed_node, customers, capacity, dmat)
        metric = _clustering_metric(clusters, customers)

        """
        At the end of the first phase, the best set of clusters (best clustering) is selected based on the measure
        defines as (Area of the convex hull) / (Number of nodes in the cluster).
        """
        if metric < best_metric:
            best_metric = metric
            best_clusters = clusters
            print(
                f"  --> Mejor agrupamiento: {seed_node} (métrica = {round(best_metric, 2)})"
            )

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

    print(
        f"\nMétrica de agrupamiento: {round(_clustering_metric(clusters, CUSTOMERS), 2)}\n"
    )

    return clusters
