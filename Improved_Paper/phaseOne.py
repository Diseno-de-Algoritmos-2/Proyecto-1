import numpy as np
from scipy.spatial.qhull import QhullError
from scipy import spatial
spatial.QhullError = QhullError
from scipy.spatial import ConvexHull
import gc

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

"""
At the beginning
of the considered two-phased heuristic, a distance matrix is
formed which contained the distances among all the vertices
including the depot.
"""

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def compute_distance_matrix(points):
    points = np.array(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)

# ------------------------  Parámetros (métricas)  --------------------------

"""
Then five parameters, including the parameter
defined above by (8), are evaluated for each and every set
of feasible clusters. The new four parameters are stated
from (9) to (12) as follows
"""

def convex_average_hull_area(points, cluster):
    if len(points) < 3:
        return 100000000.0
    try:
        hull = ConvexHull(points)
        return hull.area / len(cluster)
    except QhullError:
        return 100000000.0

def convex_hull_area(points):
    if len(points) < 3:
        return 100000000.0
    try:
        hull = ConvexHull(points)
        return hull.area
    except QhullError:
        return 100000000.0

def convex_average_demand_hull_area(points, cluster, customers):
    if len(points) < 3:
        return 100000000.0
    try:
        hull = ConvexHull(points)
        total_demand = np.sum([customers[i]["demand"] for i in cluster])
        return hull.area / total_demand
    except QhullError:
        return 100000000.0

def mean_distance_from_centroid(points):
    if len(points) < 2:
        return 100000000.0
    centroid = np.mean(points, axis=0)
    return np.mean(np.linalg.norm(points - centroid, axis=1))

def mean_distance_from_centroid_avg_demand(points, cluster, customers):
    if len(points) < 2:
        return 100000000.0
    centroid = np.mean(points, axis=0)
    mean_dist = np.mean(np.linalg.norm(points - centroid, axis=1))
    total_demand = np.sum([customers[i]["demand"] for i in cluster])
    return mean_dist / total_demand

# ---------------------------------------------------------------------------
# Agrupamiento
# ---------------------------------------------------------------------------

def _cluster_iteration(seed_node, customers, capacity, dmat):

    n = len(customers)
    unclustered = set(range(1, n))  # omit depot
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

        while unclustered:

            # Rebuild list each time.
            distances = sorted([(dmat[v, u], u) for u in unclustered])
            if not distances:
                break
            _, i = distances[0]  # Deberiamos usar min y no ordenar, pero seguimos el paper.
            if load + customers[i]["demand"] <= capacity:
                cluster.append(i)
                load += customers[i]["demand"]
                unclustered.remove(i)
                v = i
            else:
                break

        """
        Afterwards, next customer node (i) from the distance list is selected and the second cluster starts with
        that selected customer node.
        """

        clusters.append(cluster)

    return clusters

# ---------------------------------------------------------------------------
# Evaluación de métricas
# ---------------------------------------------------------------------------

"""
At the end of the first phase, the five best sets of clusters
are selected with regard to the above five parameters such
that each set of clusters corresponds to a minimum value
of the five parameters. 
"""

"""
TODA METRICA QUE NO SEA 0.0 O NO SE CALCULE POR IMPEDIMENTO
GEOMETRICO SE PENALIZA.
"""

def evaluate_metrics(clusters, customers):
    coords = np.array([c["coord"] for c in customers])

    p1 = p2 = p3 = p4 = p5 = 0.0
    for cl in clusters:
        pts = coords[cl]

        p1 += convex_average_hull_area(pts, cl)
        p2 += convex_hull_area(pts)
        p3 += convex_average_demand_hull_area(pts, cl, customers)
        p4 += mean_distance_from_centroid(pts)
        p5 += mean_distance_from_centroid_avg_demand(pts, cl, customers)

        if p1 == 0.0: p1 = 100000000.0
        if p2 == 0.0: p2 = 100000000.0
        if p3 == 0.0: p3 = 100000000.0
        if p4 == 0.0: p4 = 100000000.0
        if p5 == 0.0: p5 = 100000000.0

    return p1, p2, p3, p4, p5

# ---------------------------------------------------------------------------
# Fase 1 principal
# ---------------------------------------------------------------------------

"""
The same
method is used to construct n number of sets of feasible
clusters at the beginning of the first phase in the improved
algorithm
"""

def phase1_top_sets(customers, capacity):
    coords = np.array([c["coord"] for c in customers])
    dmat = compute_distance_matrix(coords)

    all_clusterings = []

    """
    The first phase is an
    iterative procedure which creates N (number of customers) number of sets of clusters.
    """

    for seed in range(1, len(customers)):
        clusters = _cluster_iteration(seed, customers, capacity, dmat)
        metrics = evaluate_metrics(clusters, customers)
        all_clusterings.append((clusters, metrics))

    winners = {}

    """
    At the end of the first phase, the five best sets of clusters
    are selected with regard to the above five parameters such
    that each set of clusters corresponds to a minimum value
    of the five parameters. Several parameters may take the
    minimum value for a selected set of clusters. 
    """

    for k in range(5):
        idx_min = min(range(len(all_clusterings)), key=lambda i: all_clusterings[i][1][k])
        winners[idx_min] = all_clusterings[idx_min][0]

    del dmat, all_clusterings
    gc.collect()

    return list(winners.values())


# ---------------------------------------------------------------------------
# Correr
# ---------------------------------------------------------------------------

def run_phase1(CUSTOMERS, VEHICLE_CAPACITY, if_print):
    if if_print:
        print("\n --- Fase 1 - Agrupamiento --- \n")

    top_sets = phase1_top_sets(CUSTOMERS, VEHICLE_CAPACITY)

    if if_print:
        print(f"Se obtuvieron {len(top_sets)} conjuntos ganadores (máx. 5):\n")
    
        for s, clusters in enumerate(top_sets, 1):
            p1, p2, p3, p4, p5 = evaluate_metrics(clusters, CUSTOMERS)
    
            print(f"  Conjunto {s}: {len(clusters)} clústeres  |  P1={p1:.5f}  P2={p2:.5f}  P3={p3:.5f}  P4={p4:.5f}  P5={p5:.5f}")
        
            for i, c in enumerate(clusters, 1):
                print(f"    Ruta{i}: {c}")
            print()

    return top_sets