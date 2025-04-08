import numpy as np
from scipy.spatial import ConvexHull

# ---------------------------
# IMPORT PROBLEM INSTANCE FROM FILE
# ---------------------------

from problemInstance import vehicle_capacity, customers

VEHICLE_CAPACITY = vehicle_capacity
CUSTOMERS = customers

# ---------------------------
# Algorithm Parameters
# ---------------------------
N_CLUSTER_ITER = 20
POPULATION_SIZE = 100
GENERATIONS = 400
ELITISM_RATE = 0.1

# ---------------------------
# Helper functions
# ---------------------------
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def compute_distance_matrix(points):
    n = len(points)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dmat[i][j] = euclidean_distance(points[i], points[j])
    return dmat

# -------- PAREMETERS -----------
# The following functions are used to compute the quality metric for the clusters.

# 1. Paremeter 1
def convex_average_hull_area(points, cluster):
    """Returns the area of the convex hull of the given points.
    If less than 3 points, area is zero."""

    if len(points) < 3:
        return 0
    
    pts = np.array(points)
    hull = ConvexHull(pts)

    return hull.area / len(cluster)

# 2. Paremeter 2 (not used in the original code)
def convex_hull_area(points):

    if len(points) < 3:
        return 0
    
    pts = np.array(points)
    hull = ConvexHull(pts)
    
    return hull.area

# Parameter 3: (not used in the original code)
def convex_average_demand_hull_area(points, cluster):

    if len(points) < 3:
        return 0
    
    pts = np.array(points)
    hull = ConvexHull(pts)

    total_demand = sum([customers[i]['demand'] for i in cluster])

    return hull.area / total_demand

# Parameter 4: (not used in the original code)
def mean_distance_from_centroid(points):
    """Returns the mean distance of the vertices in the cluster from the centroid of that cluster."""

    if len(points) < 2:
        return 0
    
    pts = np.array(points)
    centroid = np.mean(pts, axis=0)
    distances = [euclidean_distance(pt, centroid) for pt in pts]

    return np.mean(distances)

# Parameter 5: (not used in the original code)
def mean_distance_from_centroid_avg_demand(points, cluster):
    """Returns the mean distance of the vertices in the cluster from the centroid of that cluster
    weighted by the total demand of the cluster."""

    if len(points) < 2:
        return 0
    
    pts = np.array(points)
    centroid = np.mean(pts, axis=0)
    distances = [euclidean_distance(pt, centroid) for pt in pts]

    total_demand = sum([customers[i]['demand'] for i in cluster])

    return np.mean(distances) / total_demand