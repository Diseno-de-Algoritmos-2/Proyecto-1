import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random

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
ELITISM_RATE = 0.25

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


def convex_hull_area(points):
    """Returns the area of the convex hull of the given points.
    If less than 3 points, area is zero."""
    if len(points) < 3:
        return 0
    pts = np.array(points)
    hull = ConvexHull(pts)
    return hull.area

# ---------------------------
# Phase 1: Clustering
# ---------------------------
def cluster_customers(customers, vehicle_capacity=VEHICLE_CAPACITY, n_iter=N_CLUSTER_ITER):
    """
    Creates several sets of clusters (one set per iteration) using a greedy approach.
    Each cluster must not exceed the vehicle capacity.
    Returns the set of clusters (list of lists of customer indices) with the best quality.
    """
    best_clusters = None
    best_quality = float('inf')
    
    # For quality metric, we compute for each cluster: convex_hull_area(cluster_points)/number of vertices
    # then sum over clusters.
    
    for _ in range(n_iter):

        # Shuffle customer indices (customers are indexed 1..n; index 0 is the depot)
        customer_indices = list(range(1, len(customers)))
        random.shuffle(customer_indices)
        clusters = []

        # Maintain a record of current load for each cluster.
        clusters_load = []
        
        for cust_idx in customer_indices:

            demand = customers[cust_idx]['demand']
            assigned = False

            # Try to assign to an existing cluster if capacity permits.
            # Here, we assign to the cluster that would minimally increase the "spread" (distance to cluster center).
            best_increase = float('inf')
            best_cluster = -1

            for i, cluster in enumerate(clusters):
                if clusters_load[i] + demand <= vehicle_capacity:

                    # compute current cluster center (average of coordinates)
                    pts = [customers[j]['coord'] for j in cluster]
                    center = np.mean(pts, axis=0)
                    increase = euclidean_distance(center, customers[cust_idx]['coord'])
                    if increase < best_increase:
                        best_increase = increase
                        best_cluster = i

            if best_cluster >= 0:
                clusters[best_cluster].append(cust_idx)
                clusters_load[best_cluster] += demand
                assigned = True

            # If cannot assign, start a new cluster.
            if not assigned:
                clusters.append([cust_idx])
                clusters_load.append(demand)
        
        # Compute quality metric
        quality = 0

        for cluster in clusters:
            pts = [customers[i]['coord'] for i in cluster]
            area = convex_hull_area(pts)
            quality += area / len(cluster)
        
        if quality < best_quality:
            best_quality = quality
            best_clusters = clusters

    return best_clusters

# ---------------------------
# Phase 2: GA for TSP in each cluster
# ---------------------------
def route_distance(route, depot, customers):
    """Compute total distance for a route: depot -> customers in order -> depot"""
    total = 0
    total += euclidean_distance(depot, customers[route[0]]['coord'])

    for i in range(len(route) - 1):
        total += euclidean_distance(customers[route[i]]['coord'], customers[route[i+1]]['coord'])

    total += euclidean_distance(customers[route[-1]]['coord'], depot)
    return total

# Mutation operators
def mutation_flip(route):
    """Select a random segment and reverse it"""
    new_route = route.copy()

    if len(new_route) < 2:
        return new_route
    
    i, j = sorted(random.sample(range(len(new_route)), 2))
    new_route[i:j+1] = new_route[i:j+1][::-1]

    return new_route

def mutation_swap(route):
    """Swap two random positions"""
    new_route = route.copy()

    if len(new_route) < 2:
        return new_route
    
    i, j = random.sample(range(len(new_route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]

    return new_route

def mutation_shift(route):
    """Remove an element and insert it at a different random position"""
    new_route = route.copy()

    if len(new_route) < 2:
        return new_route
    
    i = random.randrange(len(new_route))
    elem = new_route.pop(i)
    j = random.randrange(len(new_route) + 1)
    new_route.insert(j, elem)

    return new_route

def generate_initial_population(route, population_size=POPULATION_SIZE):
    """Generate an initial population of random permutations of the route."""
    population = []
    for _ in range(population_size):
        perm = route.copy()
        random.shuffle(perm)
        population.append(perm)

    return population

def route_demand(route, customers):
    """Calculate the total demand of a route."""
    return sum(customers[i]['demand'] for i in route)

def genetic_algorithm_tsp(cluster, depot, customers, population_size=POPULATION_SIZE, generations=GENERATIONS, elitism_rate=ELITISM_RATE, vehicle_capacity=VEHICLE_CAPACITY):

    """
    Solves TSP for a given cluster using a GA with only mutation operators.
    'cluster' is a list of customer indices.
    Ensures that the route does not exceed the vehicle capacity.
    """

    # Initialize population with random permutations of the cluster.
    population = generate_initial_population(cluster, population_size)
    elite_size = int(elitism_rate * population_size)
    
    best_route = None
    best_distance = float('inf')
    
    for gen in range(generations):

        # Evaluate fitness (lower distance is better, penalize capacity violations)
        fitness = []
        for route in population:
            distance = route_distance(route, depot, customers)
            demand = route_demand(route, customers)
            
            if demand > vehicle_capacity:
                distance += 1e6
                
            fitness.append(distance)
        
        for i, d in enumerate(fitness):
            if d < best_distance:
                best_distance = d
                best_route = population[i]
                
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]

        # Elitism: keep best individuals
        new_population = sorted_population[:elite_size]
        
        # Generate the rest of the new population by mutation.
        while len(new_population) < population_size:

            # Select a parent randomly from the elite set.
            parent = random.choice(sorted_population[:elite_size])

            # Randomly choose one of the mutation operators.
            op = random.choice([mutation_flip, mutation_swap, mutation_shift])
            child = op(parent)
            new_population.append(child)
        
        population = new_population
        
    return best_route, best_distance



# ---------------------------
# Main CVRP solution
# ---------------------------

def solve_cvrp(customers, vehicle_capacity=VEHICLE_CAPACITY, n_cluster_iter=N_CLUSTER_ITER):

    """
    customers: list of dicts with keys 'coord' (tuple (x,y)) and 'demand' (number)
    The depot is assumed to be customers[0].
    """

    depot = customers[0]['coord']

    # Phase 1: clustering
    clusters = cluster_customers(customers, vehicle_capacity, n_iter=n_cluster_iter)
    print(f"Found {len(clusters)} clusters (vehicles).\n")
    
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")
    print()    
    
    # Phase 2: For each cluster, solve TSP using GA
    routes = []
    total_distance = 0

    for i, cluster in enumerate(clusters):
        best_route, route_dist = genetic_algorithm_tsp(cluster, depot, customers)
        total_distance += route_dist
        # Prepend and append depot for the route
        route = [0] + best_route + [0]
        routes.append(route)
        
        # Calculate total demand for the route
        route_demand_total = route_demand(best_route, customers)
        print(f"Vehicle {i+1}: Route: {route}, Distance: {route_dist:.2f}, Demand: {route_demand_total}")
        
        # Check if the demand exceeds the vehicle capacity
        if route_demand_total > vehicle_capacity:
            print(f"Warning: Vehicle {i+1} exceeds capacity! (Demand: {route_demand_total}, Capacity: {vehicle_capacity})")
    
    print(f"\nTotal distance: {total_distance:.2f}")
    return routes


# ---------------------------
# Visualization
# ---------------------------

def plot_routes(customers, routes):

    depot = customers[0]['coord']
    colormap = plt.colormaps.get_cmap('Set1')  # Use a colormap for distinct colors
    colors = [colormap(i / len(routes)) for i in range(len(routes))]  # Generate colors
    
    plt.figure(figsize=(8, 6))
    # Plot depot
    plt.plot(depot[0], depot[1], 'rs', markersize=12, label="Depot")
    
    # Plot customers
    for i in range(1, len(customers)):
        pt = customers[i]['coord']
        plt.plot(pt[0], pt[1], 'bo')
        plt.text(pt[0]+0.2, pt[1]+0.2, str(i), fontsize=9)
    
    # Plot routes with different colors.
    for idx, route in enumerate(routes):
        route_coords = [customers[i]['coord'] for i in route]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, '-', color=colors[idx], linewidth=2, label=f"Vehicle {idx+1}")
    
    plt.title("CVRP Solution Routes")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    
    # Solve the CVRP
    routes = solve_cvrp(CUSTOMERS)
    
    # Visualize the solution
    plot_routes(CUSTOMERS, routes)