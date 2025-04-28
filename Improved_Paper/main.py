# ---------------------------------------------------------------------------
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problemInstance import vehicle_capacity as VEHICLE_CAPACITY, customers as CUSTOMERS

from phaseOne import run_phase1
from phaseTwo import genetic_algorithm_tsp
from plot import plot_routes

# ---------------------------------------------------------------------------
# Función auxiliar necesaria
# ---------------------------------------------------------------------------

def route_demand(route, customers):
    """Calcula la demanda total de una ruta."""
    return sum(customers[i]["demand"] for i in route if i < len(customers))

# ---------------------------
# Main CVRP solution
# ---------------------------

def solve_cvrp(customers, vehicle_capacity=VEHICLE_CAPACITY):
    """
    customers: list of dicts with keys 'coord' (tuple (x,y)) and 'demand' (number)
    The depot is assumed to be customers[0].
    """

    depot = customers[0]["coord"]

    # Phase 1: clustering
    top_sets = run_phase1(customers, VEHICLE_CAPACITY, True)
    
    # Phase 2: For each cluster, solve TSP using GA

    print(" --- Fase 2 - Algoritmo Genético ---\n")

    best_cluster = []
    best_dist = float('inf')

    for clusters in top_sets:
        routes = []
        total_distance = 0

        for i, cluster in enumerate(clusters):
            best_route, route_dist = genetic_algorithm_tsp(cluster, depot, customers)
            total_distance += route_dist

            route = [0] + best_route + [0]
            routes.append(route)

        if total_distance < best_dist:
            best_dist = total_distance
            best_cluster = routes
            print(f"    Nuevo Mejor Cluster | Total distance: {best_dist:.2f}\n")

    for i, route in enumerate(best_cluster):
        route_demand_total = route_demand(route, customers)
        print(f"    Vehicle {i+1}: {route}, Distance: {best_dist:.2f}, Demand: {route_demand_total}/{vehicle_capacity}")

    print(f"\n  Total distance: {best_dist:.2f}\n")
    return best_cluster

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    routes = solve_cvrp(CUSTOMERS)
    plot_routes(CUSTOMERS, routes)