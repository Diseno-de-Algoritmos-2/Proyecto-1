# ---------------------------------------------------------------------------
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from problemInstance import vehicle_capacity as VEHICLE_CAPACITY, customers as CUSTOMERS

from phaseOne import run_phase1
from phaseTwo import genetic_algorithm_tsp, route_demand
from plot import plot_routes

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
    clusters = run_phase1()
    
    # Phase 2: For each cluster, solve TSP using GA

    print(" --- Fase 2 - Algoritmo genetico --- \n")

    routes = []
    total_distance = 0

    for i, cluster in enumerate(clusters):

        best_route, route_dist = genetic_algorithm_tsp(cluster, depot, customers, vehicle_capacity)
        total_distance += route_dist
        
        route = [0] + best_route + [0]
        routes.append(route)

        route_demand_total = route_demand(best_route, customers)
        print(f"Vehicle {i+1}: {route}, Distance: {route_dist:.2f}, Demand: {route_demand_total}/{vehicle_capacity}")
    
    print(f"\nTotal distance: {total_distance:.2f}\n")
    return routes

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    
    routes = solve_cvrp(CUSTOMERS)
    plot_routes(CUSTOMERS, routes)