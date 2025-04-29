# ---------------------------------------------------------------------------
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Improved_Paper.phaseOne import run_phase1
from Improved_Paper.phaseTwo import genetic_algorithm_tsp
from Improved_Paper.plot import plot_routes

# ---------------------------------------------------------------------------
# Función auxiliar necesaria
# ---------------------------------------------------------------------------


def route_demand(route, customers):
    """Calcula la demanda total de una ruta."""
    return sum(customers[i]["demand"] for i in route if i < len(customers))


# ---------------------------
# Main CVRP solution
# ---------------------------


def solve_cvrp(customers, vehicle_capacity, if_print):
    """
    customers: list of dicts with keys 'coord' (tuple (x,y)) and 'demand' (number)
    The depot is assumed to be customers[0].
    """

    depot = customers[0]["coord"]

    # Phase 1: clustering
    top_sets = run_phase1(customers, vehicle_capacity, if_print)
    print(f"--> Done Phase 1 ({len(top_sets)} sets)")

    # Phase 2: For each cluster, solve TSP using GA

    if if_print:
        print(" --- Fase 2 - Algoritmo genetico --- \n")

    best_cluster = []
    best_dist = float("inf")

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
            if if_print:
                print(f"    Nuevo Mejor Cluster | Total distance: {best_dist:.2f}\n")

    utilization = []
    for i, route in enumerate(best_cluster):
        route_demand_total = route_demand(route, customers)
        utilization.append(route_demand_total / vehicle_capacity)
        if if_print:
            print(
                f"    Vehicle {i+1}: {route}, Distance: {route_dist:.2f}, Demand: {route_demand_total}/{vehicle_capacity}"
            )

    if if_print:
        print(f"\n  Total distance: {best_dist:.2f}\n")
    print("--> Done Phase 2\n")
    return routes, utilization, best_dist


# ---------------------------
# Run
# ---------------------------
def run_paper(CUSTOMERS, VEHICLE_CAPACITY, if_print):

    start = time.perf_counter()
    routes, utilization, best_dist = solve_cvrp(CUSTOMERS, VEHICLE_CAPACITY, if_print)
    end = time.perf_counter()

    exceution_time = end - start
    # plot_routes(CUSTOMERS, routes)

    return routes, utilization, best_dist, exceution_time
