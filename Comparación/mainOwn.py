import time
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Helper functions
# ---------------------------


def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# ---------------------------
# Phase 1: Organize Paths
# ---------------------------


def evaluate_truck_configurations(
    customers, num_vehicles, distances, vehicle_capacity, if_print
):
    """
    Evaluate all possible numbers of trucks between num_vehicles and the number of customers.
    Stop evaluating further configurations as soon as a longer solution is found.
    Return the solution with the minimum total distance.
    """
    best_paths = None
    best_utilization = None
    min_total_distance = float("inf")

    # Iterate over all possible numbers of trucks
    for k in range(num_vehicles, len(customers)):
        if if_print:
            print(f"Evaluating configuration with {k} vehicles...")
        paths = [[0] for _ in range(k)]  # Initialize k paths (one for each vehicle)

        try:
            # Attempt to create paths for the current number of vehicles
            current_paths, utilization, total_distance = create_paths(
                paths, distances, customers, vehicle_capacity, if_print
            )

            # If the current solution is worse than the best one, stop searching
            if total_distance >= min_total_distance:
                if if_print:
                    print(f"Stopping search: Found longer solution with {k} vehicles.")
                break

            # Update the best solution if the current configuration has a smaller total distance
            min_total_distance = total_distance
            best_paths = current_paths
            best_utilization = utilization

        except ValueError:
            # If no valid route is possible with the current number of vehicles, assign infinite distance
            if if_print:
                print(
                    f"No valid solution with {k} vehicles. Assigning infinite distance."
                )
            continue

    # If no valid solution was found, raise an error
    if best_paths is None:
        raise ValueError("No valid solution found for any number of vehicles.")

    if if_print:
        print(f"\nBest solution found with total distance: {min_total_distance:.2f}")
    return best_paths, best_utilization, min_total_distance


def create_paths(paths, distances, customers, vehicle_capacity, if_print):
    """
    Create paths for the given number of vehicles while respecting vehicle capacity.
    """
    demand_per_path = [0 for _ in range(len(paths))]

    for customer_index, dist in distances:
        best_path_index = -1
        best_path_distance = float("inf")

        for path_index, path in enumerate(paths):
            # Check if the customer can be added to the path without exceeding the vehicle capacity
            path_demand = demand_per_path[path_index]
            customer_demand = customers[customer_index]["demand"]
            if path_demand + customer_demand <= vehicle_capacity:
                # Calculate the total distance if the customer is added to this path
                path_distance = sum(
                    euclidean_distance(
                        customers[path[j]]["coord"], customers[path[j + 1]]["coord"]
                    )
                    for j in range(len(path) - 1)
                )
                path_distance += euclidean_distance(
                    customers[path[-1]]["coord"], customers[customer_index]["coord"]
                )
                path_distance += euclidean_distance(
                    customers[customer_index]["coord"], customers[0]["coord"]
                )

                # Check if this path is better than the best path found so far
                if path_distance < best_path_distance:
                    best_path_distance = path_distance
                    best_path_index = path_index

        # If a valid path is found, add the customer to the best path
        if best_path_index != -1:
            paths[best_path_index].append(customer_index)
            demand_per_path[best_path_index] += customers[customer_index]["demand"]
        else:
            # If no valid path is found, raise an exception
            raise ValueError(f"Cannot assign customer {customer_index} to any path.")

    # Add the depot (0) at the end of each path to return to the depot
    for path in paths:
        path.append(0)

    # Calculate total distance and utilization
    total_distance = 0
    utilization = []
    for i, path in enumerate(paths):
        route_demand_total = sum([customers[i]["demand"] for i in path])
        route_dist = sum(
            [
                euclidean_distance(
                    customers[path[j]]["coord"], customers[path[j + 1]]["coord"]
                )
                for j in range(len(path) - 1)
            ]
        )
        utilization.append(route_demand_total / vehicle_capacity)
        total_distance += route_dist

    return paths, utilization, total_distance


def solve_cvrp(customers, vehicle_capacity, if_print):
    """
    Solve the CVRP problem by evaluating all possible numbers of trucks.
    """
    # Calculate the total demand of all customers
    total_demand = sum(customer["demand"] for customer in customers[1:])

    # Calculate the minimum number of vehicles required based on vehicle capacity
    num_vehicles = int(np.ceil(total_demand / vehicle_capacity))

    # Calculate the distance from the depot to each customer and sort them
    depot = customers[0]["coord"]
    distances = [
        (i, euclidean_distance(depot, customers[i]["coord"]))
        for i in range(1, len(customers))
    ]
    distances.sort(key=lambda x: x[1])  # Sort customers by distance from the depot

    # Evaluate all truck configurations and return the best solution
    best_paths, best_utilization, best_dist = evaluate_truck_configurations(
        customers, num_vehicles, distances, vehicle_capacity, if_print
    )
    return best_paths, best_utilization, best_dist


# ---------------------------
# Visualization
# ---------------------------
def plot_routes(customers, routes, save_path="Own/cvrp_solution.png"):

    depot = customers[0]["coord"]
    colormap = plt.colormaps.get_cmap("Set1")  # Use a colormap for distinct colors
    colors = [colormap(i / len(routes)) for i in range(len(routes))]  # Generate colors

    plt.figure(figsize=(8, 6))
    # Plot depot
    plt.plot(depot[0], depot[1], "rs", markersize=12, label="Depot")

    # Plot customers
    for i in range(1, len(customers)):
        pt = customers[i]["coord"]
        plt.plot(pt[0], pt[1], "bo")
        plt.text(pt[0] + 0.0025, pt[1] + 0.0025, str(i), fontsize=9)

    # Plot routes with different colors.
    for idx, route in enumerate(routes):
        route_coords = [customers[i]["coord"] for i in route]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, "-", color=colors[idx], linewidth=2, label=f"Vehicle {idx+1}")

    plt.title("CVRP Solution Routes")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()


# ---------------------------
# Example usage
# ---------------------------
def run_own(CUSTOMERS, vehicle_capacity, if_print):

    # Solve the CVRP
    start = time.perf_counter()
    routes, utilization, best_dist = solve_cvrp(CUSTOMERS, vehicle_capacity, if_print)
    end = time.perf_counter()

    execution_time = end - start
    # Visualize the solution
    # plot_routes(CUSTOMERS, routes)

    return routes, utilization, best_dist, execution_time
