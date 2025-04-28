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

def solve_cvrp(customers, VEHICLE_CAPACITY, if_print):
    
    # Calculate the number of vehicles needed, this is the total demand divided by the vehicle capacity
    total_demand = 0
    for customer in customers[1:]:
        total_demand += customer['demand']
        
    num_vehicles = int(total_demand / VEHICLE_CAPACITY)
    
    # Create the list of k paths
    paths = [[0] for _ in range(num_vehicles)]
    
    # Calculate the distance from the depot to each customer and sort them
    depot = customers[0]['coord']
    distances = []
    
    for i in range(1, len(customers)):
        dist = euclidean_distance(depot, customers[i]['coord'])
        distances.append((i, dist))
        
    # Sort the customers by distance from the depot
    distances.sort(key=lambda x: x[1])
    
    # Recursively add the next closest customer to a path
    # so that the total distance of all paths is minimized
    path, utilization, best_dist = create_paths(paths, distances, customers, VEHICLE_CAPACITY, if_print)
    
    return path, utilization, best_dist 

def create_paths(paths, distances, customers, VEHICLE_CAPACITY, if_print):
    
    # Iterate through the sorted distances
    # and iteratively add the closes so that the total
    # distance of all paths is minimized
    # while respecting the vehicle capacity

    # this should have a complexity of k * n, where
    # k is the number of vehicles and n is the number of customers
    
    distance_per_path = [[0] for _ in range(len(paths))]
    demand_per_path = [0 for _ in range(len(paths))]

    
    for customer_index, dist in distances:
        
        best_path_index = -1
        best_path_distance = float('inf')
        
        for path in paths:
            
            # Check if the customer can be added to the path
            # without exceeding the vehicle capacity
            path_demand = sum([customers[i]['demand'] for i in path])
            customer_demand = customers[customer_index]['demand']
            if path_demand + customer_demand <= VEHICLE_CAPACITY:
                
                # Calculate the total distance. this is the sum of the
                # distance in all paths, plus the distance from the last element of the
                # current path to the customer, plus the distance from the customer to the depot
                
                path_distance = 0
                
                for p in paths:
                    path_distance += sum(p)
                
                path_distance += euclidean_distance(customers[path[-1]]['coord'], customers[customer_index]['coord'])
                path_distance += euclidean_distance(customers[customer_index]['coord'], customers[0]['coord'])
                
                # Check if this path is better than the best path found so far
                if path_distance < best_path_distance:
                    best_path_distance = path_distance
                    best_path_index = paths.index(path)
                    
        # If a path was found, add the customer to the path
        if best_path_index != -1:     
            
            # Add the customer to the path
            paths[best_path_index].append(customer_index)
            
            # Update the distance and demand for the path
            demand_per_path[best_path_index] += customers[customer_index]['demand']
            distance_per_path[best_path_index] += euclidean_distance(customers[paths[best_path_index][-1]]['coord'], customers[customer_index]['coord'])

        else:   
            # If no path was found, create a new path
            new_path = [0, customer_index]
            paths.append(new_path)
            
            # Update the distance and demand for the new path
            distance_per_path.append(euclidean_distance(customers[0]['coord'], customers[customer_index]['coord']))
            demand_per_path.append(customers[customer_index]['demand'])
            
            if if_print: print(f"Created new path for customer {customer_index}")
            
            
    # add 0 at the end of each path to return to the depot
    for path in paths:
        path.append(0)
        
        
    total_distance = 0
    utilization = []
    for i, path in enumerate(paths):
        route = path
        route_demand_total = sum([customers[i]['demand'] for i in path])
        route_dist = sum([euclidean_distance(customers[path[j]]['coord'], customers[path[j+1]]['coord']) for j in range(len(path)-1)])
        utilization.append(route_demand_total / VEHICLE_CAPACITY)
        if if_print: print(f"Vehicle {i+1}: Route: {route}, Distance: {route_dist:.2f}, Demand: {route_demand_total}")
        
        total_distance += route_dist
    
    if if_print: print(f"\nTotal distance: {total_distance:.2f}")

    return paths, utilization, total_distance 



# ---------------------------
# Visualization
# ---------------------------
def plot_routes(customers, routes, save_path="Own/cvrp_solution.png"):

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
        plt.text(pt[0]+0.0025, pt[1]+0.0025, str(i), fontsize=9)
    
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()

# ---------------------------
# Example usage
# ---------------------------
def run_own(CUSTOMERS, VEHICLE_CAPACITY, if_print):
    
    # Solve the CVRP
    start = time.perf_counter()
    routes, utilization, best_dist = solve_cvrp(CUSTOMERS, VEHICLE_CAPACITY, if_print)
    end = time.perf_counter()
    
    execution_time = end - start
    # Visualize the solution
    #plot_routes(CUSTOMERS, routes)

    return routes, utilization, best_dist, execution_time