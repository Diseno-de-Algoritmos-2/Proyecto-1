
import matplotlib.pyplot as plt

# ---------------------------
# Visualization
# ---------------------------

def plot_routes(customers, routes, save_path="Paper/cvrp_solution.png"):

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

    # Save the figure before showing it
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    #plt.show()
