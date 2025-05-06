import matplotlib.pyplot as plt
import json
import os


# Gráficar los datos
# --------------------------------------------------------------------------------


def visualize_(data_file, name):
    num_clients = [i["num_clients"] for i in data_file.values()]
    paper_metrics = [i["paper"] for i in data_file.values()]
    own_metrics = [i["own"] for i in data_file.values()]
    gap_values = [i["gap"] for i in data_file.values()]

    plt.figure(figsize=(15, 5))

    # Scatter plot for 'paper' and 'own' metrics vs num_clients
    plt.subplot(1, 2, 1)
    plt.scatter(num_clients, paper_metrics, label="Paper", color="blue")
    plt.scatter(num_clients, own_metrics, label="Own", color="green")
    plt.title(f"{name.capitalize()} vs Num Clients")
    plt.xlabel("Num Clients")
    plt.ylabel(name.capitalize())
    plt.legend()

    # Boxplot for 'gap' values
    plt.subplot(1, 2, 2)
    plt.boxplot(gap_values)
    plt.title("Boxplot of Gap Values")
    plt.ylabel("Gap (%)")

    plt.tight_layout()
    plt.savefig(f"Comparación/graphs/comparison_{name}.png")


# Leer los datos
# --------------------------------------------------------------------------------


def visualize_distances(data_file):
    num_clients = [i["num_clients"] for i in data_file.values()]
    paper_metrics = [i["paper"] for i in data_file.values()]
    own_metrics = [i["own"] for i in data_file.values()]
    worst_case_distances = [i["worst_case_distance"] for i in data_file.values()]

    # Calculate how much better each algorithm is compared to the worst case
    paper_vs_worst = [
        (worst - paper) / worst * 100
        for paper, worst in zip(paper_metrics, worst_case_distances)
    ]
    own_vs_worst = [
        (worst - own) / worst * 100
        for own, worst in zip(own_metrics, worst_case_distances)
    ]

    plt.figure(figsize=(15, 5))

    # Scatter plot for 'paper', 'own', and 'worst_case_distance' vs num_clients
    plt.subplot(1, 2, 1)
    plt.scatter(num_clients, paper_metrics, label="Paper", color="blue")
    plt.scatter(num_clients, own_metrics, label="Own", color="green")
    plt.scatter(
        num_clients, worst_case_distances, label="Worst Case", color="red", alpha=0.6
    )
    plt.title("Total Route Distance (Logarithmic Scale) vs Num Clients")
    plt.xlabel("Num Clients")
    plt.ylabel("Distance")
    # make the y-axis logarithmic
    plt.yscale("log")
    plt.legend()

    # Boxplot for improvement over the worst case
    plt.subplot(1, 2, 2)
    plt.boxplot(
        [paper_vs_worst, own_vs_worst], labels=["Paper vs Worst", "Own vs Worst"]
    )
    plt.title("Improvement Over Worst Case")
    plt.ylabel("Improvement (%)")

    plt.tight_layout()
    plt.savefig("Comparación/graphs/comparison_distances_special.png")


# Modify run_data to handle the special case for distances
def run_data(SIMUL, param):
    todos_los_datos = {}

    for i in range(1, SIMUL + 1):
        with open(f"Comparación/results/{param}_iter_{i}.json", "r") as f:
            datos = json.load(f)
            todos_los_datos[i] = datos

    if param == "distances":
        visualize_distances(todos_los_datos)
    else:
        visualize_(todos_los_datos, param)


NUM_SIMUL = 500
run_data(NUM_SIMUL, "distances")
run_data(NUM_SIMUL, "times")
run_data(NUM_SIMUL, "trucks")
run_data(NUM_SIMUL, "utilization")
