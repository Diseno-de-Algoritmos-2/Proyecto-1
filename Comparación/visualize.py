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


def run_data(SIMUL, param):

    todos_los_datos = {}

    for i in range(1, SIMUL + 1):
        with open(f"Comparación/results/{param}_iter_{i}.json", "r") as f:
            datos = json.load(f)
            todos_los_datos[i] = datos

    visualize_(todos_los_datos, param)


NUM_SIMUL = 100
run_data(NUM_SIMUL, "distances")
run_data(NUM_SIMUL, "times")
run_data(NUM_SIMUL, "utilization")
