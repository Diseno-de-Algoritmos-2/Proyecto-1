import random

from saveSimulation import save_simulation_to_json

# Traer la informaciión y algoruitmos.
from problemInstanceReal import (
    vehicle_capacity as VEHICLE_CAPACITY,
    len_clientes as LEN_CLIENTES,
    get_clientes,
)
from mainPaper import run_paper
from mainOwn import run_own

DISTANCES_ = {}
TIMES_ = {}
UTILIZATION_ = {}

SIM = 100
IF_PRINT = False

size = random.randint(5, LEN_CLIENTES)
print("\n-------------------------------------------------------")
print(f"Simulación {SIM} con {size} clientes.")
print("-------------------------------------------------------\n")

CUSTOMERS = get_clientes(size)

# Correr el algoritmo del Paper
routes_paper, utilization_paper, best_dist_paper, exceution_time_paper = run_paper(
    CUSTOMERS, VEHICLE_CAPACITY, IF_PRINT
)
routes_own, utilization_own, best_dist_own, exceution_time_own = run_own(
    CUSTOMERS, VEHICLE_CAPACITY, IF_PRINT
)
# Guardar resultados
print(f"Distance Paper: {best_dist_paper} in {exceution_time_paper} seg\n")
print(f"Distance Own: {best_dist_own} in {exceution_time_own} seg \n")

# 1. Distancias
DISTANCES_ = {
    "num_clients": size,
    "paper": best_dist_paper,
    "own": best_dist_own,
    "gap": (best_dist_own / best_dist_paper - 1) * 100,
}

# 2. Tiempos
TIMES_ = {
    "num_clients": size,
    "paper": exceution_time_paper,
    "own": exceution_time_own,
    "gap": (exceution_time_own / exceution_time_paper - 1) * 100,
}

# 3. Utilización
utilization_paper = sum(utilization_paper) / len(utilization_paper)
utilization_own = sum(utilization_own) / len(utilization_own)

UTILIZATION_ = {
    "num_clients": size,
    "paper": utilization_paper,
    "own": utilization_own,
    "gap": (utilization_own / utilization_paper - 1) * 100,
}


save_simulation_to_json(DISTANCES_, f"distances_iter_{SIM}")
save_simulation_to_json(TIMES_, f"times_iter_{SIM}")
save_simulation_to_json(UTILIZATION_, f"utilization_iter_{SIM}")
