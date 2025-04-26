import random

from saveSimulation import save_simulation_to_json
from visualize import visualize_

# Traer la informaciión y algoruitmos.
from problemInstanceReal import vehicle_capacity as VEHICLE_CAPACITY, len_clientes as LEN_CLIENTES, get_clientes
from mainPaper import run_paper
from mainOwn import run_own

DISTANCES_ = {}
TIMES_ = {}
UTILIZATION_ = {}

SIMULATIONS = 10
IF_PRINT = True

for i in range(SIMULATIONS):

    size = random.randint(5, LEN_CLIENTES)
    print(f"Simulación {i+1} de {SIMULATIONS} con {size} clientes.")

    CUSTOMERS = get_clientes(size)

    # Correr el algoritmo del Paper
    routes_paper, utilization_paper, best_dist_paper, exceution_time_paper = run_paper(CUSTOMERS, VEHICLE_CAPACITY, False)
    routes_own, utilization_own, best_dist_own, exceution_time_own = run_own(CUSTOMERS, VEHICLE_CAPACITY, False)

    # Guardar resultados
    
    # 1. Distancias
    DISTANCES_[i] = {
        "num_clients": size,
        "paper": best_dist_paper,
        "own": best_dist_own,
        "gap": (best_dist_own / best_dist_paper - 1) * 100
    }
    
    # 2. Tiempos
    TIMES_[i] = {
        "num_clients": size,
        "paper": exceution_time_paper,
        "own": exceution_time_own,
        "gap": (exceution_time_own / exceution_time_paper - 1) * 100
    }
    
    # 3. Utilización
    utilization_paper = sum(utilization_paper) / len(utilization_paper)
    utilization_own = sum(utilization_own) / len(utilization_own)
    
    UTILIZATION_[i] = {
        "num_clients": size,
        "paper": utilization_paper,
        "own": utilization_own,
        "gap": (utilization_own / utilization_paper - 1) * 100
    }



save_simulation_to_json(DISTANCES_, "distances")
save_simulation_to_json(TIMES_, "times")
save_simulation_to_json(UTILIZATION_, "utilization")

visualize_(DISTANCES_, "distances")
visualize_(TIMES_, "times")
visualize_(UTILIZATION_, "utilization")