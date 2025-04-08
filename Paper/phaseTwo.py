import numpy as np
import random
from typing import List, Dict, Tuple

# ---------------------------
# Parámetros (idénticos)
# ---------------------------
POPULATION_SIZE = 100
GENERATIONS = 400
ELITISM_RATE = 0.25

# ---------------------------
# Funciones de Ayuda
# ---------------------------

def route_demand(route, customers):
    return sum(customers[i]['demand'] for i in route)

# ---------------------------
# Operadores de mutación
# ---------------------------

def mutation_flip(route):
    if len(route) < 2:
        return route.copy()
    i, j = sorted(random.sample(range(len(route)), 2))
    return route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]


def mutation_swap(route):
    if len(route) < 2:
        return route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route = route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def mutation_shift(route):
    if len(route) < 2:
        return route.copy()
    i = random.randrange(len(route))
    gene = route[i]
    remainder = route[:i] + route[i + 1 :]
    j = random.randrange(len(remainder) + 1)
    return remainder[:j] + [gene] + remainder[j:]

_MUT_OPS = [mutation_flip, mutation_swap, mutation_shift]

# ---------------------------
# GA optimizado
# ---------------------------

def _build_distance_matrix(coords):
    n = len(coords)
    diff = np.expand_dims(coords, 1) - np.expand_dims(coords, 0)
    return np.sqrt((diff ** 2).sum(axis=2))


def _route_distance_np(route_local, dmat):
    if not route_local:
        return 0.0
    # Construir secuencia [0, r0, r1, ..., rn, 0] con índices locales
    seq = np.array([0] + route_local + [0])
    return dmat[seq[:-1], seq[1:]].sum()


def _generate_initial_population(nodes_local):
    pop = []
    for _ in range(POPULATION_SIZE):
        perm = nodes_local.copy()
        random.shuffle(perm)
        pop.append(perm)
    return pop


def genetic_algorithm_tsp(cluster, depot_coord, customers, vehicle_capacity,):

    # ---------------- Prep datos locales ----------------
    coords_local = [depot_coord] + [customers[i]["coord"] for i in cluster]
    dmat = _build_distance_matrix(np.array(coords_local))
    local_nodes = list(range(1, len(coords_local)))
    local_final_global = {loc: glob for loc, glob in zip(local_nodes, cluster)}

    # Demanda total (constante para todas las permutaciones)
    total_demand = sum(customers[i]["demand"] for i in cluster)
    if total_demand > vehicle_capacity:
        raise ValueError("Cluster demand exceeds vehicle capacity")

    # ---------------- Inicialización ----------------
    population = _generate_initial_population(local_nodes)
    elite_size = int(ELITISM_RATE * POPULATION_SIZE)

    best_route_local: List[int] | None = None
    best_distance = float("inf")

    for _ in range(GENERATIONS):
        # 1) Fitness vectorizado
        fitness = np.fromiter((_route_distance_np(ind, dmat) for ind in population), dtype=float, count=POPULATION_SIZE,)

        # 2) Actualizar mejor global
        idx_best = int(fitness.argmin())
        if fitness[idx_best] < best_distance:
            best_distance = float(fitness[idx_best])
            best_route_local = population[idx_best].copy()

        # 3) Ordenar población por fitness
        order = fitness.argsort()
        sorted_pop = [population[i] for i in order]

        # 4) Elitismo
        new_population = sorted_pop[:elite_size]

        # 5) Rellenar con descendencia por mutación de la élite
        while len(new_population) < POPULATION_SIZE:
            parent = random.choice(sorted_pop[:elite_size])
            child = random.choice(_MUT_OPS)(parent)
            new_population.append(child)

        population = new_population

    best_route_global = [local_final_global[idx] for idx in best_route_local] if best_route_local else []
    return best_route_global, best_distance