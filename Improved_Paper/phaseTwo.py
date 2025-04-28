import numpy as np
import random
import numba

# ----------------------------------------
# Parámetros
# ----------------------------------------
"""
The population
size is set to 100 and the Tournament Selection is used
as the selection operator of the Genetic algorithm with
tournament size 4. The Swap mutation operator is applied 
under the mutation probability
0.9. The best 10% chromosomes of the current population
are directly sent to the next generation without any change
(Elitism rate = 0.1). The stopping criterion is reaching 1000
number of generations or fitness value is not improved
within 300 number of generations. 
"""

POPULATION_SIZE = 100
MAX_GENERATIONS = 1000
NO_IMPROVEMENT_LIMIT = 300
ELITISM_RATE = 0.10
TOURNAMENT_SIZE = 4
MUTATION_PROB = 0.9


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

"""
The fitness function is constructed according to the objective of the problem and used to evaluate the
fitness value of each chromosome in the population. The quality of each chromosome is reflected by the fitness
value. The fitness value is used by the genetic algorithm to select chromosomes for the reproduction of the new
generations. In other words, the chance of a chromosome being selected for reproduction is directly depends on
its fitness value. 
"""

def build_distance_matrix(coords):
    coords_np = np.array(coords)
    diff = coords_np[:, np.newaxis, :] - coords_np[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)

def evaluate_population(population, dmat):
    routes = np.hstack([np.zeros((population.shape[0], 1), dtype=int), population, np.zeros((population.shape[0], 1), dtype=int)])
    idx_from = routes[:, :-1]
    idx_to = routes[:, 1:]
    return np.sum(dmat[idx_from, idx_to], axis=1)

def tournament_selection(fitness):
    contenders = np.random.randint(0, len(fitness), size=(POPULATION_SIZE, TOURNAMENT_SIZE))
    best_idx = contenders[np.arange(POPULATION_SIZE), np.argmin(fitness[contenders], axis=1)]
    return best_idx

# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

"""
Tournament Selection is used
as the selection operator of the Genetic algorithm with
tournament size 4. The well-known Ordered Crossover
operator is applied to generate new offspring.
"""

"""
NOTA: Por sugerencia de AI se usa numba para hacer más
eficiente y sin ciclos los operadores.
"""

@numba.njit
def ordered_crossover_batch(parents1, parents2):
    n, m = parents1.shape
    offspring = np.full((n, m), -1, dtype=np.int32)

    cut_points = np.random.randint(0, m, size=(n, 2))
    
    for i in range(n):
        a, b = cut_points[i][0], cut_points[i][1]
        if a > b:
            a, b = b, a

        offspring[i, a:b+1] = parents1[i, a:b+1]

    for i in range(n):
        p2 = parents2[i]
        fill_mask = np.ones(m + 1, dtype=np.bool_)
        for k in range(m):
            if offspring[i, k] != -1:
                fill_mask[offspring[i, k]] = False
        fill_values = []
        for k in range(m):
            if fill_mask[p2[k]]:
                fill_values.append(p2[k])
        fill_idx = 0
        for k in range(m):
            if offspring[i, k] == -1:
                offspring[i, k] = fill_values[fill_idx]
                fill_idx += 1

    return offspring

"""
The Swap
mutation operator is applied under the mutation probability
0.9.
"""

@numba.njit
def mutation_swap_batch(population):
    n, m = population.shape

    if m < 2:
        return population

    swap_mask = np.random.rand(n) < MUTATION_PROB
    swap_idx = np.where(swap_mask)[0]

    for idx in swap_idx:
        i = np.random.randint(0, m)
        j = np.random.randint(0, m)
        while j == i:
            j = np.random.randint(0, m)
        temp = population[idx, i]
        population[idx, i] = population[idx, j]
        population[idx, j] = temp

    return population


# ---------------------------------------------------------------------------
# GA principal por clúster
# ---------------------------------------------------------------------------

def genetic_algorithm_tsp(cluster, depot_coord, customers):

    """
    In this study, total TSP tour distance of chromosome was used as the fitness value. The
    chromosomes with less TSP tour distance get higher chance to be selected to reproduce the next generation. 
    """
    
    # ---------------- Preparar datos locales ----------------
    coords = [depot_coord] + [customers[i]['coord'] for i in cluster]
    dmat = build_distance_matrix(coords)

    local_nodes = np.arange(1, len(coords))
    local2global = {loc: glob for loc, glob in zip(local_nodes, cluster)}

    # ---------------- Inicialización ----------------
    population = np.array([np.random.permutation(local_nodes) for _ in range(POPULATION_SIZE)], dtype=np.int32)
    best_dist = np.inf
    best_route = None
    gens_without_improve = 0

    """
    In the second phase of the Improved Heuristic
    algorithm, only the chosen best unique sets of clusters (≤
    5) are considered. The TSP of each cluster is separately
    solved by applying the standard GA and the total traveling
    distance of each unique best set of clusters is evaluated. 
    """

    for _ in range(MAX_GENERATIONS):

        """
        1. Evaluate the fitness value of each chromosome of the current generation
        """
        fitness = evaluate_population(population, dmat)

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_dist:
            best_dist = fitness[best_idx]
            best_route = population[best_idx].copy()
            gens_without_improve = 0
        else:
            gens_without_improve += 1
            if gens_without_improve >= NO_IMPROVEMENT_LIMIT:
                break

        """
        The best 10% chromosomes of the current population
        are directly sent to the next generation without any change
        (Elitism rate = 0.1).
        """
        elite_count = int(ELITISM_RATE * POPULATION_SIZE)
        elite_indices = np.argpartition(fitness, elite_count)[:elite_count]
        new_population = population[elite_indices]

        """
        Since four additional parameters are evaluated in the
        first phase and the GA is applied to solve TSP of more than
        one set of clusters in the second phase of the improved
        heuristic algorithm, more CPU time is consumed by the
        Figure 1: The pseudocode of the standard GA.
        improved heuristic algorithm than the original algorithm
        for solving CVRP. And also, the lower boundary of the
        solution generated by the Improved Heuristic algorithm is
        the optimal solution of the original heuristic algorithm. 
        """

        """
        the Tournament Selection is used
        as the selection operator of the Genetic algorithm with
        tournament size 4.
        """
        parent_indices = tournament_selection(fitness)
        parents = population[parent_indices]
        half = len(parents) // 2
        parents1 = parents[:half]
        parents2 = parents[half:]

        """
        The well-known Ordered Crossover
        operator is applied to generate new offspring
        """
        children = ordered_crossover_batch(parents1, parents2)

        """
        The Swap
        mutation operator is applied under the mutation probability
        0.9.
        """
        children = mutation_swap_batch(children)

        # Crear nueva generación
        new_population = np.vstack([new_population, children])
        population = new_population[:POPULATION_SIZE]

    # Convertir best_route a índices globales
    best_route_global = [local2global[idx] for idx in best_route.tolist()]
    return best_route_global, best_dist
