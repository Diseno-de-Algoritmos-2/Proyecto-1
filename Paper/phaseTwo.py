# # ---------------------------
import numpy as np
import random
from typing import List

# ---------------------------
# Parámetros (idénticos)
# ---------------------------
"""
In our
study, the number of generations was defined as the stopping criterion. The size of the initial population and
number of generations are set to 100 and 400 respectively by conducting a few trial executions on the
benchmarked problems. Note that according to the operations of the used genetic algorithm, the initial
population size should be divisible by four. 
"""
POPULATION_SIZE = 100
GENERATIONS = 400

# ---------------------------
# Funciones de Ayuda
# ---------------------------


def route_demand(route, customers):
    return sum(customers[i]["demand"] for i in route)


# ---------------------------
# Operadores de mutación
# ---------------------------

"""Flip: organize the values in between two selected positions in reverse order"""


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


# ---------------------------
# GA optimizado
# ---------------------------


def _build_distance_matrix(coords):
    """
    The fitness function is constructed according to the objective of the problem and used to evaluate the
    fitness value of each chromosome in the population. The quality of each chromosome is reflected by the fitness
    value. The fitness value is used by the genetic algorithm to select chromosomes for the reproduction of the new
    generations. In other words, the chance of a chromosome being selected for reproduction is directly depends on
    its fitness value.
    """
    diff = np.expand_dims(coords, 1) - np.expand_dims(coords, 0)
    return np.sqrt((diff**2).sum(axis=2))


def _route_distance_np(route_local, dmat):
    """
    In this study, total TSP tour distance of chromosome was used as the fitness value. The
    chromosomes with less TSP tour distance get higher chance to be selected to reproduce the next generation.
    """
    if not route_local:
        return 0.0
    # Construir secuencia [0, r0, r1, ..., rn, 0] con índices locales
    seq = np.array([0] + route_local + [0])
    return dmat[seq[:-1], seq[1:]].sum()


def _generate_initial_population(nodes_local):
    """
    The permutation encoding is best suited to represent the
    solutions of TSP. This encoding is generally used in ordering issues in which genetic operators are required to
    keep all the values in chromosome exactly once. In the genetic algorithm implementation of the novel heuristic,
    the permutation encoding was used.
    """
    pop = []
    for _ in range(POPULATION_SIZE):
        perm = nodes_local.copy()
        random.shuffle(perm)
        pop.append(perm)
    return pop


def genetic_algorithm_tsp(
    cluster,
    depot_coord,
    customers,
    vehicle_capacity,
):

    # ---------------- Preparar datos locales ----------------
    coords_local = [depot_coord] + [customers[i]["coord"] for i in cluster]
    dmat = _build_distance_matrix(np.array(coords_local))
    local_nodes = list(range(1, len(coords_local)))
    local_final_global = {loc: glob for loc, glob in zip(local_nodes, cluster)}

    # ---------------- Inicialización ----------------
    population = _generate_initial_population(local_nodes)
    best_route_local: List[int] | None = None
    best_distance = float("inf")

    """
    2.3.4 Iterative Steps to Create a New Generation of the Genetic Algorithm
    """
    for generation in range(GENERATIONS):

        """
        1. Evaluate the fitness value of each chromosome of the current generation
        """
        fitness_array = np.array([_route_distance_np(ind, dmat) for ind in population])

        best_index = fitness_array.argmin()
        best_distance_current = fitness_array[best_index]

        if best_distance_current < best_distance:
            best_distance = best_distance_current
            best_route_local = population[best_index].copy()

        """
        2. Rearrange the chromosome of the current generation in random order
        """
        indices = list(range(POPULATION_SIZE))
        random.shuffle(indices)
        shuffled_population = [population[i] for i in indices]
        shuffled_fitness = [fitness_array[i] for i in indices]

        """
        3. Obtaining four chromosomes at a stretch from the top of the rearranged generation, proceed following steps
        until complete the reproduction of new generation 
        """
        new_population = []
        for i in range(0, POPULATION_SIZE, 4):

            group = shuffled_population[i : i + 4]
            group_fit = shuffled_fitness[i : i + 4]

            """
            4. Find the best chromosome from these four chromosomes according to the fitness value 
            """
            best_in_group_idx = int(np.argmin(group_fit))
            best_chromosome = group[best_in_group_idx]

            """
            5. The best chromosome is added to the new generation without any change and used to reproduce another
            three offspring (Elitism rate = 0.25)
            """
            new_population.append(best_chromosome)

            """
            6. Randomly select two positions of the best chromosome and use following mutations to reproduce offspring
            a. Flip: organize the values in between two selected positions in reverse order
            b. Swap: exchange values of the two selected positions
            c. Shift: move values in between the selected positions by one position forward
            """
            child_flip = mutation_flip(best_chromosome)
            child_swap = mutation_swap(best_chromosome)
            child_shift = mutation_shift(best_chromosome)

            new_population.append(child_flip)
            new_population.append(child_swap)
            new_population.append(child_shift)

        population = new_population

    best_route_global = (
        [local_final_global[idx] for idx in best_route_local]
        if best_route_local
        else []
    )
    return best_route_global, best_distance
