# ---------------------------------------------------------------------------
import random
import numpy as np

# ---------------------------------------------------------------------------
# Parámetros del algoritmo
# ---------------------------------------------------------------------------
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

POPULATION_SIZE = 100          # Debe ser múltiplo de 2 para OX
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

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def build_distance_matrix(coords):
    n = len(coords)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(coords[i], coords[j])
            dmat[i, j] = dmat[j, i] = d
    return dmat

def route_demand(route, customers):
    return sum(customers[i]['demand'] for i in route)

# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------

"""
Tournament Selection is used
as the selection operator of the Genetic algorithm with
tournament size 4. The well-known Ordered Crossover
operator is applied to generate new offspring.
"""

def tournament_selection(population, fitness):
    """Devuelve el índice del ganador del torneo."""
    contenders = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best = min(contenders, key=lambda idx: fitness[idx])
    return best


def ordered_crossover(parent1, parent2):
    """Ordered Crossover (OX) para cromosomas de permutación."""
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    # Copiar segmento de P1
    child[a:b + 1] = parent1[a:b + 1]
    # Rellenar con genes de P2 preservando orden
    fill_pos = (b + 1) % n
    p2_pos = (b + 1) % n
    while None in child:
        gene = parent2[p2_pos]
        if gene not in child:
            child[fill_pos] = gene
            fill_pos = (fill_pos + 1) % n
        p2_pos = (p2_pos + 1) % n
    return child

"""
The Swap
mutation operator is applied under the mutation probability
0.9.
"""

def mutation_swap(route):
    if len(route) < 2:
        return route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route = route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# ---------------------------------------------------------------------------
# GA principal por clúster
# ---------------------------------------------------------------------------

def route_distance(route_local, dmat):
    """
    In this study, total TSP tour distance of chromosome was used as the fitness value. The
    chromosomes with less TSP tour distance get higher chance to be selected to reproduce the next generation. 
    """
    dist = dmat[0, route_local[0]]
    for i in range(len(route_local) - 1):
        dist += dmat[route_local[i], route_local[i + 1]]
    dist += dmat[route_local[-1], 0]
    return dist


def genetic_algorithm_tsp(cluster, depot_coord, customers):
    # ---------------- Preparar datos locales ----------------
    coords = [depot_coord] + [customers[i]['coord'] for i in cluster]
    dmat = build_distance_matrix(coords)
    local_nodes = list(range(1, len(coords)))  # 1..n
    local2global = {loc: glob for loc, glob in zip(local_nodes, cluster)}

    # ---------------- Inicialización ----------------
    population = []
    for _ in range(POPULATION_SIZE):
        perm = local_nodes.copy()
        random.shuffle(perm)
        population.append(perm)

    """
    In the second phase of the Improved Heuristic
    algorithm, only the chosen best unique sets of clusters (≤
    5) are considered. The TSP of each cluster is separately
    solved by applying the standard GA and the total traveling
    distance of each unique best set of clusters is evaluated. 
    """

    best_route = None
    best_dist = float('inf')
    gens_without_improve = 0

    for gen in range(MAX_GENERATIONS):
        
        """
        1. Evaluate the fitness value of each chromosome of the current generation
        """
        fitness = [route_distance(r, dmat) for r in population]

        # Mejor global !!
        gen_best_idx = int(np.argmin(fitness))

        if fitness[gen_best_idx] < best_dist:
            best_dist = fitness[gen_best_idx]
            best_route = population[gen_best_idx].copy()
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
        elite_indices = np.argsort(fitness)[:elite_count]
        new_population = [population[idx].copy() for idx in elite_indices]

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
        while len(new_population) < POPULATION_SIZE:
            
            """
            the Tournament Selection is used
            as the selection operator of the Genetic algorithm with
            tournament size 4.
            """
            p1 = population[tournament_selection(population, fitness)]
            p2 = population[tournament_selection(population, fitness)]
            
            """
            The well-known Ordered Crossover
            operator is applied to generate new offspring
            """
            child = ordered_crossover(p1, p2)
            
            """
            The Swap
            mutation operator is applied under the mutation probability
            0.9.
            """
            if random.random() < MUTATION_PROB:
                child = mutation_swap(child)
            new_population.append(child)

        population = new_population

    best_route_global = [local2global[idx] for idx in best_route]
    return best_route_global, best_dist
