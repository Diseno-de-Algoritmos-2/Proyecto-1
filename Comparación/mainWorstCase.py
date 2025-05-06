import time


def run_worst_case_distance(customers):
    """
    Calculate the worst-case distance where every customer requires a route
    to and from the depot (the first customer).
    """

    start_time = time.time()

    depot_coord = customers[0]["coord"]
    worst_case_distance = 0
    for customer in customers[1:]:
        customer_coord = customer["coord"]
        # Calculate Euclidean distance between depot and customer
        distance_to_depot = (
            (depot_coord[0] - customer_coord[0]) ** 2
            + (depot_coord[1] - customer_coord[1]) ** 2
        ) ** 0.5
        worst_case_distance += 2 * distance_to_depot

    end_time = time.time()
    exceution_time = end_time - start_time
    return worst_case_distance, exceution_time
