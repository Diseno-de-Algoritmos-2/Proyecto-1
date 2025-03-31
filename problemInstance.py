import random

# Set the random seed for reproducibility
random.seed(3)

# Define the vehicle capacity
vehicle_capacity = 30

# Define the customers
num_customers = 20  # excluding depot
depot = {'coord': (50, 50), 'demand': 0}
customers = [depot]
for i in range(1, num_customers + 1):
    coord = (random.uniform(0, 100), random.uniform(0, 100))
    demand = random.randint(1, 10)
    customers.append({'coord': coord, 'demand': demand})