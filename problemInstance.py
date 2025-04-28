import random
random.seed(20)

# Capacidad de vehiculos
vehicle_capacity = 50

# Clientes
num_customers = 200  # Sin el deposito
depot = {'coord': (50, 50), 'demand': 0}
customers = [depot]

for i in range(1, num_customers + 1):
    coord = (random.uniform(0, 100), random.uniform(0, 100))
    demand = random.randint(1, 20)
    customers.append({'coord': coord, 'demand': demand})