import polars as pl
import random

# DEPOSITO
planta = pl.read_excel("data/clientes_con_demanda.xlsx", sheet_name="Planta")
vehicle_capacity = planta["CAPACIDAD CAMIÓN"].to_list()[0]
depot = {'coord': (planta["COORDENADA X"].to_list()[0], planta["COORDENADA Y"].to_list()[0]), 'demand': 0}

# CLIENTES
clientes = pl.read_excel("data/clientes_con_demanda.xlsx", sheet_name="Clientes")
len_clientes = len(clientes)
customers = [depot]

def get_clientes(size):

    select_clientes = random.sample(range(0, len(clientes)), size)

    for i in select_clientes:
        #print(clientes["ID CLIENTE"].to_list()[i])
        coord = (clientes["COORDENADA X"].to_list()[i], clientes["COORDENADA Y"].to_list()[i])
        demand = clientes["DEMANDA"].to_list()[i]
        customers.append({'coord': coord, 'demand': demand})
    
    return customers