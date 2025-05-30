import random
import numpy as np
import csv
# Datos del problema CVRP
NUM_CUSTOMERS = 50
VEHICLE_CAPACITY = 160
NUM_VEHICLES = 5
DEPOTID = 0
c_m =0.01
c_c=0.10
# Parámetros del algoritmo genético
POPULATION_SIZE = 50
MAX_GENERATIONS = 600
CROSSOVER_PROB = 0.85
MUTATION_PROB = 0.1
longitudeList=[]
latitudeList=[]
locationIdList =[]
demands = {}

def haversine_distance_matrix(latitudes, longitudes):
    # Convertir a radianes
    lat = np.radians(latitudes)
    lon = np.radians(longitudes)

    # Expande dimensiones para vectorizar
    lat1 = lat[:, None]
    lat2 = lat[None, :]
    lon1 = lon[:, None]
    lon2 = lon[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371  # Radio de la Tierra en km
    distance_matrix = R * c
    return distance_matrix


# Generar una matriz de distancias aleatoria simétrica






def upload_data():
    global DEPOTID ,demands,VEHICLE_CAPACITY,locationIdList
    
    
    vehiclesIdList=[]
    with open("Caso1/depots.csv", newline='') as archivo_csv:
        lector = csv.reader(archivo_csv)
        next(lector)
        for fila in lector:
            locationIdList.append(int(fila[0]))
            DEPOTID=int(fila[0])
            longitudeList.append(float(fila[1]))
            latitudeList.append(float(fila[2]))
    with open("Caso1/clients15.csv", newline='') as archivo_csv:
            lector = csv.reader(archivo_csv)
            next(lector)
            for fila in lector:
                #self.idClientList.append(fila[0])
                locationIdList.append(int(fila[1]))
                
                demands[int(fila[1])]=int(fila[2])
                longitudeList.append(float(fila[3]))
                latitudeList.append(float(fila[4]))
    with open("Caso1/vehicles.csv", newline='') as archivo_csv:
            lector = csv.reader(archivo_csv)
            next(lector)
            for fila in lector:
                vehiclesIdList.append(int(fila[0]))
                VEHICLE_CAPACITY= int(fila[1])
    

upload_data()
id_to_index = {id_: idx for idx, id_ in enumerate(locationIdList)}
distance_matrix = haversine_distance_matrix(latitudeList, longitudeList)

def create_individual():
    """Crea un individuo válido respetando la capacidad y número de vehículos."""
    customers = locationIdList[:]
    customers.remove(DEPOTID)
    
    
    random.shuffle(customers)
    
    individual = []
    
    vehicle_count = 0

    while customers and vehicle_count < NUM_VEHICLES:
        route = [DEPOTID]
        load = 0
        i = 0
        while i < len(customers):
            if load + demands[customers[i]] <= VEHICLE_CAPACITY:
                route.append(customers[i])
                load += demands[customers[i]]
                customers.pop(i)
            else:
                i += 1
        route.append(DEPOTID)
        if(len(route)>2):
            individual.append(route)
        vehicle_count += 1

    # Si quedaron clientes sin asignar, se pueden considerar en la función de fitness
    # como penalización por individuo no factible
    if customers:
        individual.append(customers)  

    return individual




def calculate_fitness(individual):
    """Calcula el fitness de un individuo: menor distancia = mejor fitness.
    Penaliza si hay clientes no asignados."""
    total_distance = 0
    penalty = 0
    

    for route in individual:
       
        # Si la ruta no comienza ni termina en el depósito, la tratamos como clientes no asignados
       
        if route[0] != DEPOTID or route[-1] != DEPOTID:
            # Penalizamos por cada cliente no asignado
            penalty += len(route) * 300  
            continue
        for i in range(len(route) - 1):
            
            from_idx = id_to_index[route[i]]
            to_idx = id_to_index[route[i + 1]]
            total_distance += distance_matrix[from_idx][to_idx]


    # Fitness inversamente proporcional a la distancia total + penalizaciones
    return 1 / (total_distance + penalty + 1e-6)  



def select_population(population, fitnesses):
    total_fit = sum(fitnesses)
    probs = [f / total_fit for f in fitnesses]
    return random.choices(population, weights=probs, k=len(population))


def mutate(individual):
    """Mutación de intercambio de 2 puntos."""
    route_idx = random.randint(0, len(individual) - 1)
    route = individual[route_idx][1:-1]  # No cuenta el deposito
    if len(route) >= 2:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        individual[route_idx] = [DEPOTID] + route + [DEPOTID]
    return individual


def crossover(parent1, parent2):
    """Cruce fijo por rutas usando máscara binaria."""
    child = []
    taken = set()

    len1 = len(parent1)
    len2 = len(parent2)
    min_len = min(len1, len2)

    # Máscara binaria aleatoria de tamaño mínimo común
    mode = [random.randint(0, 1) for _ in range(min_len)]

    # Paso 1: Copiar rutas según máscara
    for idx in range(min_len):
        source = parent1 if mode[idx] else parent2
        route = source[idx]
        route_clean = [c for c in route if c != DEPOTID]
        if all(c not in taken for c in route_clean):
            child.append(route)
            taken.update(route_clean)

    # Paso 2: Agregar rutas restantes (clientes no repetidos)
    for source in [parent1, parent2]:
        for route in source[min_len:]:
            route_clean = [c for c in route if c != DEPOTID and c not in taken]
            if route_clean:
                route_clean = [c for c in route_clean if c not in taken]
                if route_clean:
                    new_route = [DEPOTID] + route_clean + [DEPOTID]
                    child.append(new_route)
                    taken.update(route_clean)
    # Paso 3: Completar con clientes faltantes
    missing = [c for c in demands.keys() if c not in taken]
    if missing:
        new_route = [DEPOTID]
        load = 0
        for cust in missing:
            if load + demands[cust] > VEHICLE_CAPACITY:
                new_route.append(DEPOTID)
                child.append(new_route)
                new_route = [DEPOTID]
                load = 0
            new_route.append(cust)
            load += demands[cust]
        new_route.append(DEPOTID)
        child.append(new_route)

    
    if not child:  
        child = [create_individual()[0]]  # al menos 1 ruta válida

        

    return child



import time, tracemalloc

def run_ga_base(seed=0,
                pop_size=POPULATION_SIZE,
                gens=MAX_GENERATIONS,
                pc=CROSSOVER_PROB,
                pm=MUTATION_PROB):
    """
    Ejecuta el GA sobre la instancia Caso1 y devuelve:
        { 'obj': distancia*(c_c+c_m),
          'solution': mejor individuo,
          'hist': lista mejor_dist por generación,
          'time': segundos,
          'mem': MB pico }
    """
    random.seed(seed); np.random.seed(seed)

    # Si ya se cargaron datos antes, NO los vuelvas a leer.
    # (upload_data() se ejecutó en el import y fijó los globals)
    # -----------------------------------------------------------------
    population = [create_individual() for _ in range(pop_size)]
    best_hist = []

    tracemalloc.start()
    t0 = time.perf_counter()

    for _ in range(gens):
        fitnesses = [calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(fitnesses)
        best_hist.append(1/fitnesses[best_idx])          # distancia bruta
        new_pop = [population[best_idx]]                 # elitismo

        while len(new_pop) < pop_size:
            parents = select_population(population, fitnesses)
            child = crossover(parents[0], parents[1]) if random.random() < pc else parents[0]
            if random.random() < pm:
                child = mutate(child)
            new_pop.append(child)
        population = new_pop

    runtime  = time.perf_counter() - t0
    peak_mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    best_sol   = max(population, key=calculate_fitness)
    best_dist  = 1 / calculate_fitness(best_sol)         # km totales
    best_cost  = best_dist * (c_c + c_m)                 # $ / coste “FO”

    return {"obj": best_cost,
            "solution": best_sol,
            "hist": best_hist,
            "time": runtime,
            "mem": peak_mem}


if __name__ == "__main__":
    # Repetimos el ciclo de evolución usando la nueva función crossover_fixed
    population = [create_individual() for _ in range(POPULATION_SIZE)]


    for gen in range(MAX_GENERATIONS):
        fitnesses = [calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(fitnesses)
        best = population[best_idx]
        new_population = [best]  # elitismo

        while len(new_population) < POPULATION_SIZE:
            parents = select_population(population, fitnesses)
            if random.random() < CROSSOVER_PROB:
                child = crossover(parents[0], parents[1])
            else:
                child = parents[0]
            if random.random() < MUTATION_PROB:
                child = mutate(child)
            new_population.append(child)

        population = new_population

    # Mejor solución encontrada
    best_solution = max(population, key=calculate_fitness)
    best_distance = 1 / calculate_fitness(best_solution)

    print(best_solution, best_distance*(c_c+c_m))

