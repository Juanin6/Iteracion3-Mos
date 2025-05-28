from pyomo.environ import *
import csv
from math import radians, cos, sin, asin, sqrt
import networkx as nx
import matplotlib.pyplot as plt

class Caso1:
    
    def __init__(self):
        self.model = ConcreteModel()
        self.idClientList=[]
        self.locationIdList=[]
        self.demands={}
        self.distances=None
        self.capacityList=[]
        self.rangeList=[]
        self.depotId=None
        self.numNodes=None
        
    
    def dataUpload(self):
        
        longitudeList=[]
        latitudeList=[]
        
        with open("Caso1/depots.csv", newline='') as archivo_csv:
            lector = csv.reader(archivo_csv)
            next(lector)
            for fila in lector:
                self.locationIdList.append(int(fila[0]))
                self.depotId=int(fila[0])
                longitudeList.append(fila[1])
                latitudeList.append(fila[2])
        
        with open("Caso1/clients.csv", newline='') as archivo_csv:
            lector = csv.reader(archivo_csv)
            next(lector)
            for fila in lector:
                self.idClientList.append(fila[0])
                self.locationIdList.append(int(fila[1]))
                self.demands[int(fila[1])]=int(fila[2])
                longitudeList.append(fila[3])
                latitudeList.append(fila[4])
                
        with open("Caso1/vehicles.csv", newline='') as archivo_csv:
            lector = csv.reader(archivo_csv)
            next(lector)
            for fila in lector:
                self.capacityList.append(int(fila[1]))
                self.rangeList.append(int(fila[2]))
        
        
                
        self.numNodes=len(self.locationIdList)
        self.preprocessData(longitudeList,latitudeList)
        
    
    def preprocessData(self,longitudeList,latitudeList):
        distance={}
        for i in range(len(longitudeList)):
            for j in range(len(longitudeList)):
                if i!=j:
                    distance[((self.locationIdList[i]),(self.locationIdList[j]))]=(self.calculateDistance(float(longitudeList[i]),float(latitudeList[i]),float(longitudeList[j]),float(latitudeList[j])))
                else:
                    distance[((self.locationIdList[i]),(self.locationIdList[j]))]=999
        self.distances=distance
    
    def calculateDistance(self,longitude1,latitude1,longitude2,latitude2):
        R=6371 # Radio de la Tierra en km
        lat1,lat2,lon1,lon2 = map(radians, [latitude1, latitude2, longitude1, longitude2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        distance =R* 2 * asin(sqrt(a))
        return round(distance,4)
    
    def createModel(self,maintingCost,consumptionCost):
        #Conjunto de nodos
        self.model.V=Set(initialize=self.locationIdList)
        print(self.locationIdList)
        # Conjunto de clientes
        self.model.v=Set(initialize=self.idClientList)
        
        # Conjunto de vehiculos
        self.model.K=Set(initialize=range(len(self.capacityList)))
        
        #Distancia entre nodos
        self.model.d=Param(self.model.V,self.model.V,initialize=self.distances)
        
        # Demanda de los vehiculos
        self.model.q=Param(self.model.V,initialize=self.demands)
        
        # Rango del vehiculo
        self.model.R =Param(self.model.K,initialize=self.rangeList)
        
        # Capacidad de los vehiculos
        self.model.Q =Param(self.model.K,initialize=self.capacityList)
        
        
        #Big M
        self.model.M=Param(initialize=1000)
        
        #Variable de decision
        self.model.x=Var(self.model.V,self.model.V,self.model.K,within=Binary,initialize=0)
        self.model.u=Var(self.model.V,within=NonNegativeReals )
        self.model.g=Var(self.model.V,self.model.K,within=NonNegativeReals)
        self.model.y=Var(self.model.V,self.model.K ,within=Binary)
        
        
        #Funcion Objetivo
        self.model.Obj =Objective(expr=sum((self.model.d[i,j]*self.model.x[i,j,k])*(consumptionCost+maintingCost) for i in self.model.V for j in self.model.V for k in self.model.K),sense=minimize)
        

        #Restriccion 1: Cada cliente es visitado al menos una vez
        def visitOnce(model,i):
            if(i!=self.depotId):
                return sum(model.x[i,j,k] for k in model.K for j in model.V) == 1
            else:
                return Constraint.Skip
        self.model.visitOnce=Constraint(self.model.V,rule=visitOnce)
        
        #Restriccion 2: Continuidad de la ruta (Lo que entra sale)
        def routeConservation(model,i,k):
            if(i!=self.depotId):
                return sum(model.x[j,i,k] for j in model.V) - sum(model.x[i,j,k] for j in model.V) == 0
            else:
                return Constraint.Skip
        self.model.routeConservation=Constraint(self.model.V,self.model.K,rule=routeConservation)
        
        #Restriccion 3 Carga acumulada del origen al origen: 
        def loadFromOriginToOrigin(model,k):
            return model.g[self.depotId,k] == 0
        self.model.loadFromOriginToOrigin=Constraint(self.model.K,rule=loadFromOriginToOrigin)
        
        #Restriccion 4: Carga acumulada de un vehiculo
        def loadAccumulated(model,i,j,k):
            if(i!=self.depotId and j!=self.depotId):
                return model.g[j, k] >= model.g[i, k] + model.y[j,k] - model.M * (1 - model.x[i, j, k])
            else:
                return Constraint.Skip
        self.model.loadAccumulated=Constraint(self.model.V,self.model.V,self.model.K,rule=loadAccumulated)
        #Restriccion 5: Capacidad de los vehiculos
        def maxCapacityVehicles(model,i,k):
            if(i!=self.depotId):
                return model.g[i,k] <= model.Q[k]
            else:
                return Constraint.Skip
        self.model.maxCapacityVehicles=Constraint(self.model.V,self.model.K,rule=maxCapacityVehicles)
        
        #Restriccion 6: Demanda de cliente debe ser satisfecha
        def demandSatisfaction(model,i):
            if(i!=self.depotId):
                return sum(model.g[i,k] for k in model.K) == model.q[i]
            else:
                return Constraint.Skip
        self.model.demandSatisfaction=Constraint(self.model.V,rule=demandSatisfaction)
        #Restriccion 7: Rango de los vehiculos
        def vehiculeRange(model,k):
            return sum(model.d[i,j]*model.x[i,j,k] for i in model.V for j in model.V) <= model.R[k]
        self.model.vehiculeRange=Constraint(self.model.K,rule=vehiculeRange)
        
        #Restriccion 8: No puede self loops
        def noSelfLoops(Model,i,k):
            return Model.x[i,i,k] == 0
        self.model.noSelfLoops=Constraint(self.model.V,self.model.K,rule=noSelfLoops)
        
        #Restriccion 9: Eliminacion de subtours
        def subtour_elimination(model, i, j,k):
            if i != self.depotId and j != self.depotId and i != j:
                return model.u[i] - model.u[j] + (self.numNodes - 1) * model.x[i,j,k]  <= self.numNodes - 2
            else:
                return Constraint.Skip
        self.model.subtour_elimination = Constraint(self.model.V, self.model.V,self.model.K, rule=subtour_elimination)
        #Restriccion 10: Deben salir del deposito el numero de vehiculos
        def depotExitVehicles(model,k):
            return sum(model.x[self.depotId,j,k] for j in model.V) == 1
        self.model.depotExitVehicles=Constraint(self.model.K,rule=depotExitVehicles)
        #Resitriccion 11: Deben entrar al deposito el numero de vehiculos
        def depotEnterVehicles(model,k):
            return sum(model.x[i,self.depotId,k] for i in model.V) == 1
        self.model.depotEnterVehicles=Constraint(self.model.K,rule=depotEnterVehicles)
    
        
    def solve(self):
        solver = SolverFactory('cplex', tee=True)
        solver.options['msg'] = True
        solver.options['output'] = True
        solver.solve(self.model)
        """
        
        results = 
        print("Termination condition:", results.solver.termination_condition) 
        if results.solver.status == SolverStatus.ok:
            print("Solver encontró una solución exitosa.")
        else:
            print(f"Estado del solver: {results.solver.status}")
        
        # Verificar valores de las variables después de la resolución
        """
        for i in self.model.V:
            for j in self.model.V:
                for k in self.model.K:
                    if (self.model.x[i, j, k].value >= 0.5):
                        print(f"Vehículo {k} va de {i} a {j}")
    def show(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.model.V)
        color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange', 5: 'pink', 6: 'brown', 7: 'cyan'}
        edges_by_f = {}
        for i in self.model.V:
            for j in self.model.V:
                for f in self.model.K:
                    if value(self.model.x[i, j, f]) == 1:
                        if f not in edges_by_f:
                            edges_by_f[f] = []
                        edges_by_f[f].append((i, j))
        # Dibujar el grafo
        pos = nx.spring_layout(G,k=5, iterations=200)  # Generar distribución del grafo

        # Dibujar nodos
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=200)

        # Dibujar aristas por grupo de f
        for f, edges in edges_by_f.items():
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color_map.get(f, 'black'), width=2,arrows=True,arrowsize=10)

        # Dibujar etiquetas
        nx.draw_networkx_labels(G, pos)

        plt.show()
    



c_m =0.01
c_c=0.10
caso1 = Caso1()
caso1.dataUpload()
caso1.createModel(maintingCost=c_m,consumptionCost=c_c)
caso1.solve()
