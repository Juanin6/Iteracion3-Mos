from src.implementacion_caso_base_pyomo import Caso1          
import time, tracemalloc
from pyomo.environ import value

def solve_pyomo_base(mc=0, cc=1):
    """
    Ejecuta Pyomo SOLO para Proyecto_Caso_Base (clients15) y
    devuelve obj, runtime, memoria y las aristas activas.
    mc = costo de mantenimiento por km
    cc = costo de consumo por km
    """
    c_m =0.01
    c_c=0.10
    caso = Caso1()
    caso.dataUpload()                   # carga depots, clients15, vehicles
    caso.createModel(c_m, c_c)            # construye restricciones y objetivo

    tracemalloc.start()
    t0 = time.perf_counter()
    caso.solve()                        # llama a gurobi
    runtime = time.perf_counter() - t0
    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    obj = value(caso.model.Obj)

    # Recupera aristas activas (para verificación) --------------------------
    edges = []
    for i in caso.model.V:
        for j in caso.model.V:
            for k in caso.model.K:
                if value(caso.model.x[i, j, k]) > 0.5:
                    edges.append((int(i), int(j), int(k)))
    return {"obj": obj, "time": runtime, "mem": mem, "edges": edges}
