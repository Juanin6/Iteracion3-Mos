# Proyecto III – Metaheurísticas para el CVRP  
Curso **ISIS-3302 · Metaheurísticas**

> **Instancia cubierta en esta entrega**  
> *Proyecto_Caso_Base* (15 clientes, 5 vehículos idénticos).

---

## 1 · Modo de uso
Para comparar las instancias de pyomo y la del algoritmo genetico hay que correr el archivo llamado `comparacion_base.py`, este correra los archivos `implementacion_caso_base_pyomo.py` para ejecutar el modelo de pyomo y `src/ga_case_base.py` para correr el algoritmo genetico.

Por defecto se corre el archivo con 15 clientes. Si se quiere cambiar esto por otros archivos, hay que cambiar el nombre del archivo "clients.csv" directamente en los archivos `comparacion_base.py` y `implementacion_caso_base_pyomo.py` 

---

## 2 · Resumen del proyecto
El objetivo es comparar dos enfoques para el **Capacitated Vehicle Routing Problem (CVRP)**:

| Método | Propósito | Archivo principal |
| ------ | --------- | ----------------- |
| **Modelo exacto (Pyomo + Gurobi)** | Sirve de referencia óptima para el Caso Base | `src/pyomo_caso_base.py` y `implementacion_caso_base_pyomo.py` |
| **Algoritmo Genético (GA)** | Metaheurística rápida que se escala a instancias grandes | `src/ga_case_base.py` |

---

## 3 · Metaheurístico seleccionado – Algoritmo Genético
* **Codificación:** cada individuo es una lista de rutas  
  `[[DEPOT, c₁, c₂, …, DEPOT], …, [DEPOT, cᵣ, DEPOT]]`.
* **Inicialización:** aleatoria, respetando la capacidad \(Q = 160\).
* **Selección:** ruleta ponderada por *fitness*  
  \(f = 1 / (\text{distancia}+penalizaciones)\).
* **Cruce:** máscara binaria por rutas (garantiza no repetir clientes).
* **Mutación:** intercambio 2-swap dentro de una ruta.
* **Reparación:** si quedan clientes sin asignar se crea una ruta extra.
* **Parámetros por defecto:**  
  población 50 · generaciones 600 · \(P_c = 0.85\) · \(P_m = 0.10\).

---

## 4 · Estructura de carpetas
.
├── data/
│ └── data/ # depots.csv, clients.csv, vehicles.csv, clients15.csv, clients19.csv, clients24.csv,
├── src/
│ ├── caso1.py # imlementacion del proyecto2, con graficas
│ ├── ga_case_base.py # implementacion del GA
│ ├── implementacion_caso_base_pyomo.py # implementación del modelo de pyomo completo
│ └── pyomo_caso_base.py # wrapper del modelo exacto
├── results/
│ └── figs/
│   ├── client15.png     # Imagen resultados con 15 clientes
│   ├── client19.png     # Imagen resultados con 19 clientes
│   └── client24.png     # Imagen resultados con 124 clientes
│ └── verificacion/
│   └── verificacion_caso1  # Archivo de verificacion del caso base
│
├── compare_base.py # Archivo a correr (exacto vs GA)
├── graficas.ipynb  # Donde se crean las graficas
└── README.md # este documento




---

## 4 · Requisitos de software

- **Python** ≥ 3.8  
- **Pyomo** ≥ 6.5  
- **Gurobi** 10.0 (licencia académica) o **CBC** si no hay Gurobi  
- **Bibliotecas**:  
  ```bash
  pip install numpy pandas matplotlib psutil
  pip install pyomo
  # si usas CBC:
  pip install cylp
