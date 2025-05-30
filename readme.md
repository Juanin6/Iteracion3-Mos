# Proyecto III – Metaheurísticas para el CVRP  
Curso **ISIS-3302 · Metaheurísticas**

> **Instancia cubierta en esta entrega**  
> *Proyecto_Caso_Base* (15 clientes, 5 vehículos idénticos).

---

## 1 · Resumen del proyecto
El objetivo es comparar dos enfoques para el **Capacitated Vehicle Routing Problem (CVRP)**:

| Método | Propósito | Archivo principal |
| ------ | --------- | ----------------- |
| **Modelo exacto (Pyomo + Gurobi)** | Sirve de referencia óptima para el Caso Base | `src/pyomo_caso_base.py` |
| **Algoritmo Genético (GA)** | Metaheurística rápida que se escala a instancias grandes | `src/ga_case_base.py` |

---

## 2 · Metaheurístico seleccionado – Algoritmo Genético
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

## 3 · Estructura de carpetas

