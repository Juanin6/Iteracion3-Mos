from src.pyomo_case_base import solve_pyomo_base
from src.ga_case_base    import run_ga_base

print("*****  PYOMO (exacto) *****")
py = solve_pyomo_base()
print(f"   Objetivo: {py['obj']:.2f}  |  Tiempo: {py['time']:.2f}s  |  Mem: {py['mem']:.1f} MB")

print("\n\n*****  GA (metaheurístico) *****")
ga = run_ga_base(seed=0)
print(f"   Objetivo: {py['obj']:.2f}  |  Tiempo: {py['time']:.2f}s  |  Mem: {py['mem']:.1f} MB")


print("\n\n*****  PYOMO (exacto) *****")
print(f"   Objetivo: {py['obj']:.2f}  |  Tiempo: {py['time']:.2f}s  |  Mem: {py['mem']:.1f} MB")

print("\n*****  GA (metaheurístico) *****")
print(f"   Objetivo: {ga['obj']:.2f}  |  Tiempo: {ga['time']:.2f}s  |  Mem: {ga['mem']:.1f} MB")

gap = 100 * (ga['obj'] - py['obj']) / py['obj']
print(f"\n GAP GA vs Óptimo: {gap:.2f}%")
