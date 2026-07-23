# Scripts de soporte y experimentación

El comando mantenido para el flujo estándar es `python -m bonsai ...`. Los
scripts de esta carpeta documentan auditorías, campañas y comparaciones que se
utilizaron durante la competencia.

- `audit_*` y `create_best_solution_document.py`: auditoría y documentación.
- `run_*lns*`, `run_*pool*`, `run_*branch*` y campañas: búsqueda experimental
  construida sobre el núcleo.
- `run_highs_benchmark.py`: benchmark open source opcional.
- `run_decimal_scip.py`: ruta decimal open source con OR-Tools/SCIP.
- `build_headspace_ab_probe.py`: genera una submission controlada que cambia
  sólo BR0004 para contrastar la interpretación de headspace en Kaggle.
- `run_gurobi_*` y `run_cplex_*`: benchmarks o diagnósticos comerciales;
  requieren instalación y licencia local, y no son dependencias del proyecto.

No agregar reglas de factibilidad, costos o lectura de datos nuevas sólo en un
script. Esas reglas deben vivir en `src/bonsai` y quedar cubiertas por pruebas.
