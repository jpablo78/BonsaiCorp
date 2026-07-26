# Reproducción operativa de la solución final

Esta carpeta contiene el proceso mantenido que prepara los datos, genera
candidatos, optimiza, valida el CSV final y permite operar el catálogo después
de la entrega. Se excluyeron algoritmos alternativos, campañas históricas,
diagnósticos, CPLEX, HiGHS y búsqueda local experimental.

Incluye:

- asignacion_final.csv: resultado final de 427 SKU, precisión exterior de 0,1 mm;
- bonsai/: núcleo de datos, factibilidad, costos, CP-SAT, validación y ciclo de vida;
- scripts/: única ruta decimal opcional de Gurobi que produjo el resultado final;
- validar.py y generar_metricas.py: verificación independiente y tablas;
- requirements.txt y requirements-gurobi.txt: dependencias.

Integridad del CSV final: SHA-256
A79AFF251E1867A95EFEDD2548AF5014DF985F47CABFCD3161C7482A410B0D29.

## Requisitos

Python 3.11 o posterior. Copiar los cuatro CSV fuente en datos/, siguiendo
datos/README.md.

~~~powershell
python -m pip install -r requirements.txt
~~~

La optimización decimal final requiere una licencia activa de Gurobi:

~~~powershell
python -m pip install -r requirements-gurobi.txt
~~~

## Flujo principal de análisis

~~~powershell
python -m bonsai clean-data --data-dir datos --output-dir salida_limpieza
python -m bonsai baseline --data-dir salida_limpieza --output-dir salida_baseline
python -m bonsai optimize --data-dir salida_limpieza --thickness-mm 3.0 --time-limit-seconds 300 --output-dir salida_cp_sat
~~~

El flujo CP-SAT es la ruta abierta y mantenida para reconstruir el premodelado,
las restricciones y una solución optimizada de milímetros enteros.

## Ruta que llega al CSV final de 0,1 mm

El resultado entregado usa el mismo núcleo de restricciones, con la grilla
decimal de 0,1 mm y Gurobi como motor MIP. Puede partir de la solución CP-SAT
del paso anterior y volver a optimizarla en ese universo.

~~~powershell
python scripts/optimizar_decimal_gurobi.py --data-dir salida_limpieza --warm-start salida_cp_sat/asignacion_optima.csv --decimal-places 1 --time-limit-seconds 3600 --threads 6 --output-dir salida_decimal
~~~

La salida es salida_decimal/asignacion_decimal.csv. Para certificar rápidamente
la solución publicada, se puede usar asignacion_final.csv como warm start. En
ambos casos, la salida debe volver a pasar la validación independiente.

## Validación

~~~powershell
python validar.py --data-dir datos --solution asignacion_final.csv
~~~

El comando valida esquema, SKU, grosor global, dimensiones, ajuste dimensional,
headspace, ECT, palletización y costo. El resultado esperado es:

- 427 SKU;
- 59 tipos físicos;
- USD 188.079.384,24 de costo total;
- 1.074.388 pallets.

## Métricas y tablas

~~~powershell
python generar_metricas.py --data-dir datos --solution asignacion_final.csv --output-dir salida
~~~

Genera metricas_finales.json, tabla_costos.csv y
tabla_utilizacion_por_planta.csv. Los costos históricos se reconstruyen desde
operaciones_planta.csv; los de la solución se recalculan desde las reglas
comerciales y el CSV final.

## Operación del catálogo

Recalcular una solución existente con demanda anual proyectada:

~~~powershell
python -m bonsai recalcular-demanda --data-dir salida_limpieza --solution asignacion_final.csv --operaciones-override escenarios/operaciones_proyectadas_2027.csv --output-dir salida_recalculo
~~~

Dar de alta un SKU usando exclusivamente tipos de caja vigentes:

~~~powershell
python -m bonsai alta-incremental --data-dir salida_limpieza --solution asignacion_final.csv --nuevo-producto nuevo_producto.csv --operaciones-override escenarios/operaciones_proyectadas_2027.csv --output-dir salida_alta
~~~

Si no existe un tipo vigente factible, el JSON informa requiere_nuevo_diseno.
No se genera automáticamente una caja nueva ni se reasignan SKU existentes.

Revisión focalizada ante un lanzamiento: incorpora el nuevo SKU, identifica los
tipos vigentes compatibles y libera únicamente los SKU de ese vecindario. El
modelo puede reasignarlos entre los tipos físicos activos y recalcula los tiers
de Procurement sobre toda la red. Las demás asignaciones permanecen fijas.

~~~powershell
python -m bonsai revision-focalizada --data-dir salida_limpieza --solution asignacion_final.csv --nuevo-producto nuevo_producto.csv --operaciones-override escenarios/operaciones_proyectadas_2027.csv --max-hops 1 --time-limit-seconds 300 --num-search-workers 6 --output-dir salida_revision_focalizada
~~~

El valor predeterminado `--max-hops 1` libera los tipos directamente
compatibles con el nuevo SKU y los productos que los usan. Un valor mayor
amplía el vecindario por capas de compatibilidad. La salida incluye
`asignacion_revision_focalizada.csv` y `decision_revision_focalizada.json`, con
los SKU liberados, los tipos alcanzados, el costo de alta incremental, el costo
final y el estado del solucionador. Esta funcionalidad no genera geometrías
nuevas: si el nuevo SKU no cabe en ningún tipo vigente, informa
`requiere_nuevo_diseno`.
