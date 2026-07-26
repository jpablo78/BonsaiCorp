# Optimización de packaging — Bonsai Corp

Solución reproducible para consolidar tipos de caja, minimizando packaging y flete bajo las reglas de la consigna.

## Supuestos fijados

- La demanda proviene de `operaciones_planta.csv`; los descuentos se recalculan por tipo de caja y planta.
- El escenario base cobra USD 150 por pallet. `FreightPolicy` permite incorporar en el futuro porcentajes o cantidades extra-región a USD 500.
- La regla experimental alineada con FAQ #10 permite ajustar cada eje ±10%, siempre que el volumen interno alcance al producto. El headspace positivo por eje (`interno nuevo − interno actual`) debe respetar el máximo porcentual según grosor o 40 mm; una reducción permitida no se interpreta como headspace negativo.
- El perímetro para ECT se calcula con las dimensiones exteriores.
- Las dimensiones exteriores entregadas son enteras en milímetros.
- `N_tipos` se informa, pero no es un desempate de la función objetivo.

## Instalación y uso

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m bonsai validate-data --data-dir .
.\.venv\Scripts\python.exe -m bonsai clean-data --data-dir . --output-dir output/cleaned_data
.\.venv\Scripts\python.exe -m bonsai baseline --data-dir . --output-dir output
.\.venv\Scripts\python.exe -m bonsai optimize --data-dir . --output-dir output --time-limit-seconds 300
.\.venv\Scripts\python.exe -m bonsai validate-solution output/asignacion_optima.csv --data-dir .
```

La instalación predeterminada depende solamente de OR-Tools. Para ejecutar la
suite de pruebas: `pip install -e ".[dev]"`. Los runners de
HiGHS, CPLEX y Gurobi son opcionales y no forman parte de la ruta soportada de
producción. Para instalar HiGHS: `pip install -e ".[highs]"`. Los runners
comerciales requieren además su paquete y licencia local; nunca se necesitan
para consultar, validar o reproducir el cálculo de costo de la solución base.

Los CSV fuente y la consigna no se publican en este repositorio. Para ejecutar
el modelo, deben colocarse en un directorio local y pasarse mediante
`--data-dir`.

La optimización resuelve un modelo CP-SAT por cada grosor permitido (3,0; 4,5; 5,0 mm) y conserva la mejor solución factible.

## Arquitectura y solvers

El núcleo mantenido está bajo `src/bonsai`: preparación de datos, geometría,
generación de candidatos, costos, validación y optimización CP-SAT. El MIP de
SCIP se ejecuta a través de OR-Tools. Los scripts de campañas y comparaciones
con otros solvers se conservan como evidencia experimental, pero no son una
dependencia del CLI principal. Ver [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md).

`clean-data` no modifica los archivos fuente: exporta una copia normalizada. En particular, estandariza `caja_grosor_mm` (por ejemplo, `2,5`, `2.5 mm` y `2.50` pasan a `2.5`) y conserva explícitamente los errores/inconsistencias históricas que no deben imputarse de forma silenciosa.

## Optimizador exacto y proteccion de incumbente

La estrategia predeterminada enumera toda la grilla entera documentada,
deduplica disenos con la misma capacidad y compatibilidad, y elimina candidatos
dominados antes de resolver CP-SAT.

Para la fase de 3 mm que mantiene a cada SKU en su capacidad individual maxima
y optimiza consolidacion y descuentos:

```powershell
.\.venv\Scripts\python.exe -m bonsai optimize `
  --data-dir output\cleaned_data `
  --output-dir output_exact_3mm `
  --thickness-mm 3.0 `
  --warm-start output_capacity_pairs_3mm\asignacion_optima.csv `
  --candidate-strategy exact `
  --preserve-individual-max-capacity `
  --num-search-workers 6 `
  --time-limit-seconds 300
```

`--warm-start` se trata como una incumbente protegida: se agrega como cota dura
y se devuelve como respaldo si el solucionador no encuentra una mejora estricta. El
CSV final se escribe de forma atomica y un archivo existente nunca se reemplaza
por otro de costo igual o mayor. `--restarts N` reparte el presupuesto de tiempo
entre semillas secuenciales y encadena siempre la mejor incumbente.

El JSON informa objetivo, mejor cota, gap dentro del universo de candidatos,
tiempo, ramas y conflictos. Ese gap no prueba optimalidad respecto de reglas o
universos diferentes a los de la corrida.

`--max-extra-pallets B` explora el frente costo de flete/descuentos permitiendo
como maximo `B` pallets por encima del minimo global SKU por SKU. Es alternativo
a `--preserve-individual-max-capacity`. Para calcular cotas rigurosas y descartar
espesores sin gastar tiempo de solucionador:

```powershell
.\.venv\Scripts\python.exe -m bonsai lower-bounds `
  --data-dir output\cleaned_data `
  --incumbent output_exact_pareto_3mm\asignacion_optima.csv `
  --output-path output_exact_analysis\thickness_lower_bounds.json
```

Cada corrida conserva `asignacion_ultima_corrida.csv`; `asignacion_optima.csv`
funciona como best-so-far y no se reemplaza por una salida de costo igual o
mayor. Esta proteccion supone un solo proceso escritor por directorio de salida.

## Ciclo de vida del catálogo

Estas dos extensiones no reoptimizan el portafolio completo. Permiten evaluar
un catálogo ya validado cuando cambia la demanda y atender un SKU nuevo sin
alterar las asignaciones vigentes.

### Recálculo con nueva demanda

El CSV pasado con `--operaciones-override` debe conservar exactamente el
esquema de `operaciones_planta.csv`, incluir los mismos 427 SKU y contener los
volúmenes anuales proyectados por planta. La asignación se mantiene fija; se
recalculan pallets, tiers de Procurement, cartón, flete y costo total.

```powershell
.\.venv\Scripts\python.exe -m bonsai recalcular-demanda `
  --data-dir output\cleaned_data `
  --solution baseline\asignacion_0_1mm.csv `
  --operaciones-override escenarios\operaciones_proyectadas_2027.csv `
  --output-dir output\recalculo_demanda_2027
```

El resultado queda en `resultado_recalculo.json`; informa el costo con demanda
actual, el costo proyectado y sus diferencias. No se genera un archivo de
entrega porque la geometría y las asignaciones no cambian.

### Alta incremental de un nuevo producto

Este modo mantiene fijos los SKU existentes, detecta los tipos físicos activos
en la solución y prueba el nuevo SKU contra cada uno. Elige el tipo factible de
menor costo incremental, recalculando los tiers afectados. Si no existe ninguna
caja vigente factible, devuelve `requiere_nuevo_diseno`; no diseña esa caja ni
asume costos de homologación.

```powershell
.\.venv\Scripts\python.exe -m bonsai alta-incremental `
  --data-dir output\cleaned_data `
  --solution baseline\asignacion_0_1mm.csv `
  --nuevo-producto examples\nuevo_producto_ejemplo.csv `
  --operaciones-override escenarios\operaciones_proyectadas_2027.csv `
  --output-dir output\alta_BR0428
```

`--operaciones-override` es opcional en este segundo comando: si se omite,
utiliza la demanda disponible en `--data-dir`. El CSV de nuevo producto tiene
una sola fila y exactamente estas columnas:

```text
codigo_producto
referencia_interna_largo_mm
referencia_interna_ancho_mm
referencia_interna_alto_mm
peso_neto_caja_kg
volumen_producto_planta_buenos_aires
volumen_producto_planta_curitiba
volumen_producto_planta_santiago
volumen_producto_planta_monterrey
volumen_producto_planta_bakersfield
```

Las tres dimensiones de referencia son obligatorias porque la tolerancia de
±10% se verifica por eje. El comando genera `decision_alta_incremental.json` y,
si encuentra una caja activa factible, `asignacion_incremental.csv`. Este último
es un artefacto operativo de 428 filas, no un archivo para submit. Hay un
ejemplo en `examples/nuevo_producto_ejemplo.csv`.

## Large-neighborhood search por tiers

El ejecutor de LNS reconstruye objetivos de descuento desde la incumbente, libera
grupos completos de origen y prueba stars, componentes conectados y uniones de
ambos. Cada mejora se vuelve a leer con el validador de submissions antes de
ser aceptada y se conserva como `incumbent_NNNN.csv`:

```powershell
.\.venv\Scripts\python.exe scripts\run_tier_lns.py `
  --data-dir output\cleaned_data `
  --warm-start output_exact_pareto_3mm\asignacion_optima.csv `
  --output-dir output_tier_lns `
  --target-total-usd 188079000 `
  --time-per-neighborhood 60 `
  --num-search-workers 6 `
  --max-extra-pallets 500 `
  --rounds 4
```

El modo predeterminado `--target-mode stop` admite mejoras intermedias y se
detiene al alcanzar el objetivo. `--target-mode hard` convierte cada subproblema
en factibilidad pura contra ese costo. La grilla exacta se enumera una sola vez
y se reutiliza en todos los vecindarios de la corrida.

## Salidas

- `output/asignacion_optima.csv`: formato exacto requerido por Kaggle.
- `output/resumen_optimizacion.json`: costos, pallets, utilización y tipos por grosor evaluado.
- `output/baseline_*.json`: escenarios sin consolidación para cada grosor global.
