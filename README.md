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

La optimización resuelve un modelo CP-SAT por cada grosor permitido (3,0; 4,5; 5,0 mm) y conserva la mejor solución factible.

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
y se devuelve como fallback si el solver no encuentra una mejora estricta. El
CSV final se escribe de forma atomica y un archivo existente nunca se reemplaza
por otro de costo igual o mayor. `--restarts N` reparte el presupuesto de tiempo
entre semillas secuenciales y encadena siempre la mejor incumbente.

El JSON informa objetivo, mejor cota, gap dentro del universo de candidatos,
tiempo, ramas y conflictos. Ese gap no prueba optimalidad respecto de reglas o
universos diferentes a los de la corrida.

`--max-extra-pallets B` explora el frente costo de flete/descuentos permitiendo
como maximo `B` pallets por encima del minimo global SKU por SKU. Es alternativo
a `--preserve-individual-max-capacity`. Para calcular cotas rigurosas y descartar
espesores sin gastar tiempo de solver:

```powershell
.\.venv\Scripts\python.exe -m bonsai lower-bounds `
  --data-dir output\cleaned_data `
  --incumbent output_exact_pareto_3mm\asignacion_optima.csv `
  --output-path output_exact_analysis\thickness_lower_bounds.json
```

Cada corrida conserva `asignacion_ultima_corrida.csv`; `asignacion_optima.csv`
funciona como best-so-far y no se reemplaza por una salida de costo igual o
mayor. Esta proteccion supone un solo proceso escritor por directorio de salida.

## Large-neighborhood search por tiers

El runner de LNS reconstruye targets de descuento desde la incumbente, libera
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
detiene al alcanzar el target. `--target-mode hard` convierte cada subproblema
en factibilidad pura contra ese costo. La grilla exacta se enumera una sola vez
y se reutiliza en todos los vecindarios de la corrida.

## Salidas

- `output/asignacion_optima.csv`: formato exacto requerido por Kaggle.
- `output/resumen_optimizacion.json`: costos, pallets, utilización y tipos por grosor evaluado.
- `output/baseline_*.json`: escenarios sin consolidación para cada grosor global.
