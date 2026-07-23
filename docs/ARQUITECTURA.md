# Arquitectura del proyecto

## Objetivo de esta estructura

El repositorio separa el modelo de negocio de los motores de optimización. Las
reglas de datos, geometría, pallets, ECT, Procurement y costos no deben
depender de Gurobi, CPLEX, HiGHS ni de un script de campaña.

## Núcleo mantenido: `src/bonsai`

| Capa | Módulos | Responsabilidad |
|---|---|---|
| Datos | `data.py`, `models.py`, `config.py` | Cargar y normalizar los cuatro CSV; definir entidades y supuestos. |
| Factibilidad | `geometry.py`, `exact_candidates.py`, `decimal_candidates.py` | Fit, headspace, ECT, palletización y generación de geometrías candidatas. |
| Economía | `costs.py`, `bounds.py` | Tiers de Procurement por planta, flete y cotas. |
| Optimización | `optimizer.py`, `scip_optimizer.py` | CP-SAT y MIP mediante OR-Tools, sin reglas de negocio duplicadas. |
| Interfaz | `cli.py`, `reporting.py`, `solution_validation.py` | Comandos, archivos de salida y validación independiente. |

## Solvers

- **Ruta predeterminada y open source:** OR-Tools CP-SAT.
- **MIP open source a través de OR-Tools:** SCIP, cuando esté disponible en la
  distribución instalada de OR-Tools.
- **Benchmark opcional:** HiGHS (`.[highs]`).
- **Sólo experimental:** Gurobi y CPLEX. Sus scripts están nombrados
  explícitamente y no se importan desde el núcleo.

El CSV congelado en `baseline/` fue encontrado con Gurobi, pero su factibilidad
y costo se describen por el núcleo independiente del solver. El flujo decimal
también puede ejecutarse sin software comercial mediante
`scripts/run_decimal_scip.py`, que utiliza OR-Tools MPSolver/SCIP.

La ruta histórica de CLI conserva su escritor y validador de milímetros
enteros. La ruta decimal usa `decimal_io.py`, que comparte el evaluador de
costos y la factibilidad geométrica del núcleo. Consolidar ambos formatos tras
la limpieza de scripts será una refactorización posterior, sin cambiar reglas
de negocio ni volver a ejecutar la optimización.

## Scripts

`scripts/` conserva herramientas de auditoría, campañas y benchmarks usados
durante la competencia. No es una segunda capa de reglas de negocio: debe
consumir el núcleo de `src/bonsai`. Los nombres que contienen `gurobi` o
`cplex` requieren software comercial; los que contienen `highs` son
benchmarks opcionales; los demás son exploraciones reproducibles con el núcleo.

## Criterio para futuros cambios

1. Agregar una regla de negocio sólo en el núcleo.
2. Mantener un único evaluador independiente de costos.
3. Agregar un solver como adaptador, no como dependencia de datos o geometría.
4. Probar cada ruta contra el mismo validador y el mismo evaluador de costos.
5. Antes de eliminar un script experimental, reemplazar su evidencia por una
   prueba, un documento o un comando mantenido.
