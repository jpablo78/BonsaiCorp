# Baseline congelado — Bonsai Corp

Este commit conserva la versión reproducible de la solución de competencia
aceptada por Kaggle antes de refactorizar el código o incorporar los modos de
alta de productos.

## Artefacto competitivo

- Archivo: `baseline/asignacion_0_1mm.csv`
- Precisión externa: 0,1 mm
- Grosor global: 3,0 mm
- Costo total recalculado: USD 188.079.384,24
- Packaging: USD 26.921.184,24
- Flete: USD 161.158.200,00
- Pallets: 1.074.388
- Tipos de caja: 59
- Score Kaggle confirmado: 10,11098
- SHA-256: `a79aff251e1867a95efedd2548af5014df985f47cabfcd3161c7482a410b0d29`

La solución usa dimensiones decimales aceptadas por Kaggle. Por esa razón, el
artefacto competitivo se conserva junto con su runner decimal de Gurobi. El
CLI estándar todavía representa la variante de dimensiones enteras.

## Alcance del repositorio

Se incluyen código, pruebas, documentación y los artefactos finales mínimos.
Se excluyen deliberadamente los datos fuente, la consigna, el FAQ, instalaciones
locales, logs y las numerosas corridas experimentales.
