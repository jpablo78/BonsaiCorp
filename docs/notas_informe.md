# Notas para el informe final

## Valor económico de la restricción de grosor uniforme

La consigna exige utilizar un único grosor de cartón en todo el catálogo. Para
cuantificar el costo económico de esa política se resolvió, únicamente como
análisis de sensibilidad, una variante que permite elegir el grosor por tipo de
caja entre 3,0, 4,5 y 5,0 mm. Todas las demás restricciones operativas,
geométricas, de headspace, ECT, palletización, demanda y Procurement se
mantuvieron sin cambios.

El MIP diagnóstico fue resuelto por Gurobi hasta optimalidad con brecha 0 %. El
resultado fue:

- escenario oficial, grosor global de 3 mm: USD 188.092.808,10;
- escenario diagnóstico con grosor por tipo: USD 188.061.539,66;
- ahorro teórico por flexibilizar la política: USD 31.268,44;
- flete: reducción de USD 82.650, equivalente a 551 pallets;
- packaging: aumento de USD 51.381,56;
- efecto neto: ahorro de USD 31.268,44;
- configuración mixta: 426 productos con 3 mm y solamente BR0247 con 5 mm;
- grosor de 4,5 mm: no utilizado;
- tipos de caja: 51 en el escenario oficial y 57 en el diagnóstico.

El CSV diagnóstico fue enviado a Kaggle y recibió score 0. Esto confirma
empíricamente que la uniformidad del grosor es una restricción dura de
factibilidad. Por lo tanto, el escenario mixto no debe presentarse como solución
admisible ni como alternativa de submit. Su utilidad es exclusivamente
gerencial: cuantifica en USD 31.268,44 el costo mínimo de mantener un grosor
uniforme para todo el catálogo bajo el resto de las premisas del modelo.

## Precisión dimensional y reducción adicional de tipos

La solución principal aceptada por Kaggle utiliza dimensiones expresadas con
una precisión de 0,1 mm, alcanza un costo total de USD 188.079.384,24 y emplea
59 tipos de caja. Como análisis complementario se amplió la enumeración a una
precisión de seis decimales de milímetro. Se encontró una solución con el mismo
costo total y la misma cantidad de pallets, pero con 58 tipos de caja.

La reducción no mejora el score de Kaggle, porque `N_tipos` es una métrica de
seguimiento y no forma parte de la función objetivo. Sin embargo, sí avanza en
una dirección comercial relevante para Bonsai Corp: reducir la dispersión del
portafolio de cajas y simplificar compras, inventario, homologación y operación.

Desde el punto de vista físico, las diferencias entre ambas geometrías son
fracciones inferiores a una décima de milímetro. Frente a laterales de pallet
de 800 y 1.200 mm, esa diferencia es dimensionalmente inapreciable y
previsiblemente inferior a las tolerancias normales de fabricación y armado.
Por eso, los seis decimales no deben presentarse como una especificación
industrial que deba fabricarse literalmente, sino como evidencia matemática de
que existe una oportunidad adicional de estandarización. La recomendación de
negocio es validar con Ingeniería y el proveedor si las tolerancias reales
permiten homologar esa geometría como un único tipo sin alterar fit, headspace,
resistencia ECT ni capacidad de pallet.

Evidencia reproducible: la solución decimal complementaria se encuentra en
`output_continuous_breakpoints_probe/asignacion_decimal.csv`.
