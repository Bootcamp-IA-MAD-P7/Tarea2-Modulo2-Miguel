# Tarea 2 - Modulo 2: Regresion Lineal Simple

## Parte 1: Fundamentos de Regresion

### 1. Que es la regresion y cual es su proposito en Machine Learning?

La regresion es una tecnica de aprendizaje supervisado que se utiliza para predecir un valor numerico continuo a partir de una o varias variables de entrada. Su proposito es aprender la relacion entre las variables independientes, tambien llamadas caracteristicas o features, y una variable dependiente que queremos estimar.

Por ejemplo, un modelo de regresion puede predecir el salario de una persona a partir de sus anos de experiencia, el precio de una vivienda a partir de sus metros cuadrados o la demanda futura de un producto a partir de datos historicos.

La principal diferencia con la clasificacion es el tipo de salida que produce el modelo:

- En regresion, la salida es un numero continuo, como 45000 euros, 8.7 puntos o 135.5 unidades.
- En clasificacion, la salida es una categoria o clase, como aprobado/suspenso, cliente premium/no premium o spam/no spam.

En resumen, usamos regresion cuando queremos estimar cantidades y clasificacion cuando queremos asignar etiquetas.

### 2. Que es la Regresion Lineal Simple y que representa la ecuacion y_hat = beta_0 + beta_1 * X?

La Regresion Lineal Simple es un modelo de regresion que intenta explicar o predecir una variable numerica usando una sola variable independiente. El modelo busca ajustar una linea recta que represente lo mejor posible la relacion entre ambas variables.

La ecuacion del modelo es:

```text
y_hat = beta_0 + beta_1 * X
```

Donde:

- `y_hat` es el valor predicho por el modelo.
- `X` es la variable independiente o variable de entrada.
- `beta_0` es el intercepto.
- `beta_1` es la pendiente de la recta.

El intercepto `beta_0` representa el valor estimado de `y` cuando `X` vale 0. En un contexto real, si queremos predecir salario a partir de anos de experiencia, `beta_0` seria el salario estimado para una persona con 0 anos de experiencia. No siempre tiene una interpretacion realista, pero es necesario para construir la recta.

La pendiente `beta_1` representa cuanto cambia `y` cuando `X` aumenta en una unidad. En el ejemplo del salario, si `beta_1 = 9000`, significaria que por cada ano adicional de experiencia el salario esperado aumenta aproximadamente en 9000 unidades monetarias.

### 3. Cuales son los supuestos de la Regresion Lineal?

La Regresion Lineal se apoya en varios supuestos que ayudan a que el modelo sea fiable e interpretable.

**Linealidad:** la relacion entre la variable independiente `X` y la variable dependiente `y` debe ser aproximadamente lineal. Esto significa que los datos deberian poder representarse razonablemente con una linea recta. Si la relacion tiene forma de curva, una regresion lineal simple podria quedarse corta.

**Homocedasticidad:** los errores o residuos deben tener una variabilidad parecida a lo largo de todos los valores predichos. Si el error crece mucho para valores altos o bajos, el modelo puede estar prediciendo peor en ciertas zonas.

**Independencia de errores:** los residuos deben ser independientes entre si. Es decir, el error cometido en una observacion no deberia depender del error cometido en otra. Este supuesto es especialmente importante en datos temporales, donde los valores pueden estar relacionados por orden cronologico.

**Normalidad de residuos:** los residuos deberian distribuirse de forma aproximadamente normal alrededor de 0. Esto no siempre es imprescindible para predecir, pero si es importante para interpretar correctamente el modelo y realizar inferencias estadisticas.

Verificar estos supuestos es importante porque nos permite saber si el modelo lineal es adecuado. Si los supuestos se incumplen de forma fuerte, las predicciones, las metricas o la interpretacion de los coeficientes pueden ser poco fiables.

### 4. Que es la funcion de coste en regresion? MSE y MAE

La funcion de coste, o loss function, mide que tan lejos estan las predicciones del modelo respecto a los valores reales. Durante el entrenamiento, el objetivo del modelo es encontrar los parametros que minimizan ese error.

El Error Cuadratico Medio, o MSE, se calcula como el promedio de los errores al cuadrado:

```text
MSE = (1 / n) * sum((y_i - y_hat_i)^2)
```

Al elevar los errores al cuadrado, el MSE penaliza mas los errores grandes. Es util cuando queremos que el modelo preste mucha atencion a predicciones muy alejadas del valor real. Su desventaja es que queda expresado en unidades al cuadrado, por lo que puede ser menos intuitivo.

El Error Absoluto Medio, o MAE, se calcula como el promedio de los valores absolutos de los errores:

```text
MAE = (1 / n) * sum(|y_i - y_hat_i|)
```

El MAE mide el error medio en las mismas unidades que la variable objetivo. Es mas facil de interpretar y es menos sensible a valores atipicos que el MSE.

Preferiria usar MSE cuando los errores grandes son especialmente graves y quiero penalizarlos mas. Preferiria usar MAE cuando quiero una metrica mas interpretable y robusta frente a outliers.

### 5. Que significa dividir los datos en train y test?

Dividir los datos en train y test significa separar el dataset en dos partes:

- Conjunto de entrenamiento, o train: se usa para que el modelo aprenda la relacion entre las variables.
- Conjunto de prueba, o test: se usa para evaluar el rendimiento del modelo con datos que no ha visto durante el entrenamiento.

Esta division es necesaria para comprobar si el modelo realmente generaliza. Si evaluamos el modelo con los mismos datos con los que fue entrenado, podriamos obtener metricas demasiado optimistas.

El problema que evitamos es pensar que el modelo funciona bien cuando en realidad solo ha memorizado los datos de entrenamiento. Evaluar con datos nuevos permite medir mejor su capacidad de prediccion en situaciones reales.

### 6. Que metricas se usan para evaluar un modelo de regresion?

Las metricas mas habituales para evaluar modelos de regresion son R2, MSE, RMSE y MAE.

**R2, coeficiente de determinacion**

```text
R2 = 1 - (SS_res / SS_tot)
```

Donde `SS_res` es la suma de los errores al cuadrado y `SS_tot` es la variabilidad total de los datos respecto a su media. R2 indica que proporcion de la variabilidad de la variable objetivo es explicada por el modelo. Un valor cercano a 1 indica un buen ajuste. Un valor cercano a 0 indica que el modelo no mejora mucho respecto a predecir siempre la media.

**MSE, Mean Squared Error**

```text
MSE = (1 / n) * sum((y_i - y_hat_i)^2)
```

Mide el promedio de los errores al cuadrado. Penaliza especialmente los errores grandes.

**RMSE, Root Mean Squared Error**

```text
RMSE = sqrt(MSE)
```

Es la raiz cuadrada del MSE. Tiene la ventaja de estar en las mismas unidades que la variable objetivo, por lo que es mas interpretable que el MSE.

**MAE, Mean Absolute Error**

```text
MAE = (1 / n) * sum(|y_i - y_hat_i|)
```

Mide el error absoluto promedio. Es facil de interpretar y menos sensible a outliers que MSE y RMSE.

### 7. Que son los residuos y para que sirve analizarlos?

Los residuos son las diferencias entre los valores reales y los valores predichos por el modelo:

```text
residuo = y - y_hat
```

Analizar los residuos sirve para comprobar si el modelo esta cometiendo errores de forma aleatoria o si existe algun patron que indique que el modelo no esta capturando bien la relacion entre las variables.

Un residuo cercano a 0 indica que la prediccion fue muy cercana al valor real. Cuanto mas grande sea el residuo en valor absoluto, mayor fue el error de prediccion.

En un buen modelo lineal, los residuos deberian estar distribuidos aleatoriamente alrededor de 0. Si aparece un patron claro, por ejemplo una curva, una forma de embudo o errores que aumentan sistematicamente, podria indicar que la relacion no es lineal, que hay heterocedasticidad o que faltan variables importantes.

### 8. Que es el overfitting y como se detecta comparando R2 en train vs. test?

El overfitting ocurre cuando un modelo aprende demasiado bien los detalles, ruido o particularidades del conjunto de entrenamiento, pero no generaliza correctamente a datos nuevos.

Una forma de detectarlo es comparar el R2 en train y en test:

- Si el R2 en train es muy alto y el R2 en test tambien es alto y parecido, el modelo generaliza bien.
- Si el R2 en train es muy alto pero el R2 en test baja mucho, puede haber overfitting.

Por ejemplo, si un modelo obtiene `R2_train = 0.98` y `R2_test = 0.65`, probablemente se ha ajustado demasiado a los datos de entrenamiento. En regresion lineal simple el overfitting suele ser menos frecuente que en modelos mas complejos, pero sigue siendo importante comprobarlo.

### 9. Que es la validacion cruzada y que ventaja ofrece frente a una sola division train/test?

La validacion cruzada, o cross-validation, es una tecnica de evaluacion que divide los datos en varias partes o folds. El modelo se entrena varias veces usando diferentes combinaciones de folds para entrenamiento y validacion.

En una validacion cruzada de 5 folds, por ejemplo, el dataset se divide en 5 partes. En cada iteracion, el modelo entrena con 4 partes y evalua con la parte restante. Al final se obtiene una media de las metricas de todas las iteraciones.

La ventaja frente a una sola division train/test es que la evaluacion es mas robusta. Con un unico split, el resultado puede depender demasiado de que datos hayan caido en train y cuales en test. La validacion cruzada reduce esa dependencia y ofrece una estimacion mas estable del rendimiento del modelo, especialmente cuando el dataset es pequeno.

## Proxima parte

La Parte 2 consistira en desarrollar el notebook practico de Regresion Lineal Simple con el Salary Dataset, incluyendo carga de datos, EDA, entrenamiento, evaluacion, analisis de residuos y predicciones nuevas.
