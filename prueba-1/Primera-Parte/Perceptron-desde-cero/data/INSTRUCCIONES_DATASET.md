# Instrucciones para Descargar el Dataset Iris

## Opción 1: Kaggle (Recomendado)

1. Ir a: https://www.kaggle.com/datasets/uciml/iris/data
2. Click en "Download" (requiere cuenta de Kaggle gratuita)
3. Descomprimir el archivo ZIP
4. Renombrar el archivo a `iris.csv` si es necesario
5. Colocar en la carpeta `data/` de este proyecto

## Opción 2: UCI Machine Learning Repository

1. Ir a: https://archive.ics.uci.edu/ml/datasets/iris
2. Click en "Data Folder"
3. Descargar `iris.data`
4. Renombrar a `iris.csv`
5. Agregar la siguiente línea como encabezado (primera línea del archivo):
   ```
   sepal_length,sepal_width,petal_length,petal_width,species
   ```
6. Colocar en la carpeta `data/` de este proyecto

## Opción 3: Crear desde Código (Alternativa)

Si tienes instalado scikit-learn, puedes generar el CSV con este script:

```python
from sklearn import datasets
import pandas as pd

# Cargar dataset Iris
iris = datasets.load_iris()

# Crear DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})

# Guardar como CSV
df.to_csv('data/iris.csv', index=False)
print("Dataset guardado en data/iris.csv")
```

## Verificación

Después de descargar, verifica que el archivo esté correctamente ubicado:

```
Perceptron-desde-cero/
└── data/
    └── iris.csv  ← El archivo debe estar aquí
```

Puedes verificar el contenido con:

```python
import pandas as pd
df = pd.read_csv('data/iris.csv')
print(df.head())
print(f"Dimensiones: {df.shape}")  # Debe ser (150, 5)
```

## Estructura Esperada del CSV

El archivo debe tener 5 columnas:
1. `sepal_length` (o `SepalLengthCm`)
2. `sepal_width` (o `SepalWidthCm`)
3. `petal_length` (o `PetalLengthCm`)
4. `petal_width` (o `PetalWidthCm`)
5. `species` (o `Species` o `variety`)

Y 150 filas (sin contar el encabezado).

Las 3 especies deben ser:
- Iris-setosa (50 muestras)
- Iris-versicolor (50 muestras)
- Iris-virginica (50 muestras)
