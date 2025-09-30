# Guía para Notebook de Análisis (Semana 04)

## 1. Introducción
- Describir el dataset (fuente, número de filas, variables clave, objetivo del análisis).

## 2. Carga de Librerías
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", context="notebook")
```

## 3. Carga de Datos
```python
df = pd.read_csv('datos/tu_dataset.csv')
df.head()
```

## 4. Exploración Inicial
```python
df.shape
df.dtypes
df.isna().sum()
```

## 5. Estadísticas Descriptivas
```python
df.describe(include='all').T
```

## 6. Visualizaciones Univariadas
```python
for col in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.show()
```

## 7. Visualizaciones Bivariadas
```python
sns.pairplot(df.select_dtypes(include='number'))
```

## 8. Correlaciones
```python
plt.figure(figsize=(10,6))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('Matriz de Correlación')
plt.show()
```

## 9. Hallazgos
- Lista de hallazgos clave numerados.

## 10. Conclusiones
- Resumen y próximos pasos.
