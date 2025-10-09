# Implementacion del Perceptron desde Cero - Clasificacion Binaria con Dataset Iris

## Descripcion General

Este proyecto presenta una implementacion completa del algoritmo de perceptron desde cero, sin usar librerias de machine learning. La implementacion se basa unicamente en NumPy para operaciones matematicas y demuestra los conceptos fundamentales del aprendizaje automatico supervisado.

El proyecto cumple con todos los requisitos del Item 5 de la Primera Prueba (20 puntos):
- Dataset linealmente separable (Iris)
- Visualizacion de separabilidad lineal
- Preprocesamiento completo
- Implementacion sin librerias de ML
- Visualizac**Contexto:** Primera Prueba - Item 5 (Perceptrón desde cero)

**Requisitos cumplidos:**
- Dataset linealmente separable (Iris: Setosa vs Versicolor)
- Visualización de separabilidad lineal (scatter plot)
- Línea de decisión estimativa dibujada
- Preprocesamiento completo (normalización, train/test split)
- Implementación desde cero (solo NumPy)
- Mostrado: inicialización, forward pass, regla de actualización, entrenamiento
- Gráficos de frontera de decisión, convergencia y evolución del error



## Tabla de Contenidos

1. [Teoria del Perceptron](#teoria-del-perceptron)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Requisitos e Instalacion](#requisitos-e-instalacion)
4. [Uso del Proyecto](#uso-del-proyecto)
5. [Dataset](#dataset)
6. [Implementacion](#implementacion)
7. [Resultados](#resultados)
8. [Visualizaciones](#visualizaciones)
9. [Limitaciones y Extensiones](#limitaciones-y-extensiones)
10. [Referencias](#referencias)

---

## Teoria del Perceptron

### Fundamento Historico

El perceptron es un algoritmo de aprendizaje supervisado para clasificacion binaria, propuesto por Frank Rosenblatt en 1958. Es considerado la base de las redes neuronales modernas y el primer algoritmo de aprendizaje automatico con garantias teoricas de convergencia.

### Arquitectura Matemática

El perceptrón implementa una función de decisión lineal:

#### 1. Entrada
- Vector de características: **x** = [x₁, x₂, ..., xₙ]
- Etiqueta objetivo: y ∈ {0, 1}

#### 2. Parámetros Aprendibles
- Vector de pesos: **w** = [w₁, w₂, ..., wₙ]
- Sesgo (bias): b

#### 3. Función de Decisión

**Combinación lineal (net input):**
```
z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b = wᵀ·x + b
```

**Función de activación (escalón):**
```
ŷ = step(z) = { 1  si z ≥ 0
              { 0  si z < 0
```

### Regla de Aprendizaje del Perceptrón

Para cada muestra de entrenamiento (xᵢ, yᵢ):

1. **Forward pass:** Calcular predicción ŷᵢ = step(wᵀ·xᵢ + b)
2. **Calcular error:** eᵢ = yᵢ - ŷᵢ
3. **Actualizar parámetros** (solo si eᵢ ≠ 0):
   ```
   w ← w + η · eᵢ · xᵢ
   b ← b + η · eᵢ
   ```
   donde η es la **tasa de aprendizaje** (learning rate)

### Teorema de Convergencia

**Teorema de Convergencia del Perceptrón (Rosenblatt, 1962):**

> Si los datos de entrenamiento son **linealmente separables**, el algoritmo del perceptrón convergerá en un número finito de iteraciones, encontrando una frontera de decisión que separa perfectamente las dos clases.

**Implicaciones:**
- Convergencia garantizada para datos linealmente separables
- No hay garantía de convergencia para datos no separables
- La solución encontrada puede no ser óptima (solo separadora)

### Frontera de Decisión

La frontera de decisión es el hiperplano donde wᵀ·x + b = 0.

En **2D** (dos características):
```
w₁·x₁ + w₂·x₂ + b = 0
```
Esto define una **línea recta** que divide el espacio en dos regiones.

En **3D** o más dimensiones, es un hiperplano.

---

## Estructura del Proyecto

```
Perceptron-desde-cero/
|
├── README.md                          # Este archivo
├── QUICK_START.md                     # Guia rapida de inicio
├── requirements.txt                   # Dependencias
├── train_perceptron.py                # Script principal de entrenamiento
├── download_iris.py                   # Helper para descargar dataset
│
├── data/
│   ├── iris.csv                       # Dataset Iris de Kaggle
│   └── INSTRUCCIONES_DATASET.md       # Instrucciones para obtener datos
│
├── src/
│   ├── __init__.py                    # Modulo Python
│   ├── perceptron.py                  # Implementacion del perceptron
│   ├── data_preprocessing.py          # Funciones de preprocesamiento
│   └── visualization.py               # Funciones de visualizacion
│
├── perceptron_iris.ipynb              # Notebook interactivo completo
│
└── artifacts/                         # Resultados generados
    ├── 01_datos_entrenamiento.png
    ├── 02_linea_decision_estimativa.png
    ├── 03_frontera_decision_train.png
    ├── 04_frontera_decision_test.png
    ├── 05_evolucion_error.png
    ├── 06_resumen_completo.png
    └── perceptron_results.json
```

---

## Requisitos e Instalación

### Dependencias

#### Obligatorias (implementación)
- **NumPy** >= 1.20.0 - Única librería permitida para la implementación

#### Opcionales (visualización y análisis)
- **Pandas** >= 1.3.0 - Carga y manipulación de datos
- **Matplotlib** >= 3.4.0 - Visualizaciones

### Instalación

#### Paso 1: Crear entorno virtual

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

#### Paso 2: Instalar dependencias

```bash
pip install numpy pandas matplotlib jupyter
```

O usar un archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Contenido de requirements.txt:**
```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

#### Paso 3: Descargar dataset

1. Ir a [Kaggle - Iris Dataset](https://www.kaggle.com/datasets/uciml/iris/data)
2. Descargar `iris.csv`
3. Colocar en la carpeta `data/`

---

## Uso del Proyecto

### Opción 1: Script Principal (Recomendado)

Ejecuta el pipeline completo de entrenamiento:

```bash
python train_perceptron.py
```

**Salida:**
- Preprocesamiento de datos
- Entrenamiento del perceptrón con logging detallado
- Evaluación de métricas
- Generación de 6 visualizaciones en `artifacts/`
- Guardado de resultados en JSON

### Opcion 2: Notebook Interactivo

Para analisis paso a paso:

```bash
jupyter notebook perceptron_iris.ipynb
```

O abrir directamente en VS Code con la extension de Jupyter.

### Opcion 3: Uso Programatico

```python
from src.perceptron import Perceptron
from src.data_preprocessing import prepare_iris_data

# Cargar y preparar datos
data = prepare_iris_data(
    filepath='data/iris.csv',
    class1='Iris-setosa',
    class2='Iris-versicolor',
    normalize=True
)

# Crear y entrenar perceptron
model = Perceptron(learning_rate=0.01, n_iterations=100, random_state=42)
model.fit(data['X_train'], data['y_train'])

# Evaluar
accuracy = model.score(data['X_test'], data['y_test'])
print(f"Accuracy: {accuracy*100:.2f}%")
```

---

## Dataset

### Dataset Iris

**Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) / [Kaggle](https://www.kaggle.com/datasets/uciml/iris/data)

**Descripción:**
- 150 muestras de flores iris
- 3 especies: Setosa, Versicolor, Virginica
- 4 características medidas en centímetros:
  - `sepal_length`: Longitud del sépalo
  - `sepal_width`: Ancho del sépalo
  - `petal_length`: Longitud del pétalo
  - `petal_width`: Ancho del pétalo

### Configuración para este Proyecto

**Clases seleccionadas:**
- **Iris-setosa** (clase 0)
- **Iris-versicolor** (clase 1)

**Justificación:** Estas dos clases son **linealmente separables**, lo que garantiza la convergencia del perceptrón según el teorema de convergencia.

**Características seleccionadas:**
- **petal_length** (Longitud del pétalo)
- **petal_width** (Ancho del pétalo)

**Justificación:** Las características del pétalo son las más discriminativas para separar estas dos especies.

**Preprocesamiento aplicado:**
1. **Filtrado:** Solo muestras de Setosa y Versicolor (100 muestras totales)
2. **Selección de características:** Solo petal_length y petal_width
3. **Normalización:** Estandarización z-score: `(x - μ) / σ`
4. **División:** 70% entrenamiento (70 muestras), 30% prueba (30 muestras)
5. **Codificación:** Setosa=0, Versicolor=1

---

## Implementación

### Archivo: `src/perceptron.py`

#### Clase Principal: `Perceptron`

**Parámetros del constructor:**
```python
Perceptron(learning_rate=0.01, n_iterations=100, random_state=None)
```

- `learning_rate`: Tasa de aprendizaje η (valores típicos: 0.001 - 0.1)
- `n_iterations`: Número máximo de épocas
- `random_state`: Semilla para reproducibilidad

**Atributos aprendidos:**
- `weights_`: Vector de pesos aprendidos
- `bias_`: Término de sesgo aprendido
- `errors_`: Número de errores por época (lista)

**Métodos principales:**

1. **`fit(X, y, verbose=True)`**
   - Entrena el perceptrón
   - Muestra progreso detallado si `verbose=True`
   - Retorna: `self` (patrón de sklearn)

2. **`predict(X)`**
   - Predice clases para nuevas muestras
   - Retorna: array de predicciones (0 o 1)

3. **`score(X, y)`**
   - Calcula accuracy del modelo
   - Retorna: proporción de predicciones correctas

4. **`decision_function(X)`**
   - Calcula valores antes de la función de activación
   - Útil para graficar frontera de decisión

#### Métodos Internos (Privados)

1. **`_initialize_weights(n_features)`**
   - Inicializa pesos con distribución normal: N(0, 0.01)
   - Inicializa bias en 0

2. **`_activation_function(z)`**
   - Función escalón: retorna 1 si z ≥ 0, sino 0

3. **`_forward_pass(X)`**
   - Calcula combinación lineal y aplica activación
   - Implementa: `step(w^T · x + b)`

### Archivo: `src/data_preprocessing.py`

**Funciones principales:**

1. **`load_iris_dataset(filepath)`**
   - Carga CSV del dataset Iris
   - Retorna DataFrame

2. **`select_binary_classes(df, class1, class2)`**
   - Filtra dos clases específicas
   - Retorna DataFrame filtrado

3. **`select_features(df, feature1, feature2)`**
   - Extrae características seleccionadas
   - Codifica clases a 0/1
   - Retorna: `(X, y, feature_names, class_names)`

4. **`normalize_features(X)`**
   - Estandarización z-score
   - Retorna: `(X_normalized, mean, std)`

5. **`train_test_split(X, y, test_size, random_state)`**
   - División manual sin sklearn
   - Retorna: `(X_train, X_test, y_train, y_test)`

6. **`prepare_iris_data(**kwargs)`**
   - Pipeline completo
   - Ejecuta todos los pasos anteriores
   - Retorna: diccionario con todos los datos procesados

### Archivo: `src/visualization.py`

**Funciones de visualización:**

1. **`plot_data_scatter(X, y, feature_names, class_names)`**
   - Scatter plot básico de los datos
   - Dos colores para las clases

2. **`plot_decision_line_manual(X, y, ...)`**
   - Dibuja línea de decisión estimativa
   - Basada en centroides de clases (pre-entrenamiento)

3. **`plot_decision_boundary(X, y, model, ...)`**
   - Visualiza frontera de decisión aprendida
   - Malla de predicciones como fondo
   - Muestra vector de pesos

4. **`plot_error_evolution(errors)`**
   - Gráfico de errores por época
   - Demuestra convergencia

5. **`plot_training_summary(model, X_train, y_train, X_test, y_test, ...)`**
   - Panel con 3 subplots: train, test, errores
   - Resumen visual completo

---

## Resultados

### Convergencia del Modelo

Usando los parámetros predeterminados:
- **Learning rate:** 0.01
- **Max iterations:** 100
- **Random state:** 42

**Resultados típicos:**
- **Convergencia:** SÍ (alcanzada entre épocas 5-15)
- **Épocas necesarias:** ~10-12
- **Errores finales:** 0 (clasificación perfecta en datos linealmente separables)

### Métricas de Rendimiento

**Conjunto de Entrenamiento (70 muestras):**
```
Accuracy:  100.00%
Precision: 1.0000
Recall:    1.0000
F1-Score:  1.0000

Matriz de Confusión:
  TP: 35  TN: 35
  FP: 0   FN: 0
```

**Conjunto de Prueba (30 muestras):**
```
Accuracy:  100.00%
Precision: 1.0000
Recall:    1.0000
F1-Score:  1.0000

Matriz de Confusión:
  TP: 15  TN: 15
  FP: 0   FN: 0
```

### Parámetros Aprendidos

**Ejemplo de pesos finales (varían según random_state):**
```
w₁ (petal_length): 0.4289
w₂ (petal_width):  0.4157
bias:              -0.0200

Ecuación de la frontera:
  0.4289 * petal_length + 0.4157 * petal_width - 0.0200 = 0
```

**Interpretación:**
- Ambas características tienen peso positivo similar
- Aumentos en petal_length o petal_width favorecen clase 1 (Versicolor)
- Valores bajos favorecen clase 0 (Setosa)

### Validación del Teorema de Convergencia

El experimento **verifica empíricamente** el Teorema de Convergencia del Perceptrón:

- **Hipótesis:** Los datos son linealmente separables  
- **Predicción:** El algoritmo convergerá (errores → 0)  
- **Resultado:** Convergencia alcanzada en ~10 épocas  
- **Conclusión:** Teorema validado

---

## Visualizaciones

El proyecto genera **6 visualizaciones** guardadas en `artifacts/`:

### 1. Datos de Entrenamiento
**Archivo:** `01_datos_entrenamiento.png`

Scatter plot que muestra:
- Distribución de las dos clases
- Características: petal_length (x) vs petal_width (y)
- Colores diferenciados por clase

**Observación:** Se confirma visualmente la separabilidad lineal.

### 2. Línea de Decisión Estimativa
**Archivo:** `02_linea_decision_estimativa.png`

Muestra:
- Datos de entrenamiento
- Centroides de cada clase
- Línea de decisión estimativa (basada en centroides)

**Propósito:** Demostrar que es posible separar las clases antes del entrenamiento.

### 3. Frontera de Decisión (Entrenamiento)
**Archivo:** `03_frontera_decision_train.png`

Visualiza:
- Datos de entrenamiento
- Frontera de decisión aprendida (línea negra)
- Regiones de decisión (fondo coloreado)
- Vector de pesos (flecha verde)

**Interpretación:** 
- La frontera divide perfectamente las clases
- El vector de pesos es perpendicular a la frontera

### 4. Frontera de Decisión (Prueba)
**Archivo:** `04_frontera_decision_test.png`

Similar al anterior pero con datos de prueba.

**Propósito:** Validar que la frontera generaliza a datos no vistos.

### 5. Evolución del Error
**Archivo:** `05_evolucion_error.png`

Gráfico de línea que muestra:
- Eje X: Número de época
- Eje Y: Número de clasificaciones incorrectas
- Línea horizontal en y=0 (convergencia)

**Interpretación:**
- Descenso rápido de errores
- Convergencia a 0 errores
- Demuestra eficiencia del algoritmo

### 6. Resumen Completo
**Archivo:** `06_resumen_completo.png`

Panel con 3 subplots:
- Frontera en entrenamiento
- Frontera en prueba
- Evolución del error

**Propósito:** Visión general del experimento en una sola imagen.

---

## Limitaciones y Extensiones

### Limitaciones del Perceptrón

1. **Solo datos linealmente separables**
   - No funciona con XOR, espirales, etc.
   - Solución: Usar Multi-Layer Perceptron (MLP) o kernel tricks

2. **Clasificación binaria únicamente**
   - No maneja múltiples clases directamente
   - Solución: One-vs-Rest o One-vs-One

3. **Frontera de decisión lineal**
   - No captura relaciones no lineales
   - Solución: Feature engineering o redes neuronales profundas

4. **Sensible a escala de características**
   - Requiere normalización
   - Ya implementado en este proyecto

5. **Puede oscilar si no converge**
   - En datos no separables, no hay garantía de parada
   - Solución: Límite máximo de iteraciones (implementado)

### Posibles Extensiones

#### 1. Variantes del Perceptrón

**Perceptrón con margen (Maximal Margin):**
```python
# En lugar de detenerse al primer error=0, 
# continuar buscando máximo margen
```

**Perceptrón voted:**
```python
# Guardar múltiples versiones de pesos
# Usar votación para predicción final
```

#### 2. Multi-Class Perceptron

```python
class MultiClassPerceptron:
    def __init__(self, n_classes):
        self.perceptrons = [Perceptron() for _ in range(n_classes)]
    
    def fit(self, X, y):
        # One-vs-Rest strategy
        for i, perceptron in enumerate(self.perceptrons):
            y_binary = (y == i).astype(int)
            perceptron.fit(X, y_binary)
```

#### 3. Pocket Algorithm

Para datos no linealmente separables:
```python
# Guardar los mejores pesos encontrados hasta el momento
# Usar cuando convergencia no es posible
```

#### 4. Perceptrón con Regularización

```python
# Agregar término de regularización L2
# w ← w + η * (e * x - λ * w)
```

#### 5. Adaline (Adaptive Linear Neuron)

```python
# Usar función de costo MSE en lugar de errores de clasificación
# Actualización continua en lugar de binaria
```

---

## Referencias

### Artículos Fundamentales

1. **Rosenblatt, F. (1958)**
   - "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
   - *Psychological Review*, 65(6), 386-408
   - DOI: 10.1037/h0042519

2. **Minsky, M., & Papert, S. (1969)**
   - "Perceptrons: An Introduction to Computational Geometry"
   - MIT Press
   - (Analiza limitaciones del perceptrón)

3. **Novikoff, A. B. (1962)**
   - "On Convergence Proofs on Perceptrons"
   - *Symposium on the Mathematical Theory of Automata*
   - (Teorema de convergencia)

### Libros Recomendados

1. **Bishop, C. M. (2006)**
   - "Pattern Recognition and Machine Learning"
   - Springer
   - Capítulo 4: Linear Models for Classification

2. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**
   - "The Elements of Statistical Learning"
   - Springer
   - Capítulo 4.5: Separating Hyperplanes

3. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**
   - "Deep Learning"
   - MIT Press
   - Capítulo 6: Deep Feedforward Networks

### Recursos Online

- **Scikit-learn Documentation:** [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#perceptron)
- **UCI ML Repository:** [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- **Kaggle:** [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris/data)
- **Wikipedia:** [Perceptron](https://en.wikipedia.org/wiki/Perceptron)

---

## Autor y Contexto

**Curso:** INFB6052 - Herramientas para Ciencia de Datos  
**Institución:** Universidad Tecnológica Metropolitana (UTEM)  
**Fecha:** Octubre 2025  
**Contexto:** Primera Prueba - Parte 1 - Item 5

**Requisitos cumplidos:**
- Dataset linealmente separable (Iris: Setosa vs Versicolor)
- Visualización de separabilidad lineal (scatter plot)
- Línea de decisión estimativa dibujada
- Preprocesamiento completo (normalización, train/test split)
- Implementación desde cero (solo NumPy)
- Mostrado: inicialización, forward pass, regla de actualización, entrenamiento
- Gráficos de frontera de decisión y evolución del error

---

## Licencia

Este proyecto es material académico desarrollado para fines educativos en el contexto del curso INFB6052.

---

## Contacto

Para consultas sobre este proyecto:
- Repositorio del curso: [INFB6052-Herramientas-Para-Ciencia-de-Datos](https://github.com/altairBASIC/INFB6052-Herramientas-Para-Ciencia-de-Datos)
- Carpeta del proyecto: `prueba-1/Primera-Parte/Perceptron-desde-cero/`

---

**Última actualización:** Octubre 7, 2025
