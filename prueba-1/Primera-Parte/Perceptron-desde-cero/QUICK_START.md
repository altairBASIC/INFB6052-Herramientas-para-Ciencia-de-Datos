# Guia Rapida de Inicio - Perceptron desde Cero

## Inicio Rapido (3 pasos)

### 1. Instalar Dependencias

```bash
pip install numpy pandas matplotlib
```

### 2. Descargar Dataset

**Opcion A - Automatico (requiere sklearn):**
```bash
pip install scikit-learn
python download_iris.py
```

**Opcion B - Manual:**
1. Descargar de: https://www.kaggle.com/datasets/uciml/iris/data
2. Guardar como `data/iris.csv`

### 3. Ejecutar Entrenamiento

```bash
python train_perceptron.py
```

**Salida esperada:**
- 6 graficos en `artifacts/`
- Resultados en `artifacts/perceptron_results.json`
- Accuracy: 100% (convergencia garantizada)

---

## Archivos Principales

| Archivo | Descripcion | Uso |
|---------|-------------|-----|
| `README.md` | Documentacion completa | Leer primero |
| `train_perceptron.py` | Script principal | Ejecutar para entrenar |
| `download_iris.py` | Descarga dataset | Ejecutar si no tienes iris.csv |
| `perceptron_iris.ipynb` | Analisis interactivo | Abrir en Jupyter |
| `src/perceptron.py` | Implementacion del perceptron | Codigo fuente |

---

## Estructura del Proyecto

```
Perceptron-desde-cero/
├── README.md                          # Documentacion completa
├── QUICK_START.md                     # Esta guia
├── SUMARIO_PROYECTO.md                # Resumen del proyecto
├── train_perceptron.py                # Script principal
├── download_iris.py                   # Helper para dataset
├── requirements.txt                   # Dependencias
├── data/
│   ├── iris.csv                       # Dataset (descargar)
│   └── INSTRUCCIONES_DATASET.md       # Como obtener el dataset
├── src/
│   ├── __init__.py
│   ├── perceptron.py                  # Implementacion del perceptron
│   ├── data_preprocessing.py          # Preprocesamiento
│   └── visualization.py               # Graficos
├── perceptron_iris.ipynb              # Analisis interactivo
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

## Ejemplo de Uso Programático

```python
from src.perceptron import Perceptron
from src.data_preprocessing import prepare_iris_data

# Preparar datos
data = prepare_iris_data(filepath='data/iris.csv')

# Entrenar
model = Perceptron(learning_rate=0.01, n_iterations=100)
model.fit(data['X_train'], data['y_train'])

# Evaluar
accuracy = model.score(data['X_test'], data['y_test'])
print(f"Accuracy: {accuracy*100:.2f}%")  # Esperado: 100.00%
```

---

## Visualizaciones Generadas

Al ejecutar `train_perceptron.py`, se generan automáticamente:

1. **Datos de entrenamiento** - Scatter plot de las 2 clases
2. **Línea de decisión estimativa** - Separación basada en centroides
3. **Frontera de decisión (train)** - Frontera aprendida en entrenamiento
4. **Frontera de decisión (test)** - Frontera aplicada a datos de prueba
5. **Evolución del error** - Gráfico de convergencia por época
6. **Resumen completo** - Panel con 3 visualizaciones principales

---

## Verificación de Instalación

```bash
# 1. Verificar Python
python --version  # Debe ser >= 3.7

# 2. Verificar dependencias
python -c "import numpy, pandas, matplotlib; print('OK')"

# 3. Verificar dataset
python -c "import pandas as pd; print(pd.read_csv('data/iris.csv').shape)"
# Debe imprimir: (150, 5)

# 4. Verificar implementación
python -c "from src.perceptron import Perceptron; print('OK')"
```

---

## Solución de Problemas

### Error: "No such file or directory: 'data/iris.csv'"

**Solución:**
```bash
python download_iris.py
```

O descargar manualmente de Kaggle.

### Error: "ModuleNotFoundError: No module named 'numpy'"

**Solución:**
```bash
pip install -r requirements.txt
```

### Error: "No module named 'src'"

**Solución:** Asegúrate de ejecutar desde el directorio raíz del proyecto:
```bash
cd Perceptron-desde-cero
python train_perceptron.py
```

---

## Requisitos del Sistema

- **Python:** >= 3.7
- **NumPy:** >= 1.20.0 (obligatorio)
- **Pandas:** >= 1.3.0 (recomendado)
- **Matplotlib:** >= 3.4.0 (recomendado)
- **Espacio en disco:** ~10 MB (código + dataset + visualizaciones)
- **RAM:** ~100 MB durante ejecución

---

## Comandos Útiles

```bash
# Ejecutar script principal
python train_perceptron.py

# Abrir notebook interactivo
jupyter notebook perceptron_iris.ipynb

# Descargar dataset automaticamente
python download_iris.py

# Instalar dependencias
pip install -r requirements.txt

# Limpiar artifacts (Windows)
del artifacts\*
```

---

## Resultados Esperados

**Convergencia:**
- Épocas necesarias: ~10-15
- Errores finales: 0

**Métricas:**
- Accuracy (train): 100%
- Accuracy (test): 100%

**Archivos generados:**
- 6 imágenes PNG en `artifacts/`
- 1 archivo JSON con resultados

**Tiempo de ejecución:**
- < 5 segundos en hardware moderno

---

## Proximos Pasos

Despues de ejecutar exitosamente el proyecto:

1. **Explorar el notebook:** `perceptron_iris.ipynb`
2. **Leer README completo:** Teoria y detalles de implementacion
3. **Examinar codigo fuente:** `src/perceptron.py`
4. **Experimentar con parametros:**
   - Cambiar `learning_rate` (0.001 - 0.1)
   - Usar diferentes `random_state`
   - Probar con Iris-versicolor vs Iris-virginica (no separables)

---

## Contacto y Soporte

**Documentación completa:** Ver `README.md`  
**Repositorio:** [INFB6052-Herramientas-Para-Ciencia-de-Datos](https://github.com/altairBASIC/INFB6052-Herramientas-Para-Ciencia-de-Datos)  
**Contexto:** Primera Prueba - Item 5 (Perceptrón desde cero)

---

**Última actualización:** Octubre 7, 2025
