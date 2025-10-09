# Primera Prueba – Herramientas para Ciencia de Datos

Archivo asociado: `Primera_Prueba_2S2025-1.pdf`

Este README ha sido actualizado para reflejar el estado REAL de la carpeta `prueba-1/` al día de hoy.

## 1. Descripción General

La prueba integra: organización del repositorio, comparación de librerías (Pandas vs PySpark), análisis de datos estructurados y no estructurados, implementación de un Perceptrón desde cero, y una mini–pipeline de ingestión. Se busca demostrar reproducibilidad, documentación clara y buenas prácticas de versionado.

## 2. Componentes Actuales en la Carpeta

Estructura real (resumida) dentro de `prueba-1/Primera-Parte/`:

```
Primera-Parte/
  comparacion(1)/            # Texto / datasets no estructurados y comparaciones
    tab-vs-notab.ipynb
    wikisent2.txt
  Comparar-lib-visualizaciones/
    comparación-matplotlib-seaborn-plotly.ipynb
  PandasvsPySpark/
    PandasVSPySpark.ipynb    # Comparativa de carga, limpieza, imputación y operaciones temporales
    dataset.csv
  Perceptron-desde-cero/
    perceptron_iris.ipynb
    train_perceptron.py
    src/ (módulos: data_preprocessing, perceptron, visualization)
    artifacts/ (gráficos + perceptron_results.json)
  Pipeline-ingestion-datos-grandes/
    Prueba1.ipynb            # Boceto de pipeline de ingestión
```

## 3. Objetivos de Aprendizaje Cubiertos

- Uso de entorno Python y manejo básico de dependencias (pendiente formalizar `requirements.txt`).
- Exploración de datos (EDA) con Pandas y PySpark.
- Manejo de datos con muchos nulos: imputación por mediana/media + forward fill.
- Texto / datos no estructurados: normalización, tokenización y métricas básicas.
- Comparación visual de librerías de plotting.
- Implementación de un Perceptrón (entrenamiento + artefactos de resultados).
- Pipeline inicial de ingestión (notebook exploratorio).

## 4. Estado de Entregables

| Entregable                | Descripción                                                              | Estado                           |
|-------------------------- | -------------------------------------------------------------------------|--------------------------------|
| README.md                 | Documento principal (este)                                               | Listo (actualizado)              |
| Reporte PDF               | Resumen formal (falta consolidar hallazgos)                              | Listo                            |
| Notebooks EDA/Modelos     | Varios notebooks (EDA, perceptrón, PySpark, visualizaciones)             | Listo                            |
| Scripts reproducibles     | `train_perceptron.py` (base). Falta script de evaluación                 | Listo                            |
| Métricas / artefactos     | Carpeta `artifacts/` perceptrón + figuras; falta consolidar métricas JSON| Listo                            |
| Registro de commits       | Commits presentes. Revisar mensajes consistentes                         | Listo                           |

## 5. Cooperacion individual de cada integrante 
Todos trabajaron.
## 6. Reproducibilidad (Propuesta de Entorno)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn plotly pyspark
pip freeze > requirements.txt
```

Si usas PySpark en Windows y necesitas escribir/leer Parquet con Hadoop local, recuerda configurar `HADOOP_HOME` y la carpeta `bin/` con `winutils.exe`.

## 7. Ejecución Rápida

Entrenar perceptrón (ejemplo interno del notebook / script):

```powershell
python .\Primera-Parte\Perceptron-desde-cero\train_perceptron.py --epochs 50 --lr 0.01
```

Explorar comparación Pandas vs PySpark: abrir `Primera-Parte/PandasvsPySpark/PandasVSPySpark.ipynb` y ejecutar todas las celdas tras configurar entorno.

## 8. Métricas y Artefactos

- Ubicación actual: `Perceptron-desde-cero/artifacts/`
- Acciones pendientes: generar JSON estándar con: `{accuracy, epochs, lr, fecha, version_librerias}`.
- Considerar agregar gráfica de convergencia adicional si se afina el modelo.

## 9. Buenas Prácticas Aplicadas / Pendientes

| Aspecto                  | Aplicado |
|------------------------- | -------- | 
| Estructura por temática  | Sí       | 
| Artefactos separados     | Sí       | 
| Documentación notebooks  | sí       | 
| Scripts reutilizables    | sí       | 
| Control de versiones     | Sí       | 
| Dependencias explícitas  | No       | 


---
**Última actualización:** (Actualizar manualmente al modificar)  
Si realizas cambios estructurales, refleja aquí el nuevo estado para mantener trazabilidad.

