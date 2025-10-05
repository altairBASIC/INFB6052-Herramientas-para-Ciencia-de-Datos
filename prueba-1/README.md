# Primera Prueba – Herramientas para Ciencia de Datos

> Archivo asociado: `Primera_Prueba_2S2025-1.pdf`
> Completa/ajusta los campos marcados con TODO según las indicaciones exactas del enunciado oficial.

## 1. Descripción General

Esta primera prueba evalúa los conocimientos y habilidades iniciales del curso **Herramientas para Ciencia de Datos**. Se centra en el correcto uso del entorno de trabajo, control de versiones, manejo de librerías científicas en Python y la aplicación básica de un flujo reproducible de análisis / modelado.

## 2. Objetivos de Aprendizaje

- Comprender y documentar un entorno reproducible de trabajo en Python.
- Utilizar Git y GitHub siguiendo buenas prácticas (commits claros, estructura ordenada, README informativo).
- Realizar exploración de datos básica (EDA) apoyándose en librerías estándar.
- Implementar (o reutilizar) un modelo simple (e.g. Perceptrón) y reportar resultados.
- Automatizar (cuando sea posible) pasos de preparación / entrenamiento.
- Comunicar hallazgos y resultados de forma clara y concisa.

## 3. Alcance de los Contenidos Evaluados

- Entorno y configuración (virtualenv / requirements / estructura).
- Uso de Git (commits progresivos, tags si aplica, ramas opcionales).
- Python básico + librerías: `numpy`, `pandas`, `matplotlib` / `seaborn`, `scikit-learn` (u otra según requerimiento del PDF).
- Lectura y limpieza inicial de datos.
- Métricas simples del modelo (accuracy, precision, recall, etc. según corresponda).
- Documentación y reporte.

## 4. Entregables

| Entregable                | Descripción                                                   | Estado            |
| ------------------------- | -------------------------------------------------------------- | ----------------- |
| README.md                 | Documento base del proyecto con instrucciones y justificación | ✅ (este archivo) |
| Reporte PDF               | Resumen ejecutivo + metodología + resultados                  | TODO              |
| Notebook(s)               | EDA y/o experimentos (`.ipynb`)                              | TODO              |
| Script(s) reproducibles   | Ej:`train_perceptron_scratch.py`                             | TODO              |
| Archivo de requerimientos | `requirements.txt`                                           | TODO              |
| Métricas / artefactos    | Ej:`metrics.json`, modelos serializados                      | TODO              |
| Registro de commits       | Historial lógico y limpio                                     | En progreso       |

## 5. Criterios de Evaluación (Propuesta / Ajustar según PDF)

| Criterio                      | % (o puntos) | Descripción                                                                                |
| ----------------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| Organización del repositorio | TODO         | Estructura clara, carpetas lógicas, no hay datos crudos pesados innecesarios sin comprimir |
| Reproducibilidad              | TODO         | Entorno instalable, instrucciones claras, script/notebook corre sin errores                 |
| Flujo de EDA                  | TODO         | Limpieza, descriptores, visualizaciones pertinentes                                         |
| Implementación de modelo     | TODO         | Correcta carga de datos, división train/test, entrenamiento y evaluación                  |
| Métricas y análisis         | TODO         | Métricas correctas + breve interpretación                                                 |
| Uso de Git                    | TODO         | Commits incrementales con mensajes significativos                                           |
| Reporte / Comunicación       | TODO         | Claridad, síntesis, conclusiones fundamentadas                                             |
| Buenas prácticas de código  | TODO         | Nombres claros, modularización, ausencia de código muerto                                 |

> Reemplaza TODO con los valores oficiales (puntos o porcentajes) del enunciado.

## 6. Estructura Recomendada del Repositorio

```
prueba-1/
  README.md
  Primera_Prueba_2S2025-1.pdf
  requirements.txt
  data/
    raw/               # Datos originales (si se permite incluirlos)
    processed/         # Datos limpios / derivados
  notebooks/
    01_eda.ipynb
    02_modelo.ipynb
  src/                 # Código fuente (funciones reutilizables)
    __init__.py
    data_loading.py
    preprocessing.py
    model.py
    metrics.py
  scripts/
    train_model.py
    evaluate_model.py
  artifacts/
    model.joblib
    metrics.json
    confusion_matrix.png
  report/
    informe.pdf        # Versión final formateada
```

(Ajusta si el PDF impone otra convención.)

## 7. Requisitos del Entorno

- Python: 3.10+ (confirmar versión exacta si el enunciado lo restringe)
- Sistema operativo: Windows / Linux / macOS (cualquiera soportado)
- Dependencias listadas en `requirements.txt`

### Instalación Rápida

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Congelar Dependencias (si agregas nuevas)

```powershell
pip freeze > requirements.txt
```

## 8. Flujo Recomendado de Trabajo

1. Clonar repositorio.
2. Crear y activar entorno virtual.
3. Instalar dependencias.
4. Inspeccionar datos (notebook 01_eda).
5. Preparar funciones reutilizables (en `src/`).
6. Entrenar modelo (script o notebook 02_modelo).
7. Guardar artefactos (modelo + métricas).
8. Generar visualizaciones clave.
9. Redactar reporte (síntesis técnica + interpretación).
10. Última revisión de README y limpieza de archivos innecesarios.

## 9. Ejecución (Ejemplo)

Entrenar y generar métricas:

```powershell
python .\scripts\train_model.py --data .\data\processed\dataset.csv --outdir .\artifacts
```

Evaluar un modelo guardado:

```powershell
python .\scripts\evaluate_model.py --model .\artifacts\model.joblib --data .\data\processed\test.csv
```

(Adapta a los nombres reales que utilices.)

## 10. Métricas y Reporte

- Guardar resultados numéricos en `artifacts/metrics.json`.
- Incluir al menos: tamaño de conjuntos, accuracy, (y otras métricas pedidas: TODO), fecha de ejecución, versión de librerías.
- Capturar figura(s) principales: matriz de confusión, curva de entrenamiento, etc.

## 11. Buenas Prácticas de Git

- Commits atómicos y descriptivos (en español o inglés consistente).
- Evitar subir grandes binarios no requeridos.
- Usar `.gitignore` (agregar si no existe) para: entorno virtual, cachés, artefactos temporales.

Ejemplo de mensajes de commit:

```
feat: agrega script de entrenamiento base
refactor: extrae función de limpieza a preprocessing.py
docs: actualiza sección de métricas en README
```

## 13. Preguntas Frecuentes (FAQ)

**¿Puedo usar más librerías que las listadas?**

> Solo si el enunciado lo permite. Documenta cualquier dependencia adicional.

**¿Qué pasa si mis resultados difieren ligeramente?**

> Documenta tu semilla aleatoria y versión de librerías para justificar reproducibilidad.

**¿Debo incluir datos originales?**

> Incluir solo si el tamaño lo permite y está autorizado; de lo contrario documentar la fuente y pasos para obtenerlos.

## 14. Posibles Extensiones (Opcional)

- Añadir pipeline de preprocesamiento con `scikit-learn` (`Pipeline`, `StandardScaler`).
- Incluir script de evaluación cruzada.
- Automatizar ejecución con `make` o `invoke`.
- Añadir pruebas unitarias mínimas para funciones de limpieza / métricas.
