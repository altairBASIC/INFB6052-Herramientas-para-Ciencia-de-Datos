# Semana 04 - Perceptrón: Fundamentos y Comparativa Avanzada

Este módulo integra dos capas pedagógicas:

- **Parte 1 (Básica - Guía Semana 04):** implementación mínima del Perceptrón sobre un dataset lógico (AND), siguiendo exactamente los pasos solicitados en la guía (Introducción → Dataset → Importar Clase → Entrenamiento → Métricas/Evolución → Frontera 2D → Evolución de Pesos → Conclusiones → Próximos Pasos).
- **Parte 2 (Avanzada):** flujo completo aplicado al dataset real *Breast Cancer Wisconsin (Diagnostic)* con EDA, ingeniería de características, comparación con modelos de `scikit-learn`, validación cruzada, tuning, ROC/AUC, persistencia y resumen programático.

El notebook `perceptron_comparativo.ipynb` consolida ambas partes de forma secuencial y auto-explicativa.

## Objetivos de Aprendizaje

1. Comprender la regla de decisión y actualización del Perceptrón.
2. Observar su comportamiento en un problema linealmente separable (AND) y documentar convergencia.
3. Analizar evolución de errores, accuracy y parámetros (pesos, bias).
4. Visualizar la frontera de decisión en 2D y relacionarla con los pesos aprendidos.
5. Extender el análisis a un dataset real con más dimensiones y métricas avanzadas.
6. Comparar implementación propia vs modelos optimizados (`Perceptron`, `SGDClassifier`, `LogisticRegression`).
7. Aplicar validación cruzada y búsqueda de hiperparámetros.
8. Persistir artefactos y generar un resumen reproducible de resultados.

## Componentes Clave

| Recurso                                     | Descripción                                                                    |
| ------------------------------------------- | ------------------------------------------------------------------------------- |
| `perceptron_comparativo.ipynb`            | Notebook con Parte 1 (básica) y Parte 2 (avanzada).                            |
| `perceptron_lab/perceptron/perceptron.py` | Implementación desde cero (`PerceptronScratch`) usada en notebook y scripts. |
| `train_perceptron_scratch.py`             | (Si disponible) Script CLI para entrenar y producir artefactos reproducibles.   |
| `artifacts/`                              | Modelos serializados, métricas y resúmenes exportados.                        |
| `guia-perceptron.md`                      | Guía original (Semana 05) que define los pasos mínimos.                       |

## Implementación Scratch (Resumen)

La clase `PerceptronScratch` (ver código fuente) ofrece:

- Parámetros: `learning_rate`, `n_epochs`, `shuffle`, `random_state`, (extensible a early stopping).
- Registro de historia: errores por época, accuracy (train y/o val), pesos y bias (según versión usada en el notebook).
- Métodos principales: `fit`, `predict`, `decision_function`.
- Diseño pensado para trazabilidad (historial facilita gráficas y diagnóstico de convergencia).

En la Parte 1 se usa una versión mínima; en la Parte 2 se amplía el uso dentro de un pipeline analítico.

## Estructura del Notebook

### Parte 1 (Guía Base - Pasos 1 a 8)

1. Introducción: definición y problema que resuelve (fronteras lineales).
2. Dataset de Ejemplo (AND) y visualización.
3. Importar la Clase (`PerceptronScratch` con fallback robusto de ruta).
4. Entrenamiento (épocas, errores, accuracy básico).
5. Evolución de Pesos y Frontera 2D (en el notebook se muestran gráficos de parámetros + frontera separadamente).
6. Métricas y Evolución (curvas de errores/accuracy y cálculo de convergencia).
7. Conclusiones (época de convergencia, patrón de pesos, accuracy=1.0).
8. Próximos Pasos (extender a XOR, MLP, OR/NAND, variaciones de learning rate).

### Parte 2 (Avanzada - Caso Real Breast Cancer)

1. Descarga / extracción con fallback (`sklearn.datasets.load_breast_cancer`).
2. Carga y estructura (nombres de columnas, shape, distribución).
3. Limpieza y conversión de tipos (id, mapeo diagnóstico, coerción numérica).
4. EDA básico (estadísticos, distribución de clases, correlaciones).
5. Ingeniería de características + escalado (`StandardScaler`).
6. División train / val / test estratificada.
7. Definición matemática formal (notación y regla de actualización).
8. Implementación scratch (clase en línea para auto-contenido).
   9-11. Entrenamiento, historial, evaluación métricas (precision, recall, F1, matriz de confusión).
9. Visualización de frontera (PCA 2D) como proyección ilustrativa.
10. Ajuste manual de hiperparámetros (grid pequeña lr vs épocas).
11. Implementaciones scikit-learn (Perceptron, SGDClassifier modo perceptrón, LogisticRegression baseline).
12. Comparación de modelos (tabla métrica ordenada por F1).
13. Curvas ROC y AUC.
14. Validación cruzada estratificada.
15. Manejo de desbalance (class_weight='balanced').
16. Medición de tiempos.
17. Persistencia de modelos (JSON / joblib / CSV resumen).
18. Pruebas unitarias conceptuales.
19. Pipeline + GridSearchCV (búsqueda sistemática F1).
20. Resumen programático (export consolidado reproducible).
21. Conclusiones y trabajo futuro (extensiones: regularización, calibración, SVM, multiclase, etc.).

Cada paso avanzado incluye interpretación breve para reforzar criterio analítico.

## Ejecución Rápida (Opcional Script CLI)

Si dispones del script (ajusta ruta según estructura):

```powershell
python .\semana-04-perceptron\train_perceptron_scratch.py --lr 0.01 --epochs 50 --verbose
```

Argumentos típicos:

- `--lr`: tasa de aprendizaje.
- `--epochs`: máximo de épocas.
- `--no-shuffle`: desactiva barajado.
- (Versiones extendidas pueden incluir paciencia / early stopping y export JSON/CSV).

Artefactos esperados en `artifacts/`:

- `model_perceptron.json`
- `metrics.json`
- `training_curve.csv`

## Cargar un Modelo Guardado (Ejemplo)

```python
import json
from perceptron_lab.perceptron.perceptron import PerceptronScratch

with open('artifacts/model_perceptron.json','r',encoding='utf-8') as f:
   data = json.load(f)
model = PerceptronScratch.from_dict(data)
```

## Métricas Clave (Interpretación)

- `errors` / `errors_`: cantidad de actualizaciones; cero temprano = convergencia en datos separables.
- `train_accuracy` / `val_accuracy`: estabilidad y generalización básica.
- Evolución de pesos/bias: estabilidad y ausencia de oscilaciones fuertes.
- AUC / ROC: discriminación global (Parte 2).
- F1: balance precisión–recall en dataset real.

## Buenas Prácticas

- Fijar `random_state` para reproducibilidad.
- Escalar siempre features antes de modelos lineales sensibles a escala.
- Mantener separación clara train / val / test (evitar fuga de datos).
- Versionar artefactos (hash, timestamp) si se automatiza el pipeline.
- Añadir pruebas unitarias mínimas (dimensiones, consistencia de predicción) para robustez evolutiva.

## Informe / Documentación Técnica

Al redactar un informe (Markdown / LaTeX) se recomienda incluir:

1. Formulación matemática (regla de decisión y actualización).
2. Evidencia de convergencia en AND (época y estabilización de pesos).
3. Limitaciones: incapacidad en problemas no linealmente separables (ej. XOR).
4. Resultados comparativos (tabla métricas + tiempos).
5. Interpretación de ROC/AUC y estabilidad CV.
6. Justificación de hiperparámetros seleccionados.

## Próximos Pasos (Extensiones)

- Demostrar fallo en XOR y resolver con MLP.
- Añadir regularización (L1/L2) a versión scratch.
- Implementar esquema multiclase (One-vs-Rest).
- Explorar calibración (Platt / isotónica) con modelos probabilísticos comparativos.
- Incorporar búsqueda bayesiana de hiperparámetros.
- Añadir reporte automatizado (script que regeneré métricas y figuras).

---

Última actualización: alineada con `perceptron_comparativo.ipynb` y `guia-perceptron.md`.
