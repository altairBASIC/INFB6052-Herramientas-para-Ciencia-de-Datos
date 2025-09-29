# Semana 05 – Perceptrón (Implementación desde Cero) y Comparativa

Esta semana se profundiza en el perceptrón implementado manualmente (sin scikit-learn) y se compara su desempeño con modelos equivalentes de la librería (`Perceptron`, `SGDClassifier`, `LogisticRegression`).

## Objetivos de Aprendizaje
- Entender matemáticamente la regla de actualización del perceptrón.
- Implementar y entrenar un perceptrón scratch (`PerceptronScratch`).
- Analizar convergencia (errores por época) y early stopping.
- Comparar contra implementaciones optimizadas de scikit-learn.
- Evaluar métricas fundamentales (accuracy, precision, recall, f1, ROC/AUC).
- Persistir y versionar artefactos del modelo.

## Componentes Clave
| Recurso | Descripción |
|---------|-------------|
| `perceptron_comparativo.ipynb` | Notebook principal de análisis y comparación. |
| `train_perceptron_scratch.py` | Script CLI para entrenar rápidamente y generar artefactos. |
| `perceptron_lab/perceptron/perceptron.py` | Implementación desde cero (`PerceptronScratch`). |
| `artifacts/` | Carpeta con salidas (modelo, métricas, curvas). |

## Implementación Scratch
La clase `PerceptronScratch` incluye:
- `learning_rate`, `n_epochs`, `shuffle`, `patience`, `tol_no_improve`.
- Early stopping por convergencia (errores=0) o paciencia en validación.
- Registro en `history`: errores, accuracy train, accuracy val, mejor época.
- Métodos: `fit`, `predict`, `decision_function`, `score`, `to_dict`, `from_dict`.
- Alias `Perceptron = PerceptronScratch` para compatibilidad.

## Flujo Recomendado en el Notebook
1. Cargar/descargar dataset Breast Cancer (o fallback sklearn).
2. Limpieza + escalado.
3. Entrenar perceptrón scratch con validación.
4. Visualizar errores vs época y accuracy.
5. Ajustar hiperparámetros (learning rate, épocas) con una mini cuadrícula.
6. Comparar con `Perceptron`, `SGDClassifier`, `LogisticRegression`.
7. Graficar curvas ROC y calcular AUC.
8. Validación cruzada (StratifiedKFold).
9. Persistir resultados a CSV / JSON.
10. Conclusiones.

## Uso Rápido del Script CLI
Desde la raíz del repositorio (asumiendo venv activo):

```powershell
python .\semana-05-perceptron\train_perceptron_scratch.py --lr 0.005 --epochs 80 --patience 10 --verbose
```

Argumentos principales:
- `--lr`: learning rate (default 0.01)
- `--epochs`: máximo de épocas (default 50)
- `--patience`: activa early stopping basado en validación
- `--no-shuffle`: desactiva barajado por época
- `--no-convergence-stop`: evita detener cuando errores=0 (para seguir registrando)

Genera en `artifacts/`:
- `model_perceptron.json`
- `metrics.json`
- `training_curve.csv`

## Cargar un Modelo Guardado
```python
import json
from perceptron_lab.perceptron.perceptron import PerceptronScratch

with open('semana-05-perceptron/artifacts/model_perceptron.json','r',encoding='utf-8') as f:
   data = json.load(f)
model = PerceptronScratch.from_dict(data)
```

## Métricas Clave
- `history.errors`: número de actualizaciones (indicador de no linealidad / dificultad).
- `history.train_accuracy` y `history.val_accuracy`: seguimiento del aprendizaje.
- `best_val_accuracy` / `best_epoch`: almacenadas si se usa validación.

## Buenas Prácticas
- Usa semillas (`random_state`) para reproducibilidad.
- Documenta supuestos (p.ej. escalado previo obligatorio).
- Separa claramente datos de entrenamiento vs validación vs prueba.
- Guarda siempre artefactos con versión o hash si extiendes el flujo.

## Informe LaTeX
El documento de la semana debe referenciar:
- Arquitectura de la implementación scratch.
- Comparación cuantitativa con sklearn.
- Discusión sobre limitaciones del perceptrón.

## Próximos Pasos (Ideas de Extensión)
- Añadir regularización L2 en versión scratch.
- Implementar perceptrón multiclase (uno-vs-rest).
- Integrar pipelines con transformaciones personalizadas.
- Calibración de probabilidades (no trivial con perceptrón estándar).

---
Última actualización: generada automáticamente para consolidar la semana 05.
