"""
Script Principal de Entrenamiento del Perceptrón
================================================

Este script ejecuta el pipeline completo de entrenamiento del perceptrón:
1. Carga y preprocesa datos del dataset Iris
2. Entrena el perceptrón desde cero
3. Evalúa el modelo
4. Genera visualizaciones
5. Guarda resultados en artifacts/

Uso:
----
python train_perceptron.py

Autor: INFB6052 - Herramientas para Ciencia de Datos
Fecha: Octubre 2025
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.perceptron import Perceptron, create_confusion_matrix
from src.data_preprocessing import prepare_iris_data
from src.visualization import (plot_data_scatter, plot_decision_boundary,
                               plot_error_evolution, plot_decision_line_manual,
                               plot_training_summary)


def main():
    """
    Función principal que ejecuta el pipeline completo.
    """
    print("\n" + "="*80)
    print(" PERCEPTRÓN DESDE CERO - CLASIFICACIÓN BINARIA CON DATASET IRIS")
    print("="*80 + "\n")
    
    # ==========================================================================
    # 1. PREPARACIÓN DE DATOS
    # ==========================================================================
    print("\n[1/5] PREPARACIÓN DE DATOS")
    print("-" * 80)
    
    # Preparar datos
    data = prepare_iris_data(
        filepath='data/iris.csv',
        class1='Iris-setosa',
        class2='Iris-versicolor',
        feature1='petal_length',
        feature2='petal_width',
        test_size=0.3,
        random_state=42,
        normalize=True
    )
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    class_names = data['class_names']
    
    # ==========================================================================
    # 2. VISUALIZACIÓN INICIAL DE DATOS
    # ==========================================================================
    print("\n[2/5] VISUALIZACIÓN INICIAL DE DATOS")
    print("-" * 80)
    
    # Graficar datos de entrenamiento
    fig1 = plot_data_scatter(
        X_train, y_train, feature_names, class_names,
        title="Datos de Entrenamiento - Dataset Iris (2 clases, 2 características)"
    )
    fig1.savefig('artifacts/01_datos_entrenamiento.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: artifacts/01_datos_entrenamiento.png")
    
    # Graficar con línea de decisión estimativa
    fig2 = plot_decision_line_manual(
        X_train, y_train, feature_names, class_names,
        title="Separabilidad Lineal - Línea de Decisión Estimativa"
    )
    fig2.savefig('artifacts/02_linea_decision_estimativa.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: artifacts/02_linea_decision_estimativa.png")
    
    plt.close('all')
    
    # ==========================================================================
    # 3. ENTRENAMIENTO DEL PERCEPTRÓN
    # ==========================================================================
    print("\n[3/5] ENTRENAMIENTO DEL PERCEPTRÓN")
    print("-" * 80)
    
    # Crear y entrenar modelo
    perceptron = Perceptron(
        learning_rate=0.01,
        n_iterations=100,
        random_state=42
    )
    
    perceptron.fit(X_train, y_train, verbose=True)
    
    # ==========================================================================
    # 4. EVALUACIÓN DEL MODELO
    # ==========================================================================
    print("\n[4/5] EVALUACIÓN DEL MODELO")
    print("-" * 80)
    
    # Predicciones
    y_train_pred = perceptron.predict(X_train)
    y_test_pred = perceptron.predict(X_test)
    
    # Métricas de entrenamiento
    train_metrics = create_confusion_matrix(y_train, y_train_pred)
    print("\nMétricas en conjunto de ENTRENAMIENTO:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1-Score:  {train_metrics['f1_score']:.4f}")
    print(f"\n  Matriz de Confusión:")
    print(f"    TP={train_metrics['true_positives']}, "
          f"TN={train_metrics['true_negatives']}, "
          f"FP={train_metrics['false_positives']}, "
          f"FN={train_metrics['false_negatives']}")
    
    # Métricas de prueba
    test_metrics = create_confusion_matrix(y_test, y_test_pred)
    print("\nMétricas en conjunto de PRUEBA:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"\n  Matriz de Confusión:")
    print(f"    TP={test_metrics['true_positives']}, "
          f"TN={test_metrics['true_negatives']}, "
          f"FP={test_metrics['false_positives']}, "
          f"FN={test_metrics['false_negatives']}")
    
    # ==========================================================================
    # 5. VISUALIZACIONES FINALES
    # ==========================================================================
    print("\n[5/5] GENERACIÓN DE VISUALIZACIONES")
    print("-" * 80)
    
    # Frontera de decisión en entrenamiento
    fig3 = plot_decision_boundary(
        X_train, y_train, perceptron, feature_names, class_names,
        title="Frontera de Decisión - Conjunto de Entrenamiento"
    )
    fig3.savefig('artifacts/03_frontera_decision_train.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: artifacts/03_frontera_decision_train.png")
    
    # Frontera de decisión en prueba
    fig4 = plot_decision_boundary(
        X_test, y_test, perceptron, feature_names, class_names,
        title="Frontera de Decisión - Conjunto de Prueba"
    )
    fig4.savefig('artifacts/04_frontera_decision_test.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: artifacts/04_frontera_decision_test.png")
    
    # Evolución del error
    fig5 = plot_error_evolution(
        perceptron.errors_,
        title="Evolución del Error durante Entrenamiento (Convergencia)"
    )
    fig5.savefig('artifacts/05_evolucion_error.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: artifacts/05_evolucion_error.png")
    
    # Panel de resumen
    fig6 = plot_training_summary(
        perceptron, X_train, y_train, X_test, y_test,
        feature_names, class_names
    )
    fig6.savefig('artifacts/06_resumen_completo.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: artifacts/06_resumen_completo.png")
    
    plt.close('all')
    
    # ==========================================================================
    # 6. GUARDAR RESULTADOS
    # ==========================================================================
    print("\n[6/6] GUARDANDO RESULTADOS")
    print("-" * 80)
    
    # Guardar parámetros del modelo
    model_params = perceptron.get_params()
    
    results = {
        'model_parameters': {
            'learning_rate': perceptron.learning_rate,
            'n_iterations': perceptron.n_iterations,
            'random_state': perceptron.random_state
        },
        'learned_parameters': {
            'weights': model_params['weights'].tolist(),
            'bias': float(model_params['bias']),
            'converged': model_params['converged'],
            'total_epochs': model_params['total_epochs']
        },
        'training_history': {
            'errors_per_epoch': model_params['errors_per_epoch']
        },
        'performance': {
            'train': {
                'accuracy': float(train_metrics['accuracy']),
                'precision': float(train_metrics['precision']),
                'recall': float(train_metrics['recall']),
                'f1_score': float(train_metrics['f1_score'])
            },
            'test': {
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1_score': float(test_metrics['f1_score'])
            }
        },
        'dataset_info': {
            'classes': {str(k): str(v) for k, v in class_names.items()},
            'features': feature_names,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
    }
    
    # Guardar como JSON
    with open('artifacts/perceptron_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✓ Resultados guardados: artifacts/perceptron_results.json")
    
    # ==========================================================================
    # RESUMEN FINAL
    # ==========================================================================
    print("\n" + "="*80)
    print(" RESUMEN FINAL DEL ENTRENAMIENTO")
    print("="*80)
    print(f"\n✓ Modelo: Perceptrón (implementación desde cero)")
    print(f"✓ Dataset: Iris ({class_names[0]} vs {class_names[1]})")
    print(f"✓ Características: {feature_names}")
    print(f"✓ Convergencia: {'SÍ' if model_params['converged'] else 'NO'} "
          f"(época {model_params['total_epochs']})")
    print(f"✓ Accuracy (train): {train_metrics['accuracy']*100:.2f}%")
    print(f"✓ Accuracy (test):  {test_metrics['accuracy']*100:.2f}%")
    print(f"\n✓ Pesos finales: {model_params['weights']}")
    print(f"✓ Bias final: {model_params['bias']:.6f}")
    print(f"\n✓ Visualizaciones generadas: 6 gráficos en artifacts/")
    print(f"✓ Resultados guardados en: artifacts/perceptron_results.json")
    print("\n" + "="*80)
    print(" ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Crear directorio artifacts si no existe
    Path('artifacts').mkdir(exist_ok=True)
    
    # Ejecutar pipeline
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPor favor, descarga el dataset Iris de:")
        print("https://www.kaggle.com/datasets/uciml/iris/data")
        print("\nGuárdalo como: data/iris.csv")
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
