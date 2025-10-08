"""
Script Helper para Descargar el Dataset Iris
=============================================

Este script descarga automáticamente el dataset Iris desde sklearn
y lo guarda en formato CSV.

Uso:
----
python download_iris.py

Autor: INFB6052 - Herramientas para Ciencia de Datos
Fecha: Octubre 2025
"""

import sys
from pathlib import Path

try:
    from sklearn import datasets
    import pandas as pd
    
    print("="*70)
    print(" DESCARGA DEL DATASET IRIS")
    print("="*70)
    
    # Cargar dataset Iris desde sklearn
    print("\n[1/3] Cargando dataset Iris desde scikit-learn...")
    iris = datasets.load_iris()
    
    # Crear DataFrame
    print("[2/3] Creando DataFrame...")
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({
        0: 'Iris-setosa', 
        1: 'Iris-versicolor', 
        2: 'Iris-virginica'
    })
    
    # Renombrar columnas para consistencia
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    # Crear directorio data si no existe
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Guardar como CSV
    output_path = data_dir / 'iris.csv'
    print(f"[3/3] Guardando en {output_path}...")
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print(" DATASET DESCARGADO EXITOSAMENTE")
    print("="*70)
    print(f"\nArchivo: {output_path}")
    print(f"Dimensiones: {df.shape}")
    print(f"\nPrimeras 5 filas:")
    print(df.head())
    print(f"\nDistribución de especies:")
    print(df['species'].value_counts())
    print("\n" + "="*70)
    print("Ahora puedes ejecutar: python train_perceptron.py")
    print("="*70 + "\n")
    
except ImportError as e:
    print("\n" + "="*70)
    print(" ERROR: Dependencias no encontradas")
    print("="*70)
    print("\nEste script requiere scikit-learn y pandas.")
    print("\nInstala con:")
    print("  pip install scikit-learn pandas")
    print("\nO descarga manualmente desde:")
    print("  https://www.kaggle.com/datasets/uciml/iris/data")
    print("\nY guarda como: data/iris.csv")
    print("="*70 + "\n")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ ERROR INESPERADO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
