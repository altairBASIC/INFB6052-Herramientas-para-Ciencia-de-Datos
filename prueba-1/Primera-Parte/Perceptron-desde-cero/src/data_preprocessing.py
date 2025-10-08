"""
Preprocesamiento de Datos para el Perceptrón
=============================================

Script para cargar, explorar y preprocesar el dataset Iris de Kaggle.
Selecciona 2 clases linealmente separables y 2 características.

Autor: INFB6052 - Herramientas para Ciencia de Datos
Fecha: Octubre 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_iris_dataset(filepath='data/iris.csv'):
    """
    Carga el dataset Iris desde un archivo CSV.
    
    El dataset Iris contiene 150 muestras de flores iris con 4 características:
    - sepal_length: Longitud del sépalo (cm)
    - sepal_width: Ancho del sépalo (cm)
    - petal_length: Longitud del pétalo (cm)
    - petal_width: Ancho del pétalo (cm)
    
    Y 3 clases:
    - Iris-setosa
    - Iris-versicolor
    - Iris-virginica
    
    Parámetros
    ----------
    filepath : str
        Ruta al archivo CSV del dataset Iris.
        
    Retorna
    -------
    DataFrame
        Dataset Iris completo.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset cargado exitosamente desde: {filepath}")
        print(f"Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}")
        print("\nDescarga el dataset de:")
        print("https://www.kaggle.com/datasets/uciml/iris/data")
        print("\nGuárdalo como 'data/iris.csv'")
        raise


def select_binary_classes(df, class1='Iris-setosa', class2='Iris-versicolor'):
    """
    Selecciona dos clases del dataset para clasificación binaria.
    
    Iris-setosa vs Iris-versicolor son linealmente separables.
    Iris-versicolor vs Iris-virginica NO son completamente separables.
    
    Parámetros
    ----------
    df : DataFrame
        Dataset Iris completo.
    class1 : str
        Primera clase a seleccionar.
    class2 : str
        Segunda clase a seleccionar.
        
    Retorna
    -------
    DataFrame
        Subset del dataset con solo las dos clases seleccionadas.
    """
    # Asumir que la columna de especies puede tener diferentes nombres
    species_col = None
    for col in ['species', 'Species', 'variety', 'class']:
        if col in df.columns:
            species_col = col
            break
    
    if species_col is None:
        # Si no se encuentra, asumir que es la última columna
        species_col = df.columns[-1]
    
    # Filtrar solo las dos clases
    df_binary = df[df[species_col].isin([class1, class2])].copy()
    
    print(f"\nClases seleccionadas: {class1} vs {class2}")
    print(f"Distribución de clases:")
    print(df_binary[species_col].value_counts())
    
    return df_binary


def select_features(df, feature1='petal_length', feature2='petal_width'):
    """
    Selecciona dos características para visualización 2D.
    
    Las características de pétalo (petal_length y petal_width) son las más
    discriminativas para separar Iris-setosa de Iris-versicolor.
    
    Parámetros
    ----------
    df : DataFrame
        Dataset Iris.
    feature1 : str
        Primera característica a seleccionar.
    feature2 : str
        Segunda característica a seleccionar.
        
    Retorna
    -------
    tuple
        (X, y, feature_names, class_names)
        X: array de características
        y: array de etiquetas (0 o 1)
        feature_names: nombres de las características
        class_names: nombres de las clases
    """
    # Normalizar nombres de columnas (manejar diferentes formatos)
    df.columns = df.columns.str.lower().str.replace('.', '_')
    
    # Encontrar columna de especies
    species_col = None
    for col in ['species', 'variety', 'class']:
        if col in df.columns:
            species_col = col
            break
    if species_col is None:
        species_col = df.columns[-1]
    
    # Verificar que las características existan
    available_features = [col for col in df.columns if col != species_col]
    
    # Intentar encontrar las características solicitadas
    if feature1 not in df.columns:
        # Buscar alternativas
        for feat in available_features:
            if 'petal' in feat and 'length' in feat:
                feature1 = feat
                break
    
    if feature2 not in df.columns:
        for feat in available_features:
            if 'petal' in feat and 'width' in feat:
                feature2 = feat
                break
    
    # Extraer características
    X = df[[feature1, feature2]].values
    
    # Convertir etiquetas de clase a binario (0 y 1)
    unique_classes = df[species_col].unique()
    class_names = {0: unique_classes[0], 1: unique_classes[1]}
    
    y = df[species_col].map({unique_classes[0]: 0, unique_classes[1]: 1}).values
    
    feature_names = [feature1, feature2]
    
    print(f"\nCaracterísticas seleccionadas: {feature_names}")
    print(f"Mapeo de clases: 0={class_names[0]}, 1={class_names[1]}")
    
    return X, y, feature_names, class_names


def normalize_features(X):
    """
    Normaliza las características usando estandarización (z-score).
    
    Para cada característica:
        X_norm = (X - mean) / std
    
    La normalización ayuda al perceptrón a converger más rápido y estable.
    
    Parámetros
    ----------
    X : array, shape (n_samples, n_features)
        Datos sin normalizar.
        
    Retorna
    -------
    tuple
        (X_normalized, mean, std)
        Datos normalizados y parámetros de normalización.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Evitar división por cero
    std[std == 0] = 1.0
    
    X_normalized = (X - mean) / std
    
    print(f"\nNormalización completada:")
    print(f"  Media: {mean}")
    print(f"  Desviación estándar: {std}")
    
    return X_normalized, mean, std


def train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Implementación manual sin usar sklearn.
    
    Parámetros
    ----------
    X : array, shape (n_samples, n_features)
        Características.
    y : array, shape (n_samples,)
        Etiquetas.
    test_size : float, default=0.3
        Proporción de datos para prueba (0.0 a 1.0).
    random_state : int, default=42
        Semilla para reproducibilidad.
        
    Retorna
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Configurar semilla
    np.random.seed(random_state)
    
    # Número total de muestras
    n_samples = len(X)
    
    # Índices aleatorios
    indices = np.random.permutation(n_samples)
    
    # Calcular punto de división
    test_set_size = int(n_samples * test_size)
    
    # Dividir índices
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    # Dividir datos
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"\nDivisión de datos:")
    print(f"  Conjunto de entrenamiento: {len(X_train)} muestras ({(1-test_size)*100:.0f}%)")
    print(f"  Conjunto de prueba: {len(X_test)} muestras ({test_size*100:.0f}%)")
    print(f"  Distribución en entrenamiento: Clase 0={np.sum(y_train==0)}, Clase 1={np.sum(y_train==1)}")
    print(f"  Distribución en prueba: Clase 0={np.sum(y_test==0)}, Clase 1={np.sum(y_test==1)}")
    
    return X_train, X_test, y_train, y_test


def prepare_iris_data(filepath='data/iris.csv', 
                      class1='Iris-setosa', 
                      class2='Iris-versicolor',
                      feature1='petal_length',
                      feature2='petal_width',
                      test_size=0.3,
                      random_state=42,
                      normalize=True):
    """
    Pipeline completo de preprocesamiento de datos.
    
    Ejecuta todos los pasos:
    1. Cargar dataset
    2. Seleccionar clases binarias
    3. Seleccionar características
    4. Normalizar (opcional)
    5. Dividir en train/test
    
    Parámetros
    ----------
    filepath : str
        Ruta al archivo CSV.
    class1, class2 : str
        Clases a seleccionar.
    feature1, feature2 : str
        Características a seleccionar.
    test_size : float
        Proporción de datos de prueba.
    random_state : int
        Semilla para reproducibilidad.
    normalize : bool
        Si True, normaliza las características.
        
    Retorna
    -------
    dict
        Diccionario con todos los datos preprocesados:
        - X_train, X_test, y_train, y_test
        - feature_names, class_names
        - mean, std (si normalize=True)
    """
    print("="*70)
    print("PREPROCESAMIENTO DE DATOS - DATASET IRIS")
    print("="*70)
    
    # 1. Cargar dataset
    df = load_iris_dataset(filepath)
    
    # 2. Seleccionar clases binarias
    df_binary = select_binary_classes(df, class1, class2)
    
    # 3. Seleccionar características
    X, y, feature_names, class_names = select_features(df_binary, feature1, feature2)
    
    # 4. Normalizar (opcional)
    mean, std = None, None
    if normalize:
        X, mean, std = normalize_features(X)
    
    # 5. Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print("\n" + "="*70)
    print("PREPROCESAMIENTO COMPLETADO")
    print("="*70)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'class_names': class_names,
        'mean': mean,
        'std': std,
        'normalized': normalize
    }


if __name__ == "__main__":
    # Ejemplo de uso
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
    
    print("\nDatos preparados:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {value}")
