"""
Funciones de Visualización para el Perceptrón
==============================================

Módulo con funciones para visualizar:
- Scatter plots de datos
- Frontera de decisión
- Evolución del error durante entrenamiento

Autor: INFB6052 - Herramientas para Ciencia de Datos
Fecha: Octubre 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_data_scatter(X, y, feature_names, class_names, title="Visualización de Datos"):
    """
    Crea un scatter plot de los datos con dos clases.
    
    Parámetros
    ----------
    X : array, shape (n_samples, 2)
        Características (solo 2D).
    y : array, shape (n_samples,)
        Etiquetas (0 o 1).
    feature_names : list
        Nombres de las características [feature1, feature2].
    class_names : dict
        Mapeo de etiquetas a nombres {0: 'clase0', 1: 'clase1'}.
    title : str
        Título del gráfico.
    """
    plt.figure(figsize=(10, 7))
    
    # Colores para cada clase
    colors = ['#FF6B6B', '#4ECDC4']
    markers = ['o', 's']
    
    # Graficar cada clase
    for idx, class_label in enumerate([0, 1]):
        plt.scatter(X[y == class_label, 0], 
                   X[y == class_label, 1],
                   c=colors[idx], 
                   marker=markers[idx],
                   s=100, 
                   alpha=0.7,
                   edgecolors='black',
                   linewidth=1,
                   label=f'{class_names[class_label]}')
    
    plt.xlabel(feature_names[0].replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.ylabel(feature_names[1].replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(title='Clase', fontsize=11, title_fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return plt.gcf()


def plot_decision_boundary(X, y, model, feature_names, class_names, 
                          title="Frontera de Decisión del Perceptrón",
                          resolution=0.02):
    """
    Visualiza la frontera de decisión aprendida por el perceptrón.
    
    Parámetros
    ----------
    X : array, shape (n_samples, 2)
        Datos de entrada.
    y : array, shape (n_samples,)
        Etiquetas verdaderas.
    model : Perceptron
        Modelo de perceptrón entrenado.
    feature_names : list
        Nombres de las características.
    class_names : dict
        Mapeo de etiquetas a nombres.
    title : str
        Título del gráfico.
    resolution : float
        Resolución de la malla para la frontera.
    """
    # Configurar figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Definir rango de la malla
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Crear malla de puntos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predecir la clase para cada punto de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Graficar regiones de decisión
    cmap_light = ListedColormap(['#FFAAAA', '#AADDDD'])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    
    # Graficar la línea de decisión (frontera)
    ax.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5])
    
    # Graficar los puntos de datos
    colors = ['#FF6B6B', '#4ECDC4']
    markers = ['o', 's']
    
    for idx, class_label in enumerate([0, 1]):
        ax.scatter(X[y == class_label, 0], 
                  X[y == class_label, 1],
                  c=colors[idx], 
                  marker=markers[idx],
                  s=100, 
                  alpha=0.8,
                  edgecolors='black',
                  linewidth=1.5,
                  label=f'{class_names[class_label]}')
    
    # Dibujar el vector de pesos (dirección perpendicular a la frontera)
    if model.weights_ is not None:
        # Punto central
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Vector de pesos normalizado
        w_norm = model.weights_ / np.linalg.norm(model.weights_)
        
        # Escalar para visualización
        scale = min(x_max - x_min, y_max - y_min) * 0.3
        
        ax.arrow(x_center, y_center, 
                w_norm[0] * scale, w_norm[1] * scale,
                head_width=0.1, head_length=0.1, 
                fc='green', ec='green', linewidth=2,
                label='Vector de Pesos (w)', alpha=0.7)
    
    ax.set_xlabel(feature_names[0].replace('_', ' ').title(), 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_names[1].replace('_', ' ').title(), 
                 fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_error_evolution(errors, title="Evolución del Error durante Entrenamiento"):
    """
    Grafica la evolución del número de errores por época.
    
    Parámetros
    ----------
    errors : list
        Lista con el número de errores en cada época.
    title : str
        Título del gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = np.arange(1, len(errors) + 1)
    
    # Línea de errores
    ax.plot(epochs, errors, marker='o', linewidth=2, markersize=6,
            color='#E74C3C', label='Errores por Época')
    
    # Línea de convergencia (errores = 0)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, 
              label='Convergencia (0 errores)', alpha=0.7)
    
    # Rellenar área bajo la curva
    ax.fill_between(epochs, errors, alpha=0.3, color='#E74C3C')
    
    ax.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Errores', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Anotaciones
    if len(errors) > 0:
        # Marcar primera época
        ax.annotate(f'Inicio: {errors[0]} errores',
                   xy=(1, errors[0]), xytext=(len(errors)*0.1, max(errors)*0.9),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, fontweight='bold')
        
        # Marcar última época
        ax.annotate(f'Final: {errors[-1]} errores',
                   xy=(len(errors), errors[-1]), 
                   xytext=(len(errors)*0.7, max(errors)*0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_decision_line_manual(X, y, feature_names, class_names, 
                              title="Datos con Línea de Decisión Estimativa"):
    """
    Grafica los datos con una línea de decisión dibujada manualmente/estimativamente.
    
    Esta función se usa ANTES del entrenamiento para mostrar una separación estimada.
    
    Parámetros
    ----------
    X : array, shape (n_samples, 2)
        Características.
    y : array, shape (n_samples,)
        Etiquetas.
    feature_names : list
        Nombres de características.
    class_names : dict
        Mapeo de clases.
    title : str
        Título del gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Graficar puntos
    colors = ['#FF6B6B', '#4ECDC4']
    markers = ['o', 's']
    
    for idx, class_label in enumerate([0, 1]):
        ax.scatter(X[y == class_label, 0], 
                  X[y == class_label, 1],
                  c=colors[idx], 
                  marker=markers[idx],
                  s=100, 
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=1,
                  label=f'{class_names[class_label]}')
    
    # Calcular línea de decisión estimativa (punto medio entre centroides)
    centroid_0 = X[y == 0].mean(axis=0)
    centroid_1 = X[y == 1].mean(axis=0)
    
    # Punto medio
    midpoint = (centroid_0 + centroid_1) / 2
    
    # Vector perpendicular (simplificado)
    direction = centroid_1 - centroid_0
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    
    # Crear línea
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    t = np.linspace(-10, 10, 100)
    
    line_x = midpoint[0] + t * perpendicular[0]
    line_y = midpoint[1] + t * perpendicular[1]
    
    # Filtrar puntos dentro del rango
    mask = (line_x >= x_min) & (line_x <= x_max)
    line_x = line_x[mask]
    line_y = line_y[mask]
    
    # Graficar línea estimativa
    ax.plot(line_x, line_y, 'g--', linewidth=2.5, 
           label='Línea de Decisión Estimativa', alpha=0.7)
    
    # Marcar centroides
    ax.scatter(*centroid_0, c='red', marker='X', s=300, 
              edgecolors='black', linewidth=2, 
              label=f'Centroide {class_names[0]}', zorder=5)
    ax.scatter(*centroid_1, c='blue', marker='X', s=300, 
              edgecolors='black', linewidth=2, 
              label=f'Centroide {class_names[1]}', zorder=5)
    
    ax.set_xlabel(feature_names[0].replace('_', ' ').title(), 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_names[1].replace('_', ' ').title(), 
                 fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, frameon=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_training_summary(model, X_train, y_train, X_test, y_test, 
                         feature_names, class_names):
    """
    Crea un panel de resumen con múltiples visualizaciones.
    
    Parámetros
    ----------
    model : Perceptron
        Modelo entrenado.
    X_train, y_train : arrays
        Datos de entrenamiento.
    X_test, y_test : arrays
        Datos de prueba.
    feature_names : list
        Nombres de características.
    class_names : dict
        Mapeo de clases.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Datos de entrenamiento con frontera
    ax1 = fig.add_subplot(gs[0, 0])
    plot_decision_boundary_ax(ax1, X_train, y_train, model, feature_names, class_names,
                             "Conjunto de Entrenamiento")
    
    # 2. Datos de prueba con frontera
    ax2 = fig.add_subplot(gs[0, 1])
    plot_decision_boundary_ax(ax2, X_test, y_test, model, feature_names, class_names,
                             "Conjunto de Prueba")
    
    # 3. Evolución de errores
    ax3 = fig.add_subplot(gs[1, :])
    epochs = np.arange(1, len(model.errors_) + 1)
    ax3.plot(epochs, model.errors_, marker='o', linewidth=2, markersize=6,
            color='#E74C3C')
    ax3.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.fill_between(epochs, model.errors_, alpha=0.3, color='#E74C3C')
    ax3.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Número de Errores', fontsize=12, fontweight='bold')
    ax3.set_title('Evolución del Error durante Entrenamiento', 
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    return fig


def plot_decision_boundary_ax(ax, X, y, model, feature_names, class_names, title):
    """
    Versión auxiliar de plot_decision_boundary para usar con subplots.
    """
    # Definir rango de la malla
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Crear malla
    resolution = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predecir
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Graficar
    cmap_light = ListedColormap(['#FFAAAA', '#AADDDD'])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    ax.contour(xx, yy, Z, colors='black', linewidths=2, levels=[0.5])
    
    # Puntos
    colors = ['#FF6B6B', '#4ECDC4']
    markers = ['o', 's']
    
    for idx, class_label in enumerate([0, 1]):
        ax.scatter(X[y == class_label, 0], 
                  X[y == class_label, 1],
                  c=colors[idx], 
                  marker=markers[idx],
                  s=80, 
                  alpha=0.8,
                  edgecolors='black',
                  linewidth=1,
                  label=f'{class_names[class_label]}')
    
    ax.set_xlabel(feature_names[0].replace('_', ' ').title(), fontweight='bold')
    ax.set_ylabel(feature_names[1].replace('_', ' ').title(), fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')


if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de visualización para el Perceptrón")
    print("Importa estas funciones en tu notebook o script principal")
