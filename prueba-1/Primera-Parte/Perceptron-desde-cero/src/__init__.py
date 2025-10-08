"""
Paquete de Implementación del Perceptrón desde Cero
===================================================

Este paquete contiene una implementación completa del algoritmo de perceptrón
sin usar librerías de machine learning (solo NumPy).

Módulos:
--------
- perceptron: Implementación de la clase Perceptron
- data_preprocessing: Funciones para cargar y preprocesar datos
- visualization: Funciones para visualizar resultados

Autor: INFB6052 - Herramientas para Ciencia de Datos
Fecha: Octubre 2025
"""

from .perceptron import Perceptron, create_confusion_matrix
from .data_preprocessing import prepare_iris_data, load_iris_dataset
from .visualization import (plot_data_scatter, plot_decision_boundary, 
                            plot_error_evolution, plot_decision_line_manual)

__version__ = '1.0.0'
__all__ = [
    'Perceptron',
    'create_confusion_matrix',
    'prepare_iris_data',
    'load_iris_dataset',
    'plot_data_scatter',
    'plot_decision_boundary',
    'plot_error_evolution',
    'plot_decision_line_manual'
]
