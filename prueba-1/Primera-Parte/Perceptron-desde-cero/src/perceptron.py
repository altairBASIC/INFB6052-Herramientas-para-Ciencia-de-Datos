"""
Implementación de Perceptrón desde Cero
========================================

Implementación del algoritmo de perceptrón sin usar librerías de machine learning.
Solo se utiliza NumPy para operaciones matemáticas básicas.

Autor: INFB6052 - Herramientas para Ciencia de Datos
Fecha: Octubre 2025
"""

import numpy as np


class Perceptron:
    """
    Implementación del algoritmo de Perceptrón de Rosenblatt (1958).
    
    El perceptrón es un clasificador binario lineal que aprende una frontera de
    decisión para separar dos clases linealmente separables.
    
    Parámetros
    ----------
    learning_rate : float, default=0.01
        Tasa de aprendizaje (eta) para la actualización de pesos.
        Valores típicos: 0.001 - 0.1
        
    n_iterations : int, default=100
        Número máximo de épocas de entrenamiento.
        
    random_state : int, default=None
        Semilla para la generación de números aleatorios.
        Permite reproducibilidad de resultados.
        
    Atributos
    ---------
    weights_ : array, shape (n_features,)
        Pesos aprendidos después del entrenamiento.
        
    bias_ : float
        Término de sesgo (bias) aprendido.
        
    errors_ : list
        Número de clasificaciones incorrectas en cada época.
        Útil para visualizar la convergencia del algoritmo.
        
    Notas
    -----
    El perceptrón solo converge si los datos son linealmente separables.
    La regla de actualización es:
        w = w + eta * (y_true - y_pred) * x
        b = b + eta * (y_true - y_pred)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=100, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        # Atributos que se aprenden durante el entrenamiento
        self.weights_ = None
        self.bias_ = None
        self.errors_ = []
        
    def _initialize_weights(self, n_features):
        """
        Inicializa los pesos y el bias.
        
        Los pesos se inicializan con valores aleatorios pequeños de una
        distribución normal con media 0 y desviación estándar 0.01.
        El bias se inicializa en 0.
        
        Parámetros
        ----------
        n_features : int
            Número de características en los datos de entrada.
        """
        rng = np.random.RandomState(self.random_state)
        
        # Inicialización de pesos con valores pequeños aleatorios
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        
        # Inicialización del bias en 0
        self.bias_ = 0.0
        
        print(f"Inicialización de pesos:")
        print(f"  Pesos: {self.weights_}")
        print(f"  Bias: {self.bias_}")
        
    def _activation_function(self, z):
        """
        Función de activación (función escalón).
        
        Retorna 1 si z >= 0, sino retorna 0.
        
        Parámetros
        ----------
        z : float o array
            Resultado de la combinación lineal (net input).
            
        Retorna
        -------
        int o array
            Clase predicha (0 o 1).
        """
        return np.where(z >= 0.0, 1, 0)
    
    def _forward_pass(self, X):
        """
        Forward pass: calcula la salida del perceptrón.
        
        Realiza:
        1. Combinación lineal: z = w^T * x + b
        2. Función de activación: y = step(z)
        
        Parámetros
        ----------
        X : array, shape (n_samples, n_features)
            Datos de entrada.
            
        Retorna
        -------
        array, shape (n_samples,)
            Predicciones del modelo (0 o 1).
        """
        # Combinación lineal (net input)
        z = np.dot(X, self.weights_) + self.bias_
        
        # Función de activación
        predictions = self._activation_function(z)
        
        return predictions
    
    def fit(self, X, y, verbose=True):
        """
        Entrena el perceptrón usando la regla de aprendizaje del perceptrón.
        
        Proceso de entrenamiento:
        1. Inicializa pesos y bias
        2. Para cada época:
            a. Para cada muestra:
                - Forward pass: calcular predicción
                - Calcular error: (y_true - y_pred)
                - Actualizar pesos: w = w + eta * error * x
                - Actualizar bias: b = b + eta * error
            b. Contar errores totales en la época
        3. Repetir hasta convergencia o máximo de iteraciones
        
        Parámetros
        ----------
        X : array, shape (n_samples, n_features)
            Datos de entrenamiento.
            
        y : array, shape (n_samples,)
            Etiquetas objetivo (0 o 1).
            
        verbose : bool, default=True
            Si True, imprime información de progreso.
            
        Retorna
        -------
        self : object
            Retorna la instancia del perceptrón entrenado.
        """
        # Validar entrada
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Inicializar pesos
        self._initialize_weights(n_features)
        
        # Limpiar historial de errores
        self.errors_ = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"INICIO DEL ENTRENAMIENTO")
            print(f"{'='*60}")
            print(f"Parámetros:")
            print(f"  Tasa de aprendizaje: {self.learning_rate}")
            print(f"  Máximo de iteraciones: {self.n_iterations}")
            print(f"  Número de muestras: {n_samples}")
            print(f"  Número de características: {n_features}")
            print(f"{'='*60}\n")
        
        # Proceso de entrenamiento por épocas
        for epoch in range(self.n_iterations):
            errors = 0
            
            # Iterar sobre cada muestra
            for xi, target in zip(X, y):
                # Forward pass: obtener predicción
                prediction = self._forward_pass(xi.reshape(1, -1))[0]
                
                # Calcular error
                error = target - prediction
                
                # Regla de actualización del perceptrón
                # Solo se actualiza si hay error (error != 0)
                if error != 0:
                    # Actualizar pesos: w = w + eta * error * x
                    update = self.learning_rate * error
                    self.weights_ += update * xi
                    
                    # Actualizar bias: b = b + eta * error
                    self.bias_ += update
                    
                    errors += 1
            
            # Guardar número de errores en esta época
            self.errors_.append(errors)
            
            if verbose and (epoch % 10 == 0 or epoch == self.n_iterations - 1):
                print(f"Época {epoch + 1:3d}/{self.n_iterations} - "
                      f"Errores: {errors:3d} - "
                      f"Accuracy: {(n_samples - errors) / n_samples * 100:.2f}%")
            
            # Criterio de parada: si no hay errores, el algoritmo convergió
            if errors == 0:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"CONVERGENCIA ALCANZADA en época {epoch + 1}")
                    print(f"{'='*60}")
                break
        
        if verbose:
            print(f"\nPesos finales:")
            print(f"  Pesos: {self.weights_}")
            print(f"  Bias: {self.bias_}")
            print(f"\nNúmero total de actualizaciones: {sum(self.errors_)}")
        
        return self
    
    def predict(self, X):
        """
        Predice las etiquetas de clase para las muestras en X.
        
        Parámetros
        ----------
        X : array, shape (n_samples, n_features)
            Muestras para predecir.
            
        Retorna
        -------
        array, shape (n_samples,)
            Etiquetas predichas (0 o 1).
        """
        if self.weights_ is None:
            raise ValueError("El perceptrón no ha sido entrenado. "
                           "Llama a fit() primero.")
        
        return self._forward_pass(np.array(X))
    
    def score(self, X, y):
        """
        Calcula la precisión (accuracy) del modelo.
        
        Parámetros
        ----------
        X : array, shape (n_samples, n_features)
            Datos de prueba.
            
        y : array, shape (n_samples,)
            Etiquetas verdaderas.
            
        Retorna
        -------
        float
            Accuracy: proporción de predicciones correctas.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def get_params(self):
        """
        Obtiene los parámetros aprendidos del modelo.
        
        Retorna
        -------
        dict
            Diccionario con pesos, bias y errores por época.
        """
        return {
            'weights': self.weights_,
            'bias': self.bias_,
            'errors_per_epoch': self.errors_,
            'converged': self.errors_[-1] == 0 if self.errors_ else False,
            'total_epochs': len(self.errors_)
        }
    
    def decision_function(self, X):
        """
        Calcula el valor de la función de decisión (antes de la activación).
        
        Útil para trazar la frontera de decisión.
        
        Parámetros
        ----------
        X : array, shape (n_samples, n_features)
            Muestras de entrada.
            
        Retorna
        -------
        array, shape (n_samples,)
            Valores de la función de decisión: w^T * x + b
        """
        if self.weights_ is None:
            raise ValueError("El perceptrón no ha sido entrenado.")
        
        return np.dot(X, self.weights_) + self.bias_


def create_confusion_matrix(y_true, y_pred):
    """
    Crea una matriz de confusión manual (sin sklearn).
    
    Parámetros
    ----------
    y_true : array
        Etiquetas verdaderas.
    y_pred : array
        Etiquetas predichas.
        
    Retorna
    -------
    dict
        Diccionario con TP, TN, FP, FN, precision, recall, f1-score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcular métricas
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    # Evitar división por cero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }
