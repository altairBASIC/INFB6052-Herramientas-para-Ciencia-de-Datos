"""Perceptrón (implementación educativa desde cero) con características avanzadas.

Características incluidas respecto a la versión experimental anterior:
- Inicialización reproducible.
- Registro detallado por época: errores (actualizaciones), accuracy entrenamiento y validación opcional.
- Early stopping configurable por convergencia (cero errores) o paciencia en validación.
- Opción de barajar (shuffle) los patrones cada época.
- Métodos estándar: predict, score, decision_function.

Notas didácticas:
Este modelo implementa la regla clásica de actualización del Perceptrón:
    w := w + eta * (y - y_hat) * x
    b := b + eta * (y - y_hat)
Usa una función de activación escalón en el hiperplano lineal. No optimiza una función
de pérdida diferenciable (a diferencia de regresión logística o SVM); la convergencia sólo
está garantizada si los datos son linealmente separables.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TrainingHistory:
    errors: List[int] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    # weights almacena únicamente el vector de pesos w (sin bias)
    weights: List[np.ndarray] = field(default_factory=list)
    # biases almacena la evolución del sesgo b por época
    biases: List[float] = field(default_factory=list)
    stopped_early: bool = False
    best_val_accuracy: Optional[float] = None
    best_epoch: Optional[int] = None


class PerceptronScratch:
    """Perceptrón binario.

    Parámetros
    ----------
    learning_rate : float, default=0.01
        Tasa de aprendizaje.
    n_epochs : int, default=50
        Número máximo de épocas.
    shuffle : bool, default=True
        Si baraja los patrones cada época (favorece exploración).
    random_state : int | None, default=42
        Semilla para reproducibilidad.
    patience : int | None, default=None
        Número de épocas sin mejora de validación antes de detener (si se da X_val, y_val).
    tol_no_improve : float, default=1e-6
        Umbral mínimo de mejora relativa para considerar que hubo mejora.
    verbose : bool, default=False
        Si imprime progreso por época.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        shuffle: bool = True,
        random_state: int | None = 42,
        patience: int | None = None,
        tol_no_improve: float = 1e-6,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.patience = patience
        self.tol_no_improve = tol_no_improve
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0
        self.history = TrainingHistory()

    # ------------------------------- Utilidades internas -------------------------------
    def _initialize(self, n_features: int) -> None:
        self.w_ = self.rng.normal(0.0, 0.01, size=n_features)
        self.b_ = 0.0

    def _shuffle(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = self.rng.permutation(len(X))
        return X[idx], y[idx]

    # ------------------------------- API pública -------------------------------
    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_  # type: ignore[arg-type]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.net_input(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)

    def predict_single(self, x: np.ndarray) -> int:
        return int(np.dot(x, self.w_) + self.b_ >= 0)  # type: ignore[arg-type]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stop_on_convergence: bool = True,
    ) -> TrainingHistory:
        """Entrena el modelo.

        Parámetros
        ----------
        X, y : ndarray
            Datos y etiquetas (0/1).
        X_val, y_val : ndarray | None
            Conjunto de validación opcional.
        early_stop_on_convergence : bool
            Si se detiene cuando no hay errores en una época.
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        n_samples, n_features = X.shape
        self._initialize(n_features)
        self.history = TrainingHistory()

        best_val = -np.inf
        epochs_no_improve = 0

        for epoch in range(self.n_epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            errors = 0
            for xi, target in zip(X, y):
                pred = self.predict_single(xi)
                err = target - pred
                if err != 0:
                    # Regla de actualización
                    self.w_ += self.learning_rate * err * xi  # type: ignore[operator]
                    self.b_ += self.learning_rate * err
                    errors += 1

            # Métricas entrenamiento
            train_acc = self.score(X, y)
            self.history.errors.append(errors)
            self.history.train_accuracy.append(train_acc)
            # Guardar snapshot de pesos y bias
            self.history.weights.append(self.w_.copy())  # type: ignore[union-attr]
            self.history.biases.append(float(self.b_))

            # Validación
            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = self.score(X_val, y_val)
                self.history.val_accuracy.append(val_acc)
                # Early stopping por paciencia
                if val_acc > best_val + self.tol_no_improve:
                    best_val = val_acc
                    self.history.best_val_accuracy = val_acc
                    self.history.best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if self.patience is not None and epochs_no_improve >= self.patience:
                        if self.verbose:
                            print(f"Stop por paciencia en época {epoch+1}")
                        self.history.stopped_early = True
                        break

            if self.verbose:
                msg = f"Epoch {epoch+1}/{self.n_epochs} - errors={errors} train_acc={train_acc:.3f}"
                if val_acc is not None:
                    msg += f" val_acc={val_acc:.3f}"
                print(msg)

            if early_stop_on_convergence and errors == 0:
                if self.verbose:
                    print(f"Convergencia sin errores en época {epoch+1}")
                self.history.stopped_early = True
                break

        return self.history

    # ------------------------------- Exportación / Estado -------------------------------
    def get_params(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "patience": self.patience,
            "tol_no_improve": self.tol_no_improve,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.get_params(),
            "w": None if self.w_ is None else self.w_.tolist(),
            "b": self.b_,
            "history": {
                "errors": self.history.errors,
                "train_accuracy": self.history.train_accuracy,
                "val_accuracy": self.history.val_accuracy,
                "biases": self.history.biases,
                "stopped_early": self.history.stopped_early,
                "best_val_accuracy": self.history.best_val_accuracy,
                "best_epoch": self.history.best_epoch,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptronScratch":
        params = data.get("params", {})
        model = cls(**params)
        w = data.get("w")
        if w is not None:
            model.w_ = np.array(w, dtype=float)
        model.b_ = float(data.get("b", 0.0))
        return model


# Alias para mantener compatibilidad con el nombre anterior si algún import externo lo usa
Perceptron = PerceptronScratch

