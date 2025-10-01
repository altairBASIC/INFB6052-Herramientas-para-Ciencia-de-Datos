"""Script CLI para entrenar el Perceptrón scratch (Semana 05).

Uso básico (PowerShell desde la raíz del repo):
    .\.venv\Scripts\python.exe .\semana-05-perceptron\train_perceptron_scratch.py --epochs 80 --lr 0.005

Genera artefactos en semana-05-perceptron/artifacts/:
    - model_perceptron.json : pesos y metadatos
    - metrics.json          : métricas finales sobre test
    - training_curve.csv    : evolución errores/accuracy

Dataset: Breast Cancer (sklearn) para reproducibilidad rápida.
"""
from __future__ import annotations
import json, csv, argparse, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Intento de import directo del paquete local; añadir fallback dinámico de ruta.
try:
    from perceptron_lab.perceptron.perceptron import PerceptronScratch
except ModuleNotFoundError:
    import sys
    ROOT = Path(__file__).resolve().parent.parent  # directorio raíz del repo (asumiendo estructura actual)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from perceptron_lab.perceptron.perceptron import PerceptronScratch  # reintento

ART_DIR = Path(__file__).parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)


def prepare_data(test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign (mapearemos a 1 maligno para consistencia invertida si queremos)
    # Opcional: invertir etiquetas para que 1 signifique clase positiva (malignant ya es 0). Mantendremos original.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), stratify=y_temp, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train(args):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(
        test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )

    model = PerceptronScratch(
        learning_rate=args.lr,
        n_epochs=args.epochs,
        shuffle=not args.no_shuffle,
        random_state=args.seed,
        patience=args.patience,
        tol_no_improve=1e-5,
        verbose=args.verbose,
    )

    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train, X_val=X_val, y_val=y_val, early_stop_on_convergence=not args.no_convergence_stop
    )
    train_time = time.perf_counter() - t0

    y_pred_test = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test)),
        "recall": float(recall_score(y_test, y_pred_test)),
        "f1": float(f1_score(y_test, y_pred_test)),
        "epochs_run": len(history.errors),
        "stopped_early": history.stopped_early,
        "best_val_accuracy": history.best_val_accuracy,
        "best_epoch": history.best_epoch,
        "train_time_sec": train_time,
        "learning_rate": args.lr,
    }

    # Guardar artefactos
    model_json = ART_DIR / "model_perceptron.json"
    metrics_json = ART_DIR / "metrics.json"
    curve_csv = ART_DIR / "training_curve.csv"

    with open(model_json, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "errors", "train_accuracy", "val_accuracy"])  # val_accuracy puede ser vacío
        for i, err in enumerate(history.errors):
            val_acc = history.val_accuracy[i] if i < len(history.val_accuracy) else ""
            writer.writerow([i + 1, err, history.train_accuracy[i], val_acc])

    if args.verbose:
        print("Modelo guardado en:", model_json)
        print("Métricas guardadas en:", metrics_json)
        print("Curva de entrenamiento:", curve_csv)
        print("Metrics:", metrics)


def parse_args():
    p = argparse.ArgumentParser(description="Entrenar Perceptrón scratch (Semana 05)")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--epochs", type=int, default=50, help="Número máximo de épocas")
    p.add_argument("--patience", type=int, default=None, help="Paciencia validación (early stopping)")
    p.add_argument("--test-size", type=float, default=0.15, help="Proporción test")
    p.add_argument("--val-size", type=float, default=0.15, help="Proporción validación")
    p.add_argument("--seed", type=int, default=42, help="Semilla RNG")
    p.add_argument("--no-shuffle", action="store_true", help="Desactivar shuffle por época")
    p.add_argument("--no-convergence-stop", action="store_true", help="No detener cuando errores=0")
    p.add_argument("--verbose", action="store_true", help="Mostrar progreso")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
