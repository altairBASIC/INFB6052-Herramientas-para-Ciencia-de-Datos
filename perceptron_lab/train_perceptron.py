"""Script CLI simple para entrenar el Perceptrón sobre compuertas lógicas.

Uso:
    python train_perceptron.py --gate AND --epochs 15
    python train_perceptron.py --gate OR --epochs 15
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from perceptron import Perceptron


def load_logic_gate(gate: str, path: str = "data/logic_gates.csv"):
    df = pd.read_csv(path)
    gate_df = df[df["gate"].str.upper() == gate.upper()].copy()
    X = gate_df[["x1", "x2"]].values.astype(float)
    y = gate_df["y"].values.astype(int)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Entrena un perceptrón sobre compuertas lógicas")
    parser.add_argument("--gate", type=str, default="AND", choices=["AND", "OR"], help="Compuerta lógica a usar")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=0.1, help="Tasa de aprendizaje")
    args = parser.parse_args()

    X, y = load_logic_gate(args.gate)

    model = Perceptron(n_inputs=X.shape[1], learning_rate=args.lr, random_state=42)
    history = model.fit(X, y, epochs=args.epochs)

    print(f"Entrenamiento terminado para {args.gate}")
    print("Errores por época:", history.errors)
    print("Accuracy final:", model.score(X, y))
    print("Pesos finales:", model.weights)


if __name__ == "__main__":
    main()
