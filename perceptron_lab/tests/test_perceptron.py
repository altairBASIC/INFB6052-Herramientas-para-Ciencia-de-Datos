import numpy as np
from perceptron import Perceptron

def test_and_gate_convergence():
    # Datos AND
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    model = Perceptron(n_inputs=2, learning_rate=0.1, random_state=0)
    history = model.fit(X, y, epochs=15)
    assert model.score(X, y) == 1.0, "El perceptrón debería aprender la compuerta AND"
    # Debe haber llegado a cero errores al final
    assert history.errors[-1] == 0, "Al final debe no haber errores"
