# Guía para Notebook del Perceptrón (Semana 05)

## 1. Introducción
- Explicar brevemente qué es el perceptrón y qué problema resuelve.
- Mencionar que aprende fronteras lineales.

## 2. Dataset de Ejemplo
```python
import numpy as np
# AND
df_X = np.array([[0,0],[0,1],[1,0],[1,1]])
df_y = np.array([0,0,0,1])
```
(Opcional: usar OR, NAND, dataset sintético linealmente separable.)

## 3. Importar la Clase
```python
from perceptron import Perceptron
```

## 4. Entrenamiento
```python
model = Perceptron(n_inputs=2, learning_rate=0.1, random_state=0)
history = model.fit(df_X, df_y, epochs=20)
```

## 5. Métricas y Evolución
```python
import matplotlib.pyplot as plt
plt.plot(history.errors)
plt.title('Errores por Época')
plt.xlabel('Época')
plt.ylabel('# Errores')
plt.show()

plt.plot(history.accuracy)
plt.title('Accuracy por Época')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.ylim(0,1.05)
plt.show()
```

## 6. Frontera de Decisión (2D)
```python
import numpy as np
import matplotlib.pyplot as plt

# Crear malla
gx1, gx2 = np.meshgrid(np.linspace(-0.5,1.5,200), np.linspace(-0.5,1.5,200))
grid = np.c_[gx1.ravel(), gx2.ravel()]
z = model.predict(grid).reshape(gx1.shape)
plt.contourf(gx1, gx2, z, alpha=0.3, cmap='coolwarm')
plt.scatter(df_X[:,0], df_X[:,1], c=df_y, cmap='coolwarm', edgecolor='k')
plt.title('Frontera de Decisión')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

## 7. Evolución de Pesos
```python
import numpy as np
weights_array = np.array(history.weights)
plt.plot(weights_array[:,0], label='bias')
plt.plot(weights_array[:,1], label='w1')
plt.plot(weights_array[:,2], label='w2')
plt.title('Evolución de Pesos')
plt.xlabel('Época')
plt.legend()
plt.show()
```

## 8. Conclusiones
- ¿Cuántas épocas hasta converger?
- ¿Qué patrón muestran los pesos?
- ¿El accuracy llegó a 1.0?

## 9. Próximos Pasos
- Extender a perceptrón multicapa.
- Probar datasets no linealmente separables.
