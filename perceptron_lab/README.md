# Laboratorio: Implementación de un Perceptrón

Asignatura: INFB6052 – Herramientas para la Ciencia de Datos (UTEM)

## Objetivo
Implementar un perceptrón simple para aprender funciones lógicamente linealmente separables (AND / OR) y documentar el proceso usando Git y GitHub.

## Estructura del Proyecto
```
perceptron_lab/
├── data/
│   └── logic_gates.csv
├── notebooks/
│   └── 01_perceptron.ipynb
├── perceptron/
│   ├── __init__.py
│   └── perceptron.py
├── tests/
│   └── test_perceptron.py (se creará)
├── train_perceptron.py
├── requirements.txt
└── README.md
```

## Entorno Virtual (Windows PowerShell)
```powershell
python -m venv venv
# Activar
./venv/Scripts/Activate.ps1
# Actualizar pip (opcional)
pip install --upgrade pip
# Instalar dependencias
pip install -r requirements.txt
```
Para desactivar: `deactivate`

## Uso Rápido del Script CLI
```powershell
python train_perceptron.py --gate AND --epochs 15
python train_perceptron.py --gate OR --epochs 15
```

## Uso del Notebook
Abrir `notebooks/01_perceptron.ipynb` y ejecutar las celdas en orden.

## Flujo de Trabajo con Git
Asegúrate de estar dentro de la carpeta del proyecto `perceptron_lab`.

### 1. Inicializar repositorio
```powershell
git init
```
### 2. Configurar tu identidad (si no lo has hecho antes)
```powershell
git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@example.com"
```
### 3. Ver estado
```powershell
git status
```
### 4. Agregar archivos al staging
```powershell
git add .
```
### 5. Crear primer commit
```powershell
git commit -m "Inicial: estructura de perceptron"
```
### 6. Crear repositorio en GitHub
- Ve a GitHub y crea un repositorio vacío (sin README ni .gitignore). Ej: `perceptron-lab`

### 7. Agregar remoto y subir
```powershell
git remote add origin https://github.com/<tu_usuario>/perceptron-lab.git
git branch -M main
git push -u origin main
```
### 8. Hacer cambios y nuevos commits
Edita archivos y luego:
```powershell
git add perceptron/perceptron.py
git commit -m "Mejora: registro de accuracy"
```
### 9. Subir cambios
```powershell
git push
```
### 10. Clonar (desde otra carpeta/equipo)
```powershell
git clone https://github.com/<tu_usuario>/perceptron-lab.git
```

## Buenas Prácticas de Commits
- Mensajes cortos y descriptivos
- Un cambio lógico por commit
- Usar infinitivo: "Agregar", "Corregir", "Refactorizar"

## Criterios de Evaluación Cubiertos
| Criterio | Evidencia |
|----------|-----------|
| Código Funcional | `perceptron.py`, `train_perceptron.py` |
| Documentación | README, comentarios en código, notebook |
| Análisis de Resultados | Gráficos en notebook y explicación |
| Puntualidad | Subir antes de la fecha límite |

## Próximos Pasos (Opcional)
- Agregar soporte a tasa de aprendizaje variable
- Añadir early stopping
- Implementar un notebook para XOR mostrando no linealidad
- Crear un perceptrón multicapa (MLP) con `scikit-learn`

---
¡Éxito en tu laboratorio!
