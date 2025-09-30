# Semana 03 – Librerías de Python y EDA con Dataset Médico

## Objetivos
- Practicar manipulación de datos con Pandas y NumPy en un caso real.
- Realizar análisis exploratorio (EDA) básico guiado y con interpretaciones.
- Incorporar el dataset médico Breast Cancer Wisconsin (Diagnostic) para enriquecer el ejercicio.
- Identificar variables redundantes mediante correlaciones simples.
- Documentar hallazgos en un notebook claro y reproducible.

## Justificación del Dataset Médico
Se eligió intencionalmente un dataset real de cáncer de mama (**Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository) para pasar de ejemplos sintéticos a un escenario con significado aplicado:

**Beneficios de usar el dataset médico:**
- Problema binario real (Benigno vs Maligno) con ligera desproporción de clases.
- Variables morfológicas (radius, texture, perimeter, area, concavity, etc.) que permiten discutir interpretaciones clínicas básicas.
- Evaluación temprana de redundancia: varios descriptores de tamaño están altamente correlacionados.
- Fundamenta la importancia de métricas distintas de accuracy (en etapas posteriores) dada la naturaleza sensible del dominio.
- Facilita transición hacia un modelo supervisado ligero (e.g. Regresión Logística, SVM) en una etapa siguiente.

Notebook central: `eda_semana03.ipynb` – contiene interpretaciones textuales debajo de cada visualización para reforzar el aprendizaje.

## Estructura del Notebook (`eda_semana03.ipynb`)
1. Introducción y objetivos
2. Descarga y carga del dataset
3. Exploración estructural (shape, info)
4. Calidad de datos (nulos y duplicados)
5. Estadísticas descriptivas y distribución de clases
6. Visualizaciones univariadas (proporciones e histogramas)
7. Visualizaciones bivariadas (boxplot, scatter) con interpretación
8. Correlaciones básicas y selección preliminar de variables
9. Conclusiones
10. Referencias

Cada bloque de visualizaciones incluye ahora una celda adicional de interpretación didáctica.

## Buenas Prácticas
- Ejecutar todas las celdas antes de subir.
- Eliminar salidas irrelevantes.
- Etiquetar ejes y títulos en los gráficos.
- Evitar duplicación de código: crear funciones auxiliares si repites patrones.

## Archivos Clave
- Notebook principal: `eda_semana03.ipynb`
- (Histórico) Recursos de informe: `informes/semanas/semana-04/informe.tex` (solo si se requiere comparar evolución)

## Próximos Pasos Sugeridos (para un análisis futuro) 
- Normalizar / escalar variables antes de modelado.
- Entrenar un modelo base (Regresión Logística) con variables seleccionadas (ej. radius_mean, concavity_mean, concave_points_mean, texture_mean).
- Evaluar métricas: precision, recall, F1, matriz de confusión.
- Considerar reducción de dimensionalidad (PCA) para visualizar separación de clases.

---
*INFB6052 – Ingeniería Civil en Ciencia de Datos – Semana 03*
