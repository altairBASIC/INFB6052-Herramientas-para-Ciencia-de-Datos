# Comparación de Librerías de Visualización en Python

## Descripción General

Este proyecto presenta un análisis comparativo exhaustivo de las tres librerías fundamentales para visualización de datos en el ecosistema Python: **Matplotlib**, **Seaborn** y **Plotly**. El análisis incluye ejemplos prácticos, evaluación de capacidades, y recomendaciones de uso según diferentes escenarios profesionales.

## Contenido del Proyecto

### Archivo Principal

- **`comparación-matplotlib-seaborn-plotly.ipynb`**: Jupyter Notebook con análisis completo incluyendo:
  - Configuración e instalación de librerías
  - Generación de dataset sintético de ventas
  - Ejemplos prácticos con cada librería
  - Análisis comparativo detallado
  - Recomendaciones por escenario de uso

## Librerías Evaluadas

### 1. Matplotlib
**Versión utilizada:** 3.9.0

**Características principales:**
- Librería de bajo nivel con control granular
- Base del ecosistema de visualización en Python
- Ideal para publicaciones científicas y académicas
- Soporte para múltiples formatos de exportación (PDF, PNG, SVG, EPS)

**Ventajas:**
- Máxima personalización de elementos visuales
- Ampliamente adoptada en la comunidad científica
- Excelente rendimiento con datasets grandes
- Documentación exhaustiva

**Desventajas:**
- Sintaxis verbosa que requiere más líneas de código
- Curva de aprendizaje empinada
- Estética predeterminada básica
- Interactividad limitada

### 2. Seaborn
**Versión utilizada:** 0.13.2

**Características principales:**
- Librería de alto nivel construida sobre Matplotlib
- Especializada en visualizaciones estadísticas
- Integración nativa con Pandas DataFrames
- Paletas de colores modernas por defecto

**Ventajas:**
- Sintaxis concisa y declarativa
- Visualizaciones estadísticas predefinidas (boxplot, violinplot, heatmap)
- Estética profesional sin configuración adicional
- Ideal para análisis exploratorio de datos (EDA)

**Desventajas:**
- Menor control que Matplotlib para personalizaciones avanzadas
- Limitado a visualizaciones estáticas
- Puede ser más lento con datasets muy grandes

### 3. Plotly
**Versión utilizada:** 6.3.1

**Características principales:**
- Librería de visualización interactiva basada en D3.js
- Soporte nativo para dashboards con framework Dash
- Gráficos 3D robustos
- Exportación a HTML standalone

**Ventajas:**
- Interactividad completa (zoom, pan, hover, filtros)
- Aspecto profesional y moderno
- Excelente para dashboards web
- Actualización en tiempo real de datos

**Desventajas:**
- Archivos de salida más pesados (HTML + JavaScript)
- Dependencia de navegador para visualización completa
- Rendimiento puede degradarse con datasets muy grandes sin optimización

## Ejemplos Implementados

### Matplotlib
1. **Gráfico de dispersión**: Relación Precio vs Ventas por categoría con personalización completa
2. **Gráfico dual**: Combinación de barras y líneas con ejes secundarios

### Seaborn
1. **Gráfico de dispersión mejorado**: Mismo dataset con sintaxis simplificada
2. **Panel estadístico**: Combinación de boxplot, violinplot, heatmap y barplot con cálculos automáticos

### Plotly
1. **Gráfico de dispersión interactivo**: Con tooltips, zoom y filtros por categoría
2. **Dashboard complejo**: Múltiples subplots sincronizados (boxplot, línea temporal, barras, polar)
3. **Visualización 3D**: Análisis tridimensional con rotación 360° interactiva

## Dataset Utilizado

**Descripción:** Dataset sintético de ventas de productos

**Características:**
- 200 muestras
- 5 categorías de productos (Electrónica, Ropa, Alimentos, Hogar, Deportes)
- Variables: Producto, Categoría, Ventas, Precio, Satisfacción, Mes, Región, Ingresos
- Generado con `numpy.random` (seed=42 para reproducibilidad)

## Análisis Comparativo

### Tabla de Evaluación

El notebook incluye una tabla comparativa con 18 criterios de evaluación:

| Aspecto | Matplotlib | Seaborn | Plotly |
|---------|-----------|---------|--------|
| Nivel de abstracción | Bajo | Alto | Alto |
| Interactividad | Limitada | No | Nativa |
| Personalización | Máxima | Moderada | Alta |
| Integración Pandas | Buena | Excelente | Excelente |
| Dashboards | No nativo | No nativo | Excelente |
| Publicaciones académicas | Excelente | Muy bueno | Limitado |
| Gráficos 3D | Básico | No | Excelente |

## Recomendaciones por Escenario

### Escenario 1: Publicación Académica
**Librería recomendada:** Matplotlib

**Justificación:** Control total sobre elementos visuales, formatos vectoriales de alta calidad (PDF, EPS), estándar aceptado en journals científicos.

### Escenario 2: Análisis Exploratorio de Datos (EDA)
**Librería recomendada:** Seaborn (principal) + Plotly (exploración interactiva)

**Justificación:** Seaborn ofrece visualizaciones estadísticas rápidas con sintaxis mínima. Plotly complementa para exploración interactiva de outliers.

### Escenario 3: Dashboard Ejecutivo
**Librería recomendada:** Plotly + Dash

**Justificación:** Interactividad nativa, actualización en tiempo real, aspecto profesional moderno, exportable a HTML.

### Escenario 4: Reporte Estático (PDF/PowerPoint)
**Librería recomendada:** Seaborn

**Justificación:** Estética moderna lista para presentaciones, menos código que Matplotlib, exportación de alta calidad.

### Escenario 5: Visualización 3D o Geoespacial
**Librería recomendada:** Plotly

**Justificación:** Soporte robusto para gráficos 3D interactivos, mapas geográficos nativos, rotación y zoom.

### Escenario 6: Personalización Extrema
**Librería recomendada:** Matplotlib

**Justificación:** Control granular sobre cada elemento, creación de gráficos no convencionales, transformaciones avanzadas.

## Requisitos Técnicos

### Dependencias

```bash
pip install matplotlib seaborn plotly pandas numpy
```

### Versiones Utilizadas

- Python: 3.10.11
- Matplotlib: 3.9.0
- Seaborn: 0.13.2
- Plotly: 6.3.1
- Pandas: 2.2.2
- NumPy: 1.26.4
- nbformat: 5.10.4 (para renderizado de Plotly en VS Code)

### Entorno de Desarrollo

- Editor: Visual Studio Code
- Extensión: Jupyter Notebook
- Kernel: Python 3.10 (.venv)
- Sistema operativo: Windows

## Configuración Especial

### Plotly en VS Code

Para visualizar correctamente los gráficos de Plotly en Jupyter Notebooks dentro de VS Code, se requiere la siguiente configuración:

```python
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook_connected"
```

Esta configuración está incluida en el notebook (celda 4).

## Ejecución del Proyecto

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/altairBASIC/INFB6052-Herramientas-Para-Ciencia-de-Datos.git
cd INFB6052-Herramientas-Para-Ciencia-de-Datos/prueba-1/Primera-Parte/Comparar-lib-visualizaciones
```

### Paso 2: Crear entorno virtual

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

Nota: Si no existe `requirements.txt`, instalar manualmente:

```bash
pip install matplotlib seaborn plotly pandas numpy jupyter nbformat
```

### Paso 4: Abrir el notebook

```bash
jupyter notebook comparación-matplotlib-seaborn-plotly.ipynb
```

O abrir directamente en VS Code con la extensión de Jupyter.

### Paso 5: Ejecutar celdas

Ejecutar las celdas secuencialmente. Se recomienda ejecutar primero las celdas de configuración (1-4) antes de proceder con los ejemplos.

## Estructura del Notebook

1. **Introducción y objetivos** (Celdas 1-2)
2. **Instalación y configuración** (Celdas 3-4)
3. **Preparación de datos** (Celdas 5-6)
4. **Análisis Matplotlib** (Celdas 7-11)
5. **Análisis Seaborn** (Celdas 12-16)
6. **Análisis Plotly** (Celdas 17-23)
7. **Comparación y conclusiones** (Celdas 24-28)

## Resultados Clave

### Hallazgos Principales

1. **No existe una librería superior universal**: La elección depende del contexto, audiencia y requisitos específicos del proyecto.

2. **Matplotlib es fundamental**: Entender Matplotlib es esencial ya que Seaborn se construye sobre ella y muchos conceptos aplican a otras librerías.

3. **Seaborn optimiza el flujo estadístico**: Reduce significativamente el código necesario para análisis exploratorio y visualizaciones estadísticas comunes.

4. **Plotly domina la interactividad**: Cuando se requiere exploración dinámica, dashboards o presentaciones interactivas, Plotly es la opción clara.

5. **La combinación es válida**: Es común usar múltiples librerías en un mismo proyecto según las necesidades de cada etapa.

### Estrategia Híbrida Recomendada

1. **Exploración inicial**: Seaborn para EDA rápido
2. **Análisis interactivo**: Plotly para exploración profunda con stakeholders
3. **Publicación final**: Matplotlib para control fino o Plotly para compartir en web

## Mejores Prácticas

1. **Consistencia visual**: Mantener un estilo coherente en todo el proyecto
2. **Simplicidad**: Evitar sobrecarga visual (menos es más)
3. **Accesibilidad**: Usar paletas amigables para daltonismo
4. **Etiquetas descriptivas**: Siempre incluir títulos, ejes y leyendas claras
5. **Contexto adecuado**: Adaptar el tipo de visualización al mensaje
6. **Código reproducible**: Documentar versiones y configuraciones

## Recursos Adicionales

### Documentación Oficial

- **Matplotlib**: https://matplotlib.org/stable/tutorials/index.html
- **Seaborn**: https://seaborn.pydata.org/tutorial.html
- **Plotly Python**: https://plotly.com/python/

### Herramientas Complementarias

- **From Data to Viz**: https://www.data-to-viz.com/ (guía de selección de gráficos)
- **Colorbrewer**: https://colorbrewer2.org/ (paletas de colores accesibles)
- **Plotly Dash**: https://dash.plotly.com/ (framework para dashboards)

## Autor

**Curso:** INFB6052 - Herramientas para Ciencia de Datos  
**Institución:** Universidad Tecnológica Metropolitana (UTEM)  
**Fecha:** Octubre 2025  
**Contexto:** Primera Prueba - Item 4 (Comparación de librerías de visualización)

## Licencia

Este proyecto es material académico desarrollado para fines educativos en el contexto del curso INFB6052.

## Contacto

Para consultas o mejoras sobre este análisis, contactar a través del repositorio del curso:  
https://github.com/altairBASIC/INFB6052-Herramientas-Para-Ciencia-de-Datos

---

**Última actualización:** Octubre 7, 2025
