# Pipeline de Ingestión de Datos Grandes (>= 200 MB) desde la Nube

Este directorio contiene el notebook `Prueba1.ipynb`, que demuestra **estrategias prácticas para trabajar con datasets grandes sin descargarlos completamente al disco local**. El foco es educativo y se orienta a comparar enfoques y herramientas (DuckDB, Pandas, Dask, streaming manual) en escenarios de más de 200 MB de datos remotos.

## ✅ Objetivos del Notebook
- Leer múltiples archivos **Parquet remotos** (NYC Yellow Taxi) superando 200 MB combinados sin almacenarlos localmente.
- Aplicar **consultas SQL con DuckDB** sobre archivos en la nube (HTTP range requests, predicate pushdown, column projection).
- Mostrar un **patrón de streaming por chunks** para CSV remoto grande usando `requests` + `pandas`.
- Comparar **Pandas vs Dask**: memoria, paralelismo, modelo de ejecución, rendimiento y cuándo elegir cada uno.
- Ejecutar un **benchmark reproducible** (agregaciones) para ilustrar overhead y escalabilidad.

---
## 🗂 Dataset Principal
**NYC Yellow Taxi Trip Records** (CloudFront CDN):
- Base URL: `https://d37ci6vzurychx.cloudfront.net/trip-data/`
- Formato: `yellow_tripdata_YYYY-MM.parquet`
- Tamaño por archivo: ~180–250 MB
- Ejemplo usado: meses `2024-01`, `2024-02`, `2024-03` (conjunto > 500 MB lógicos).

> No se descargan completos; DuckDB y Dask leen sólo lo necesario (lectura columnar + HTTP range requests).

### Columnas Relevantes Usadas
- `tpep_pickup_datetime`
- `trip_distance`
- `total_amount`
- `passenger_count`

---
## 🧰 Herramientas y Roles
| Herramienta | Rol Principal | Ventajas Clave |
|-------------|--------------|----------------|
| **DuckDB** | SQL local sobre datos remotos (Parquet) | Pushdown de filtros y columnas; sintaxis SQL familiar |
| **Pandas** | Exploración y manipulación en memoria | API madura, ideal si cabe en RAM |
| **Dask DataFrame** | Escalar DataFrames mayores que RAM | Ejecución lazy, paralelismo multi-core, múltiples archivos |
| **requests** | Streaming manual HTTP | Control detallado chunk a chunk |
| **PyArrow** | Backend Parquet/columnar | Eficiencia en lectura de columnas |
| **fsspec** | Acceso unificado a sistemas de archivos | Soporte HTTP/S3 sin copiar local |

---
## ⚙️ Dependencias
Se instalan dinámicamente en el notebook si faltan:
```
duckdb
pyarrow
fsspec
datasets
pandas
dask[dataframe]
requests
```

### Instalación manual (opcional) en PowerShell
```powershell
# (Recomendado) Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
python -m pip install duckdb pyarrow fsspec datasets pandas "dask[dataframe]" requests
```

> Si aparece el error de políticas de ejecución al activar, ejecutar:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---
## ▶️ Flujo de Trabajo en el Notebook
1. **Selección de URLs Parquet**: Construcción de la lista de meses a consultar.
2. **Consulta inicial con DuckDB**: `read_parquet([...])` + `LIMIT` para muestra.
3. **Agregaciones**: Group-by diario aplicando filtros (`WHERE trip_distance BETWEEN 0.1 AND 100`).
4. **Inspección de esquema** sin leer datos: `DESCRIBE SELECT * FROM read_parquet('...') LIMIT 0`.
5. **Streaming CSV**: Lectura progresiva con `requests.iter_lines()` y acumulación chunk a chunk.
6. **Dask Parquet**: Carga lazy de patrón (`yellow_tripdata_2024-0*.parquet`) y agregación.
7. **Dask CSV**: Lectura con `blocksize` configurado.
8. **Resumen comparativo (Markdown)**: Diferencias Pandas vs Dask.
9. **Benchmark**: Tiempos de agregación Pandas vs Dask (subset de filas) para ilustrar overhead.

---
## 🧪 Benchmark Incluido
El benchmark realiza:
- Pandas: extracción de 300k filas con DuckDB + `groupby`.
- Dask: misma agregación sobre DataFrame lazy (limitando con `head()` para volumen comparable).
- Repite 3 veces y calcula medias y desviaciones.

### Interpretación Típica
| Caso | Resultado Común |
|------|-----------------|
| Subconjunto pequeño | Pandas suele ser más rápido (menor overhead) |
| Volumen grande / muchos archivos | Dask escala mejor al paralelizar |
| Pre-filtrado SQL antes de Pandas/Dask | Acelera ambas rutas; reduce memoria |

> Ajustar el `LIMIT` o eliminarlo para observar cuándo Dask empieza a superar a Pandas.

---
## 🧠 Pandas vs Dask (Resumen Conceptual)
| Dimensión | Pandas | Dask |
|----------|--------|------|
| Ejecución | Inmediata | Lazy (DAG) |
| Escalado | RAM local | Multi-core / cluster |
| Lectura múltiple Parquet | Manual (lista + concat) | Patrón con wildcard | 
| Control Memoria | Limitado (chunks CSV) | Particiones automáticas |
| Overhead | Bajo | Scheduler (~ms) |
| Ideal Para | Exploración rápida | ETL repetible, big data |

**Complemento**: Usar DuckDB antes de Pandas/Dask para minimizar el volumen.

---
## 🛠 Patrones Clave del Código
### 1. Consulta remota Parquet (DuckDB)
```python
query = f"""
SELECT date_trunc('day', tpep_pickup_datetime) AS pickup_date,
       passenger_count, trip_distance, total_amount
FROM read_parquet([{','.join([repr(u) for u in urls])}])
WHERE trip_distance > 0
LIMIT 1000
"""
df_sample = duckdb.query(query).to_df()
```
### 2. Agregación remota
```python
agg_query = f"""
SELECT date_trunc('day', tpep_pickup_datetime) AS day,
       COUNT(*) AS trips,
       AVG(trip_distance) AS avg_distance,
       AVG(total_amount) AS avg_amount
FROM read_parquet([{','.join([repr(u) for u in urls])}])
WHERE trip_distance BETWEEN 0.1 AND 100
GROUP BY 1 ORDER BY 1 LIMIT 15
"""
df_agg = duckdb.query(agg_query).to_df()
```
### 3. Streaming CSV manual
```python
with requests.get(csv_url, stream=True) as r:
    for i, line in enumerate(r.iter_lines(decode_unicode=True)):
        # acumulación y parseo por bloques
```
### 4. Dask Parquet
```python
pattern = base_url + '/yellow_tripdata_2024-0*.parquet'
ddf = dd.read_parquet(pattern, engine='pyarrow', gather_statistics=False)
```
### 5. Benchmark (extracto)
```python
pdf = duckdb.query(SQL_SUBSET).to_df()
agg_pdf = pdf.groupby('pickup_date').agg({...})
# vs
result = ddf.assign(pickup_date=...).groupby('pickup_date').agg({...}).head(20).compute()
```

---
## 🚀 Cómo Ejecutar el Notebook
1. Abrir VS Code y este directorio.
2. (Opcional) Activar entorno virtual.
3. Abrir `Prueba1.ipynb` y ejecutar celdas secuencialmente (algunas celdas instalan dependencias si faltan).
4. Revisar la celda de **benchmark** al final para observar tiempos.

### Consejos de Ejecución
- Si la red es lenta, aumentar el `LIMIT` gradualmente.
- Para un análisis más profundo, retirar `LIMIT` de consultas y observar consumo.
- Si aparece error de columna ausente, verificar nombre (`tpep_pickup_datetime`).

---
## 🔍 Posibles Extensiones
| Mejora | Descripción |
|--------|-------------|
| Polars | Añadir comparación adicional (lazy + SIMD). |
| Persistencia Dask | Añadir `.persist()` tras filtrado para reutilizar resultados. |
| Métricas Memoria | Integrar medición con `psutil` o `memory_profiler`. |
| Visual DAG | Usar `ddf.visualize()` para enseñar el grafo de tareas. |
| Más Meses | Incluir más meses y observar escalado. |
| Joins complejos | Unir con tablas de tarifas o zonas para enriquecer análisis. |

---
## ❓ FAQ Rápido
**¿Por qué DuckDB si ya uso Dask?**
Para filtrar/seleccionar columnas tempranamente con SQL eficiente y reducir bytes leídos.

**¿Necesito descargar los Parquet primero?**
No. DuckDB y PyArrow pueden hacer HTTP range requests.

**¿Dask siempre será más rápido?**
No: en volúmenes pequeños el overhead lo hace más lento que Pandas.

---
## ⚖️ Decisión de Herramienta (Guía Express)
| Situación | Elige |
|-----------|-------|
| Dataset cabe en RAM y exploración ad-hoc | Pandas |
| Muchos archivos mensuales (GB totales) y pipeline repetible | Dask |
| Necesitas SQL y reducción previa sin cluster | DuckDB |
| Flujos lineales streaming (CSV/JSON) | requests + chunks |

---
## 📝 Licenciamiento / Uso de Datos
Los datos de NYC Taxi tienen términos públicos para uso educativo (ver documentación oficial). Verifica siempre restricciones antes de redistribuir.

---
## 📌 Resumen Final
Este notebook demuestra un **enfoque híbrido**: combinar SQL columnar (DuckDB) + DataFrames (Pandas/Dask) + streaming manual para abordar datasets que exceden memoria sin infraestructura compleja. La elección de herramienta depende del tamaño, patrón de acceso y requisitos de reproducibilidad.

> Sugerencia: Usa este README como guía de laboratorio para experimentos incrementales. Empieza con 1 mes, escala a 6–12 meses y compara tiempos/memoria registrando tus observaciones.
