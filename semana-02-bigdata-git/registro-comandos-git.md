# Registro de Comandos Git – Semana 02

| Paso | Objetivo | Comando Ejecutado | Explicación | Evidencia (captura) |
|------|----------|-------------------|------------|---------------------|
| 1 | Clonar repositorio | `git clone URL` | Descarga copia local | 01-clone.png |
| 2 | Ver estado | `git status` | Cambios pendientes | 02-status.png |
| 3 | Añadir archivos | `git add .` | Lleva cambios al área de staging | 03-add.png |
| 4 | Confirmar | `git commit -m "feat: mensaje"` | Crea snapshot versionado | 04-commit.png |
| 5 | Sincronizar remoto | `git push origin main` | Envía commits al remoto | 05-push.png |
| 6 | Ver historial | `git log --oneline` | Lista resumida de commits | 06-log.png |
| 7 | Crear rama | `git checkout -b feature/ejemplo` | Rama para cambio aislado | 07-branch.png |
| 8 | Fusionar | `git merge feature/ejemplo` | Combina rama a main | 08-merge.png |

## Lista de Commits Documentados
| Hash corto | Mensaje | Propósito | Fecha |
|------------|---------|----------|-------|
|            |         |          |       |

## Ejemplo de Buen Mensaje
```
feat: agregar análisis exploratorio inicial

Se añade script que carga dataset y calcula estadísticas básicas. Prepara base para visualizaciones (issue #3).
```

## Uso de Copilot
- Prompt utilizado / comentario inicial.
- Sugerencia aceptada (describir brevemente).
- Ajustes realizados tras sugerencia.

## Observaciones / Lecciones
- Beneficios de commits atómicos.
- Errores comunes evitados.
