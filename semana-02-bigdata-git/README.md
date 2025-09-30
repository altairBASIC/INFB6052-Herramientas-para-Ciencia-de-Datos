# Semana 02 – Fundamentos Big Data y Flujo Git

## Objetivos
- Comprender conceptos básicos de Big Data (5V, valor, aplicaciones).
- Configurar repositorio en GitHub y practicar flujo básico (clone, add, commit, push).
- Probar sugerencias de GitHub Copilot.

## Evidencias Esperadas
- Registro de comandos Git utilizados con explicación.
- Capturas de commits iniciales (historial / panel Source Control).
- Demostración de uso de Copilot (captura sugerencia aceptada).
- Archivo `registro-comandos-git.md` completado.

## Flujo Recomendado
1. Crear repositorio remoto en GitHub (público).
2. Clonar localmente:
   ```powershell
   git clone https://github.com/usuario/repositorio.git
   ```
3. Crear/editar archivos.
4. Commit atómico por cambio lógico.
5. Push frecuente.
6. Ver historial: `git log --oneline --graph --decorate`.

## Buenas Prácticas de Commits
Formato: `<tipo>: descripción breve`  
Tipos: `feat`, `fix`, `docs`, `refactor`, `style`, `test`.

## Archivo Clave
Completar `registro-comandos-git.md` y referenciar en informe LaTeX (`informes/semanas/semana-02/main.tex`).

## Compilación Informe (Opcional)
```powershell
pwsh infb6052-herramientas-para-ciencia-de-datos/informes/build_reports.ps1 -Semana 02
```
