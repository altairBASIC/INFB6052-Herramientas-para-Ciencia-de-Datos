# Semana 01 – Configuración de Entorno

## Objetivo

Dejar operativo un entorno de desarrollo reproducible con Python, VS Code, extensiones y repositorio GitHub listo.

## Checklist de Evidencias

- [ ] Cuenta GitHub creada (captura)
- [ ] VS Code instalado (captura)
- [ ] Extensiones: Python, GitLens, Copilot (captura)
- [ ] Entorno virtual `.venv` creado y activado (captura)
- [ ] Configuración global de Git (`user.name`, `user.email`) (captura)
- [ ] Dependencias instaladas: `pip install -r perceptron_lab/requirements.txt`
- [ ] Informe LaTeX compilado (`informes/semanas/semana-01/main.tex`)

## Pasos Resumidos

1. Crear cuenta en GitHub y verificar correo.
2. Instalar VS Code desde la web oficial.
3. Instalar extensiones (panel Extensions): Python, GitLens, GitHub Copilot.
4. Crear entorno virtual:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r perceptron_lab/requirements.txt
   ```
5. Configurar Git:
   ```powershell
   git config --global user.name "Nombre Apellido"
   git config --global user.email "correo@utem.cl"
   ```
6. Realizar primer commit inicial.
7. Compilar informe LaTeX si procede:
   ```powershell
   pwsh infb6052-herramientas-para-ciencia-de-datos/informes/build_reports.ps1 -Semana 01
   ```

## Informe

Completar `plantilla-informe-semana01.md` (o LaTeX) y exportar a PDF.
