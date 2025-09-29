Param(
    [string]$Name = "venv",
    [switch]$Force
)

Write-Host "[INFO] Creando entorno virtual '$Name'" -ForegroundColor Cyan
if (Test-Path $Name) {
    if ($Force) {
        Write-Host "[WARN] Eliminando entorno existente '$Name' (Force)" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $Name
    } else {
        Write-Host "[INFO] Ya existe '$Name' -> omitiendo creación (usa -Force para recrear)" -ForegroundColor Green
    }
}

if (-not (Test-Path $Name)) {
    python -m venv $Name
}

Write-Host "[INFO] Activando entorno..." -ForegroundColor Cyan
& .\$Name\Scripts\Activate.ps1

if (-not (Test-Path 'requirements.txt')) {
    Write-Host "[WARN] No se encontró requirements.txt en el directorio actual." -ForegroundColor Yellow
} else {
    Write-Host "[INFO] Instalando dependencias de requirements.txt" -ForegroundColor Cyan
    pip install --upgrade pip > $null
    pip install -r requirements.txt
}

Write-Host "[OK] Entorno listo. Usa:  .\\$Name\\Scripts\\Activate.ps1  para reactivarlo." -ForegroundColor Green
