# Démarre Redis (si nécessaire) puis l'API du cache sémantique.
# Usage : .\start.ps1   (depuis le dossier du projet)

# 1. Redis : démarrer le binaire portable s'il ne tourne pas déjà
try {
    $c = New-Object Net.Sockets.TcpClient
    $c.Connect('127.0.0.1', 6379)
    $c.Close()
    Write-Host "Redis : deja actif" -ForegroundColor Green
} catch {
    Start-Process -FilePath "$PSScriptRoot\redis\redis-server.exe" -WindowStyle Hidden
    Start-Sleep -Seconds 2
    Write-Host "Redis : demarre" -ForegroundColor Green
}

# 2. API FastAPI (port 8001 car 8000 est occupe par une autre application)
try {
    $c = New-Object Net.Sockets.TcpClient
    $c.Connect('127.0.0.1', 8001)
    $c.Close()
    Write-Host "API : deja active sur http://127.0.0.1:8001 - rien a faire." -ForegroundColor Green
    Write-Host "(Pour la redemarrer : arretez d'abord l'ancien processus, ou fermez ce qui ecoute sur le port 8001.)"
} catch {
    Write-Host "API : http://127.0.0.1:8001  (docs interactives : http://127.0.0.1:8001/docs)" -ForegroundColor Cyan
    & "$PSScriptRoot\venv\Scripts\uvicorn.exe" main:app --host 127.0.0.1 --port 8001
}
