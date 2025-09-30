# Docker Management Script for Crypto Trading Platform
# Usage: .\scripts\docker-manage.ps1 [command] [options]

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "up", "down", "restart", "logs", "status", "clean", "backup", "restore", "shell", "migrate", "help")]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Service = "",
    
    [switch]$Production,
    [switch]$Development,
    [switch]$Monitoring,
    [switch]$Force,
    [switch]$Follow
)

# Configuration
$ProjectName = "crypto-platform"
$ComposeFile = "docker-compose.yml"
$OverrideFile = "docker-compose.override.yml"

function Show-Help {
    Write-Host "Docker Management Script for Crypto Trading Platform" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\scripts\docker-manage.ps1 [command] [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host "  build      Build all services or specific service"
    Write-Host "  up         Start all services or specific service"
    Write-Host "  down       Stop all services"
    Write-Host "  restart    Restart all services or specific service"
    Write-Host "  logs       Show logs for all services or specific service"
    Write-Host "  status     Show status of all services"
    Write-Host "  clean      Clean up containers, images, and volumes"
    Write-Host "  backup     Create database backup"
    Write-Host "  restore    Restore database from backup"
    Write-Host "  shell      Open shell in service container"
    Write-Host "  migrate    Run database migrations"
    Write-Host "  help       Show this help message"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  -Production    Use production profile"
    Write-Host "  -Development   Use development profile"
    Write-Host "  -Monitoring    Include monitoring services"
    Write-Host "  -Force         Force operation (for clean, down)"
    Write-Host "  -Follow        Follow logs output"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\scripts\docker-manage.ps1 up -Production"
    Write-Host "  .\scripts\docker-manage.ps1 logs web -Follow"
    Write-Host "  .\scripts\docker-manage.ps1 shell web"
    Write-Host "  .\scripts\docker-manage.ps1 backup"
}

function Get-ComposeCommand {
    $cmd = "docker-compose -p $ProjectName"
    
    if (Test-Path $ComposeFile) {
        $cmd += " -f $ComposeFile"
    }
    
    if ($Development -and (Test-Path $OverrideFile)) {
        $cmd += " -f $OverrideFile"
    }
    
    # Add profiles
    $profiles = @()
    if ($Production) { $profiles += "production" }
    if ($Development) { $profiles += "development" }
    if ($Monitoring) { $profiles += "monitoring" }
    
    if ($profiles.Count -gt 0) {
        foreach ($profile in $profiles) {
            $cmd += " --profile $profile"
        }
    }
    
    return $cmd
}

function Invoke-DockerCommand {
    param([string]$DockerCmd)
    
    Write-Host "Executing: $DockerCmd" -ForegroundColor Gray
    Invoke-Expression $DockerCmd
}

function Build-Services {
    $composeCmd = Get-ComposeCommand
    
    if ($Service) {
        $cmd = "$composeCmd build $Service"
    } else {
        $cmd = "$composeCmd build"
    }
    
    if ($Force) {
        $cmd += " --no-cache"
    }
    
    Invoke-DockerCommand $cmd
}

function Start-Services {
    $composeCmd = Get-ComposeCommand
    
    # Check if .env file exists
    if (-not (Test-Path ".env")) {
        Write-Warning ".env file not found. Creating from template..."
        if (Test-Path ".env.docker") {
            Copy-Item ".env.docker" ".env"
            Write-Host "Please edit .env file with your configuration before starting services." -ForegroundColor Yellow
            return
        }
    }
    
    if ($Service) {
        $cmd = "$composeCmd up -d $Service"
    } else {
        $cmd = "$composeCmd up -d"
    }
    
    Invoke-DockerCommand $cmd
    
    Write-Host "Services started successfully!" -ForegroundColor Green
    Write-Host "Access the application at: http://localhost" -ForegroundColor Cyan
    
    if ($Monitoring) {
        Write-Host "Monitoring services:" -ForegroundColor Cyan
        Write-Host "  - Grafana: http://localhost:3000" 
        Write-Host "  - Prometheus: http://localhost:9090"
        Write-Host "  - Flower: http://localhost:5555"
    }
}

function Stop-Services {
    $composeCmd = Get-ComposeCommand
    
    if ($Force) {
        $cmd = "$composeCmd down -v --remove-orphans"
    } else {
        $cmd = "$composeCmd down"
    }
    
    Invoke-DockerCommand $cmd
}

function Restart-Services {
    $composeCmd = Get-ComposeCommand
    
    if ($Service) {
        $cmd = "$composeCmd restart $Service"
    } else {
        $cmd = "$composeCmd restart"
    }
    
    Invoke-DockerCommand $cmd
}

function Show-Logs {
    $composeCmd = Get-ComposeCommand
    
    $cmd = "$composeCmd logs"
    
    if ($Follow) {
        $cmd += " -f"
    }
    
    if ($Service) {
        $cmd += " $Service"
    }
    
    Invoke-DockerCommand $cmd
}

function Show-Status {
    $composeCmd = Get-ComposeCommand
    Invoke-DockerCommand "$composeCmd ps"
}

function Clean-Docker {
    if ($Force) {
        Write-Host "Cleaning Docker resources..." -ForegroundColor Yellow
        
        # Stop all containers
        Stop-Services
        
        # Remove containers, networks, images, and volumes
        docker system prune -af --volumes
        
        Write-Host "Docker cleanup completed!" -ForegroundColor Green
    } else {
        Write-Host "Use -Force flag to confirm cleanup operation" -ForegroundColor Yellow
    }
}

function Backup-Database {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupFile = "backups/crypto_platform_$timestamp.sql"
    
    Write-Host "Creating database backup..." -ForegroundColor Yellow
    
    $cmd = "docker exec crypto_postgres pg_dump -U postgres crypto_platform > $backupFile"
    Invoke-DockerCommand $cmd
    
    Write-Host "Backup created: $backupFile" -ForegroundColor Green
}

function Restore-Database {
    param([string]$BackupFile)
    
    if (-not $BackupFile) {
        Write-Host "Please specify backup file path" -ForegroundColor Red
        return
    }
    
    if (-not (Test-Path $BackupFile)) {
        Write-Host "Backup file not found: $BackupFile" -ForegroundColor Red
        return
    }
    
    Write-Host "Restoring database from: $BackupFile" -ForegroundColor Yellow
    
    $cmd = "docker exec -i crypto_postgres psql -U postgres crypto_platform < $BackupFile"
    Invoke-DockerCommand $cmd
    
    Write-Host "Database restored successfully!" -ForegroundColor Green
}

function Open-Shell {
    if (-not $Service) {
        $Service = "web"
    }
    
    Write-Host "Opening shell in $Service container..." -ForegroundColor Yellow
    
    $cmd = "docker exec -it crypto_$Service /bin/bash"
    Invoke-DockerCommand $cmd
}

function Run-Migration {
    Write-Host "Running database migrations..." -ForegroundColor Yellow
    
    $cmd = "docker exec crypto_web python -c `"
from app import create_app, db
from flask_migrate import upgrade
app = create_app()
with app.app_context():
    upgrade()
    print('Migrations completed')
`""
    
    Invoke-DockerCommand $cmd
    Write-Host "Migrations completed!" -ForegroundColor Green
}

# Main command execution
switch ($Command) {
    "build" { Build-Services }
    "up" { Start-Services }
    "down" { Stop-Services }
    "restart" { Restart-Services }
    "logs" { Show-Logs }
    "status" { Show-Status }
    "clean" { Clean-Docker }
    "backup" { Backup-Database }
    "restore" { Restore-Database $Service }
    "shell" { Open-Shell }
    "migrate" { Run-Migration }
    "help" { Show-Help }
    default { Show-Help }
}
