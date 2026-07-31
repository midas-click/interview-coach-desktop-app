# Build standalone Windows .exe for Interview Transcriber
# Usage: .\build.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== Interview Transcriber - Standalone Build ===" -ForegroundColor Cyan

# --- check Python -----------------------------------------------------------
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "ERROR: .venv not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

# --- install deps -----------------------------------------------------------
Write-Host ""
Write-Host "[1/3] Installing dependencies..." -ForegroundColor Yellow
& $python -m pip install -q -r requirements.txt pyinstaller
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed" -ForegroundColor Red
    exit 1
}

# --- build ------------------------------------------------------------------
$name = "InterviewTranscriber"
$main = "src/main.py"

Write-Host ""
Write-Host "[2/3] Building $name.exe..." -ForegroundColor Yellow
& $python -m PyInstaller `
    --onedir `
    --windowed `
    --name $name `
    --add-data "models;models" `
    --add-data ".venv\Lib\site-packages\faster_whisper\assets\silero_vad_v6.onnx;faster_whisper\assets" `
    --hidden-import faster_whisper `
    --hidden-import sounddevice `
    --hidden-import soundcard `
    --hidden-import pyaudio `
    --hidden-import qasync `
    --hidden-import aiosqlite `
    --hidden-import boto3 `
    --hidden-import pydantic_settings `
    --hidden-import yaml `
    --hidden-import numpy `
    --clean `
    --noconfirm `
    $main

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: build failed" -ForegroundColor Red
    exit 1
}

# --- done -------------------------------------------------------------------
$out = Join-Path $PSScriptRoot "dist\$name"
$exe = Join-Path $out "$name.exe"

Write-Host ""
Write-Host "[3/3] Done." -ForegroundColor Green
Write-Host ""
Write-Host "  Output:  $out" -ForegroundColor White
Write-Host "  Launch:  $exe" -ForegroundColor White
Write-Host ""
Write-Host "Copy the entire ${name} folder to another machine to run it." -ForegroundColor Cyan
