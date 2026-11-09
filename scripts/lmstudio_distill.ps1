# Distill SFT from DavidAU loaded in local LM Studio (Windows).
# Prerequisites: LM Studio local server on, Python 3.10+, `pip install -e .` from repo root.
# Usage (PowerShell):
#   .\scripts\lmstudio_distill.ps1
#   $env:LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"; .\scripts\lmstudio_distill.ps1 -Limit 20

[CmdletBinding()]
param(
    [string]$BaseUrl = $(if ($env:LMSTUDIO_BASE_URL) { $env:LMSTUDIO_BASE_URL } else { "http://127.0.0.1:1234/v1" }),
    [string]$Model = $(if ($env:LMSTUDIO_MODEL) { $env:LMSTUDIO_MODEL } else { "" }),
    [string]$OutDir = $(if ($env:OUT_DIR) { $env:OUT_DIR } else { "artifacts/lmstudio" }),
    [string]$Prompts = $(if ($env:PROMPTS) { $env:PROMPTS } else { "data/prompts/reason_mode_distill.jsonl" }),
    [int]$Limit = $(if ($env:LIMIT) { [int]$env:LIMIT } else { 40 })
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:LMSTUDIO_BASE_URL = $BaseUrl
$env:LMSTUDIO_MODEL = $Model

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path "data/distill" | Out-Null

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $py) { throw "Python not found on PATH. Install Python 3.10+ and retry." }
$Python = $py.Source

Write-Host "==> Probe modes via LM Studio at $BaseUrl"
& $Python "scripts/lmstudio_probe_modes.py" --base-url $BaseUrl --model $Model --out-dir $OutDir
if ($LASTEXITCODE -ne 0) { throw "lmstudio_probe_modes.py failed (exit $LASTEXITCODE)" }

if (-not $Model) {
    $Model = & $Python -c @"
import json, os, urllib.request
url = os.environ['LMSTUDIO_BASE_URL'].rstrip('/') + '/models'
with urllib.request.urlopen(url, timeout=10) as r:
    data = json.load(r)
print((data.get('data') or [{}])[0].get('id') or 'local-model')
"@
    if ($LASTEXITCODE -ne 0 -or -not $Model) {
        throw "Failed to resolve LM Studio model id. Set -Model or `$env:LMSTUDIO_MODEL."
    }
    $Model = $Model.Trim()
    $env:LMSTUDIO_MODEL = $Model
}

Write-Host "==> Distill with teacher model: $Model"
& $Python -m swift_cf_approx.cli distill `
    --prompts $Prompts `
    --out "data/distill/lmstudio_davidau.jsonl" `
    --base-url $BaseUrl `
    --model $Model `
    --limit $Limit `
    --require-teacher
if ($LASTEXITCODE -ne 0) { throw "swift_cf_approx.cli distill failed (exit $LASTEXITCODE)" }

Write-Host "==> Merge into prepared_train.jsonl"
& $Python "scripts/merge_lmstudio_distill.py"
if ($LASTEXITCODE -ne 0) { throw "merge_lmstudio_distill.py failed (exit $LASTEXITCODE)" }

Write-Host "LM Studio distill ready -> data/distill/prepared_train.jsonl"
