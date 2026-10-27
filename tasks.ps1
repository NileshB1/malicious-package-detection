# Runs the project's steps in order. From the project root, with the virtual environment active:
#
#   .\tasks.ps1 setup       create .venv and install pinned dependencies
#   .\tasks.ps1 all         data -> features -> model -> evaluation -> business results
#   .\tasks.ps1 app         start the Streamlit app
#   .\tasks.ps1 test        run the test suite
#   .\tasks.ps1 <step>      run one step: data, audit, features, train, evaluate, simulate,
#                           errors, reproduce
#
# If PowerShell refuses to run the script, allow local scripts for this session first:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

param([string]$Task = "help")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

$steps = [ordered]@{
    data      = "scripts\download_data.py"
    audit     = "scripts\run_data_audit.py"
    features  = "scripts\build_dataset.py"
    train     = "scripts\train_model.py"
    evaluate  = "scripts\evaluate_model.py"
    simulate  = "scripts\run_business_simulation.py"
    errors    = "scripts\error_analysis.py"
    reproduce = "scripts\check_reproducibility.py"
}

function Invoke-Native([string[]]$arguments) {
    # Windows PowerShell treats anything a native program writes to stderr (Python's log lines,
    # library warnings) as an error when ErrorActionPreference is Stop. Judge by exit code only.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    # Out-Host so the program's output goes to the console and only the exit code is returned
    try { & $python @arguments 2>&1 | ForEach-Object { "$_" } | Out-Host }
    finally { $ErrorActionPreference = $previous }
    return $LASTEXITCODE
}

function Invoke-Step([string]$name) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $code = Invoke-Native @($steps[$name])
    if ($code -ne 0) { throw "step '$name' failed (exit code $code)" }
}

switch ($Task) {
    "setup" {
        if (-not (Test-Path ".venv")) { python -m venv .venv }
        & .venv\Scripts\python.exe -m pip install --upgrade pip
        & .venv\Scripts\python.exe -m pip install -r requirements.txt
        & .venv\Scripts\python.exe -m pip install -e . --no-deps
        & .venv\Scripts\python.exe -m ipykernel install --user --name pkgtriage --display-name "Python (pkgtriage)"
    }
    "all" {
        # about 6-7 minutes in total; training is the slow part
        foreach ($name in @("data", "audit", "features", "train", "evaluate", "simulate", "errors")) {
            Invoke-Step $name
        }
    }
    "app" { $null = Invoke-Native @("-m", "streamlit", "run", "app\streamlit_app.py") }
    "test" {
        $code = Invoke-Native @("-m", "pytest")
        if ($code -ne 0) { throw "tests failed (exit code $code)" }
    }
    "help" { Get-Content $PSCommandPath -TotalCount 12 }
    default {
        if ($steps.Contains($Task)) { Invoke-Step $Task }
        else { throw "unknown task '$Task' - run .\tasks.ps1 help" }
    }
}
