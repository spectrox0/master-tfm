# Check if Python is installed
$python = Get-Command python -ErrorAction SilentlyContinue
$python3 = Get-Command python3 -ErrorAction SilentlyContinue

if ($python3) {
    Write-Host "Python3 is installed. Running initialize.py..."
    python3 initialize.py
} elseif ($python) {
    Write-Host "Python is installed. Running initialize.py..."
    python initialize.py
} else {
    Write-Host "Python is not installed. Please install Python to proceed."
    exit 1
}
