$ErrorActionPreference = 'Stop'

Write-Host "[1/4] Training parity-aware BPE..." -ForegroundColor Cyan
python parity_aware_bpe/parity_aware_learn_bpe.py `
  --symbols 100 `
  --variant base `
  --input data_tiny/train.en data_tiny/train.de `
  --dev data_tiny/dev.en data_tiny/dev.de `
  --output merges.parity.txt

Write-Host "[2/4] Creating baseline concat file..." -ForegroundColor Cyan
Get-Content data_tiny/train.en, data_tiny/train.de | Set-Content data_tiny/train.concat

Write-Host "[3/4] Training classical BPE baseline..." -ForegroundColor Cyan
python parity_aware_bpe/learn_bpe.py `
  --symbols 100 `
  --input data_tiny/train.concat `
  --output merges.classic.txt

Write-Host "[4/4] Checking outputs..." -ForegroundColor Cyan
$parityLines = (Get-Content merges.parity.txt).Count
$classLines = (Get-Content merges.classic.txt).Count
Write-Host "merges.parity.txt lines: $parityLines"
Write-Host "merges.classic.txt lines: $classLines"

if ($parityLines -lt 1 -or $classLines -lt 1) {
  throw "One of the output merge files is empty."
}

Write-Host "Done. Tiny smoke test passed." -ForegroundColor Green
