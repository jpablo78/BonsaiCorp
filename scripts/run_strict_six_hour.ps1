param(
    [string]$ProjectRoot = "C:\AlixPartners"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$data = Join-Path $ProjectRoot "output\cleaned_data"
$warmStart = Join-Path $ProjectRoot "output_strict_reoptimized\asignacion_optima.csv"
$outputRoot = Join-Path $ProjectRoot "output_strict_six_hour"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $warmStart)) {
    throw "Validated 3-mm warm start not found: $warmStart"
}

$common = @(
    "--data-dir", $data,
    "--num-search-workers", "6",
    "--pair-profile-limit", "100",
    "--max-extra-pair-designs", "4500",
    "--pallet-variant-profile-limit", "90",
    "--max-pallet-variants-per-profile", "18",
    "--warm-start-variant-profile-limit", "60",
    "--warm-start-compromise-group-limit", "60",
    "--max-compromise-variants-per-group", "24"
)

& $python -m bonsai optimize @common `
    "--output-dir" (Join-Path $outputRoot "3mm") `
    "--thickness-mm" "3.0" `
    "--warm-start" $warmStart `
    "--time-limit-seconds" "14400"

& $python -m bonsai optimize @common `
    "--output-dir" (Join-Path $outputRoot "4p5mm") `
    "--thickness-mm" "4.5" `
    "--time-limit-seconds" "3600"

& $python -m bonsai optimize @common `
    "--output-dir" (Join-Path $outputRoot "5mm") `
    "--thickness-mm" "5.0" `
    "--time-limit-seconds" "3600"
