[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Clear-Host

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   REAL-TIME PERFORMANCE BENCHMARKING ENGINE   " -ForegroundColor White
Write-Host "=========================================================" -ForegroundColor Cyan


$dataset = "C:\Datasets\TrafficTestSet"

if (!(Test-Path $dataset)) {
    Write-Host "HATA: $dataset klasörü bulunamadı!" -ForegroundColor Red
    exit
}

Write-Host "[*] Donanım: NVIDIA CUDA Core Aktivasyonu..." -ForegroundColor Green
Write-Host "[*] Veri Kümesi: $dataset" -ForegroundColor Yellow
Write-Host "[*] İşlem: YOLOv11 Feature Extraction + XGBoost Refinement" -ForegroundColor Gray
Write-Host "---------------------------------------------------------"


python testbench.py

Write-Host "---------------------------------------------------------"
Write-Host "Analiz başarıyla tamamlandı." -ForegroundColor Cyan
Write-Host "Raporu kapatmak için bir tuşa basın..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")