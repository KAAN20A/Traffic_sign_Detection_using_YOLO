
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8


if ($args.Count -eq 0) {
    Write-Host ""
    Write-Host "[SİSTEM] Kullanım: .\run.ps1 [görsel_yolu]" -ForegroundColor Cyan
    Write-Host "Örnek: .\run.ps1 .\test_resimleri\dur_tabelasi.jpg"
    Write-Host ""
    exit
}

$IMAGE_PATH = $args[0]


Write-Host ""
Write-Host "-----------------------------------------------" -ForegroundColor Gray
Write-Host "[SİSTEM] Hibrit Analiz Modülü Başlatılıyor..." -ForegroundColor Cyan
Write-Host "[SİSTEM] Görsel İşleniyor: " -NoNewline -ForegroundColor Cyan
Write-Host "$IMAGE_PATH" -ForegroundColor Green
Write-Host "-----------------------------------------------" -ForegroundColor Gray


python main.py --mode predict_hybrid --input "$IMAGE_PATH"

Write-Host "-----------------------------------------------" -ForegroundColor Gray
Write-Host "[BİTTİ] Analiz tamamlandı." -ForegroundColor Gray
Write-Host "