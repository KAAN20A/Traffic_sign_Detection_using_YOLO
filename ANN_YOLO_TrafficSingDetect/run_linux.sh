#!/bin/bash

# Renk tanımlamaları (Daha havalı görünmesi için)
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # Renksiz

# Eğer kullanıcı yol girmemişse uyarı ver
if [ -z "$1" ]; then
    echo -e "${BLUE}[SİSTEM]${NC} Kullanım: ./run.sh [görsel_yolu]"
    echo "Örnek: ./run.sh ./test_resimleri/dur_tabelasi.jpg"
    exit 1
fi

IMAGE_PATH=$1

echo -e "${BLUE}[SİSTEM]${NC} Hibrit Analiz Modülü Başlatılıyor..."
echo -e "${BLUE}[SİSTEM]${NC} Görsel İşleniyor: ${GREEN}$IMAGE_PATH${NC}"
echo "-----------------------------------------------"

# Sadece tahmin modunda çalıştır
python main.py --mode predict_hybrid --input "$IMAGE_PATH"

echo "-----------------------------------------------"