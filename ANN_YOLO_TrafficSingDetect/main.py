import os
import sys
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import cv2
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class TrafficSignSystem:
    def __init__(self, yolo_path="best.pt", xgb_path="xgboost_refiner.joblib"):
        self.yolo_path = yolo_path
        self.xgb_path = xgb_path
        self.yolo_model = None
        self.xgb_model = None

    def load_models(self, load_xgb=True):
        """Modelleri ihtiyaç halinde yükler."""
        if self.yolo_model is None and os.path.exists(self.yolo_path):
            self.yolo_model = YOLO(self.yolo_path)
        
        if load_xgb and self.xgb_model is None and os.path.exists(self.xgb_path):
            self.xgb_model = joblib.load(self.xgb_path)

    def get_features(self, image_path):
        """YOLO kullanarak XGBoost için özellik (feature) vektörü çıkarır."""
        self.load_models(load_xgb=False)
        results = self.yolo_model.predict(image_path, verbose=False)
        result = results[0]
        
        if len(result.boxes) == 0:
            return None

        # En yüksek güven skoruna sahip kutuyu al
        box = result.boxes[0]
        # YOLOv11 olasılık dağılımı (nc=57 sınıf için)
        probs = box.conf.tolist() 
        # Geometrik özellikler
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        aspect_ratio = (x2 - x1) / (y2 - y1)
        area = (x2 - x1) * (y2 - y1)
        
        return probs + [aspect_ratio, area]

    def predict_hybrid(self, image_path):
        """YOLO + XGBoost Hibrit Tahmin"""
        self.load_models()
        features = self.get_features(image_path)
        if features is None:
            return "Nesne Tespit Edilemedi"
        
        # XGBoost tahmini
        prediction = self.xgb_model.predict(np.array([features]))
        class_id = int(prediction[0])
        return self.yolo_model.names[class_id]

    def train_yolo(self, data_yaml):
        """Sadece YOLO Eğitimi"""
        model = YOLO("yolo11n.pt")
        model.train(data=data_yaml, epochs=100, imgsz=640, device=0)
    def predict_video(self, video_path, output_path="output_result.mp4", show_live=True):
        """
        Video dosyasını okur, her kareyi Hibrit modelle işler ve sonucu kaydeder.
        """
        self.load_models()
        cap = cv2.VideoCapture(video_path)
        
        # Video özelliklerini al
        width  = int(cap.get(cv2.W_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.W_CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.W_CAP_PROP_FPS))
        
        # Kayıt yapıcı (VideoWriter)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"[INFO] Video işleniyor: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. YOLO tespiti (Her kare için)
            results = self.yolo_model.predict(frame, verbose=False)
            result = results[0]

            for box in result.boxes:
                # 2. Hibrit Özellikleri Çıkar (Probs + Geometry)
                probs = box.conf.tolist()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                area = (x2 - x1) * (y2 - y1)
                
                features = probs + [aspect_ratio, area]

                # 3. XGBoost Onayı
                prediction = self.xgb_model.predict(np.array([features]))
                hybrid_label = self.yolo_model.names[int(prediction[0])]
                conf_score = float(box.conf[int(prediction[0])]) if len(box.conf) > int(prediction[0]) else float(box.conf.max())

                # 4. Görselleştirme (Kutu ve Yazı çiz)
                color = (0, 255, 0) # Yeşil
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"Hybrid: {hybrid_label} ({conf_score:.2f})"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Çerçeveyi kaydet ve (opsiyonel) göster
            out.write(frame)
            if show_live:
                cv2.imshow('Hybrid Traffic Sign Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"[BİTTİ] İşlenmiş video kaydedildi: {output_path}")
    def train_xgb(self, csv_path):
        """YOLO'dan gelen verilerle XGBoost Eğitimi"""
        df = pd.read_csv(csv_path)
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, objective='multi:softprob')
        clf.fit(X_train, y_train)
        
        joblib.dump(clf, self.xgb_path)
        print("XGBoost Eğitimi Tamamlandı ve Kaydedildi.")
class BenchmarkEngine:
    def __init__(self, yolo_path="best.pt", xgb_path="xgboost_refiner.joblib"):
        self.yolo = YOLO(yolo_path)
        self.xgb = joblib.load(xgb_path)
        self.yolo_correct = 0
        self.hybrid_correct = 0
        self.total_images = 0

    def get_features(self, results):
        """YOLO sonucundan XGBoost özelliklerini çıkarır."""
        if len(results[0].boxes) == 0: return None
        box = results[0].boxes[0]
        probs = box.conf.tolist()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        return probs + [(x2 - x1) / (y2 - y1), (x2 - x1) * (y2 - y1)]

    def run_test(self, folder_path):
        # Klasördeki görselleri al (1.png, 2.png...)
        images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
        self.total_images = len(images)
        
        print(f"\n[INFO] {self.total_images} görsel analiz ediliyor...\n")

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            
            # --- GERÇEK ETİKET (DOSYA ADINDAN VEYA KLASÖRDEN ALINMALI) ---
            # Örn: 'dur_1.png' ise etiket 'dur' olur. 
            # Biz burada dosya isminin başında class_id olduğunu varsayıyoruz (Örn: 5_resim.png)
            try:
                true_label_id = int(img_name.split('_')[0]) 
            except:
                true_label_id = 0 # Varsayılan (Senin isimlendirmene göre burayı düzelt)

            # 1. Klasik YOLO Tahmini
            yolo_res = self.yolo.predict(img_path, verbose=False)
            if len(yolo_res[0].boxes) > 0:
                yolo_pred = int(yolo_res[0].boxes[0].cls[0])
                if yolo_pred == true_label_id: self.yolo_correct += 1

            # 2. Hibrit (Fusion) Tahmini
            features = self.get_features(yolo_res)
            if features:
                hybrid_pred = int(self.xgb.predict(np.array([features]))[0])
                if hybrid_pred == true_label_id: self.hybrid_correct += 1

        self.print_report()
    def print_report(self):
        yolo_acc = (self.yolo_correct / self.total_images) * 100
        hybrid_acc = (self.hybrid_correct / self.total_images) * 100
        
        print("\n" + "="*50)
        print(f"{'MODEL':<25} | {'DOĞRU':<10} | {'BAŞARI'}")
        print("-" * 50)
        print(f"Classical YOLOv11       | {self.yolo_correct:<10} | %{yolo_acc:.1f}")
        print(f"Hybrid Fusion System    | {self.hybrid_correct:<10} | %{hybrid_acc:.1f}")
        print("="*50)
        print(f"Gelişim: %{hybrid_acc - yolo_acc:.2f} puan artış sağlandı.")   
        
def main():
   parser = argparse.ArgumentParser(description="Traffic Sign Hybrid Fusion System")
    # Sadece input parametresi alıyoruz, mode artık sabit (predict_hybrid)
    parser.add_argument("--input", required=True, help="Analiz edilecek görselin yolu")
    
    args = parser.parse_args()
    system = TrafficSignSystem()

    # Doğrudan hibrit tahmini çalıştır
    result = system.predict_hybrid(args.input)
    
    print("\n" + "="*30)
    print(f" ANALİZ SONUCU: {result}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()