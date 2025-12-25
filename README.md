# Akıllı Yüz İfadesi Analizi ve Duygu Tanıma Sistemi
Bu projede, gerçek zamanlı olarak çalışan bir sistem geliştirdik. Alınan görüntüler üzerinde yüz tespiti yaptık. Tespit edilen yüzlerdeki ifadeleri analiz ederek hangi duyguların (mutlu, üzgün, şaşkın, kızgın vs.) ön planda olduğunu tahmin ettik. Sonrasında bu tahminleri kullanıcıya grafiklerle gösteriyoruz.
Bu projede görüntü işleme ve derin öğrenme tabanlı iki temel yaklaşım kullanılmaktadır:

## Haar Cascade Algoritması 
Haar Cascade sınıflandırıcıları, nesne tespiti için makine öğrenimi tabanlı bir yöntemdir. Bir
sınıflandırıcıyı eğitmek için bir dizi pozitif ve negatif görüntü kullanırlar, daha sonra bu
sınıflandırıcı yeni görüntülerdeki nesneleri tespit etmek için kullanılır.

## DeepFace
Facebook, Google ve VGG-Face gibi modelleri barındıran, yüz tanıma ve duygu analizi (kızgın, mutlu, üzgün vb.) yapabilen gelişmiş bir derin öğrenme (Deep Learning) kütüphanesidir.

## Pandas & Matplotlib
Elde edilen duygu verilerini zaman damgasıyla birlikte yapısal olarak saklamak (CSV) ve analiz bitiminde grafiksel olarak görselleştirmek için kullanılır.

** Temel Mantık **
Kod, kullanıcıdan alınan girdiye (Webcam veya Video) göre kare kare (frame-by-frame) işleme yapar ve aşağıdaki adımları izler:

- Hazırlık: Haar Cascade sınıflandırıcısı ve DeepFace modelleri belleğe yüklenir.
- Yüz Tespiti (Ön İşleme): Gelen görüntü önce FaceDetection sınıfına gönderilir. Eğer karede bir yüz yoksa, sistem gereksiz yere ağır olan Duygu Analizi işlemini yapmaz. Bu, işlemci yükünü ciddi oranda azaltır.
- Duygu Analizi: Eğer bir yüz tespit edilirse (face_rect > 0), görüntü DeepFace kütüphanesine gönderilir ve baskın duygu (dominant emotion) analiz edilir.
- Çeviri ve Görselleştirme: İngilizce dönen duygu sonuçları (happy, angry vb.) Türkçe karşılıklarına çevrilir ve ekrana yazdırılır.
- Kayıt: Tespit edilen her duygu, zaman bilgisiyle birlikte bir listeye eklenir.
- Raporlama: İşlem sonlandırıldığında (q tuşu ile), veriler gunluk_rapor.csv dosyasına kaydedilir ve duygu dağılımını gösteren bir sütun grafiği (analiz_sonucu.png) oluşturulur.

Avantajları 
- Performans Optimizasyonu: Yüzün algılanmadığı karelerde DeepFace çalıştırılmaz, böylece sistem kaynakları boşuna tüketilmez.
- Veri Raporlama: Sadece anlık görüntü vermez, süreç boyunca kişinin hangi duygularda olduğunu Excel/CSV formatında kaydeder.
- Görsel Analiz: Program sonunda oluşturulan grafik ile duygu durumunun genel özeti tek bakışta görülebilir.
- Esneklik: Hem canlı kamera görüntüsü hem de hazır video dosyaları üzerinde çalışabilir.

Kodun Çalışma Prensibi

Uygulama iki ana sınıftan oluşur:

FaceDetection Sınıfı:
- OpenCV'nin haarcascade_frontalface_default.xml dosyasını kullanır.
- Görüntüyü gri tonlamaya çevirir ve yüz koordinatlarını bulur.
- Yüzün etrafına çerçeve çizer.

SentimentAnalysis Sınıfı:
- Ana döngüyü yönetir.
- Kullanıcıdan "Kamera" veya "Video Dosyası" seçimini alır.
- DeepFace ile duyguyu tahmin eder ve tr_ceviri sözlüğü ile Türkçeleştirir.
- Sonuçları Pandas DataFrame'e çevirip kaydeder.

Aşağıda Python kodu ve açıklamaları yer almaktadır:

1. Yüz Tespit Modülü (hascade.py)
Bu modül sadece yüzün yerini bulmakla ilgilenir.
```python
import cv2

class FaceDetection:
    def __init__(self):
        # Haar Cascade modelini yükle
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def adjusted_detect_face(self, img1, face_cascade):
        face_img = img1.copy()
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Yüzü tara
        self.face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        # Yüz yoksa orijinal resmi dön
        if len(self.face_rect) == 0:
            return face_img
        
        # Yüz varsa çerçeve çiz
        for (x, y, w, h) in self.face_rect:
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
            
        return face_img

    def main(self, img):
        face_img = self.adjusted_detect_face(img, self.face_cascade)
        return face_img
```
2. Duygu Analizi ve Ana Döngü (main.py)
Bu modül ana programı çalıştırır, analizi yapar ve raporu oluşturur.

```python
import cv2
from deepface import DeepFace
import pandas as pd
import datetime
from hascade import FaceDetection # Yukarıdaki sınıfı import eder
import matplotlib.pyplot as plt

class SentimentAnalysis:
    def __init__(self):
        self.feeling_log = []
        # Duyguların Türkçe karşılıkları
        self.tr_ceviri = {
            'angry': 'Kizgin', 'disgust': 'Tiksinti', 'fear': 'Korku',
            'happy': 'Mutlu', 'sad': 'Uzgun', 'surprise': 'Saskin', 'neutral': 'Dogal'
        }

    def deep_face(self, frame):
        try:
            # DeepFace ile analiz yap
            analiz = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            # (Kodun geri kalanı dominant duyguyu bulur ve listeye ekler)
            # ...
        except:
            print("Yüz algılanamadı")

    def main(self):
        haarcascade = FaceDetection()
        # Kullanıcı seçimi (Webcam / Video)
        # ...
        
        while True:
            ret, frame = cap.read()
            # Önce Haar Cascade ile yüz var mı kontrol et
            frame1 = haarcascade.main(frame)
            
            # SADECE yüz varsa DeepFace çalıştır (Optimizasyon)
            if len(haarcascade.face_rect) > 0:
                self.deep_face(frame)
            else:
                print("Yüz aranıyor...")
            
            # Sonuçları göster ve 'q' ile çıkış yap
            # ...
        
        # Çıkışta raporları kaydet
        self.saves()
        self.drawing()

if __name__ == "__main__":
    proses = SentimentAnalysis()
    proses.main()
```
