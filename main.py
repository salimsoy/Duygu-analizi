import cv2
from deepface import DeepFace
import pandas as pd
import datetime
from hascade import FaceDetection
import matplotlib.pyplot as plt

class SentimentAnalysis:
    
    def __init__(self):
        self.feeling_log = []
        self.end_feel = "Analiz ediliyor..."
        self.tr_ceviri = {
                    'angry': 'Kizgin', 'disgust': 'Tiksinti', 'fear': 'Korku',
                    'happy': 'Mutlu', 'sad': 'Uzgun', 'surprise': 'Saskin', 'neutral': 'Dogal'
                }
    def saves(self):
        self.df = pd.DataFrame(self.feeling_log)
        self.df.to_csv('gunluk_rapor.csv')
        print("Rapor kaydedildi.")
        
    
    def drawing(self):
        try:
            plt.figure(figsize=(8, 5))
            self.df['Duygu'].value_counts().plot(kind='bar', color='orange')
            plt.title('Duygu Analizi Sonuçları')
            plt.savefig('analiz_sonucu.png')
            plt.show()

        except Exception as e:
            print(e)
            
        
    def deep_face(self,frame):
        try:
            analiz = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analiz, list):
                self.end_feel = analiz[0]['dominant_emotion']
            else:
                self.end_feel = analiz['dominant_emotion']
            
            self.end_feel = self.tr_ceviri.get(self.end_feel, self.end_feel)
            cv2.putText(frame, self.end_feel, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.feeling_log.append({"Zaman": self.time, "Duygu": self.end_feel})
            print(f"[{self.time}] Algılanan: {self.end_feel}")

        except:
            print("yüz algılanamadı")
        
    def main(self):
        
        haarcascade = FaceDetection()
        
        self.vote = input("webcam ile devam etmek icin 1 e vidyo ile devam etmek icin 2 ye tıklayın: ")
        
        if self.vote == "1":
            decision = 0
        elif self.vote == "2":
            decision =  input("dosya yolunu giriniz:")
        
        else:
            print("hatalı secim")
            exit()
        
        cap = cv2.VideoCapture(decision)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.vote == "1":
                self.time = datetime.datetime.now().strftime("%H:%M:%S")
            elif self.vote == "2":
                milisaniye = cap.get(cv2.CAP_PROP_POS_MSEC)
                top_second = milisaniye / 1000
                minute = int(top_second // 60)
                second = int(top_second % 60)
                
                self.time = f"{minute}:{second}"
            
            frame1 = haarcascade.main(frame)
            if len(haarcascade.face_rect) > 0:
                self.deep_face(frame)
            else:
                print("yuz algilaniyor...")
            
            
           
            cv2.imshow('Duygu Analizi', frame)
            cv2.imshow('Yuz Algilama', frame1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
            
        self.saves()
        
        self.drawing()
        
        
if __name__ == "__main__":
    proses = SentimentAnalysis()
    proses.main()

    
        
    

