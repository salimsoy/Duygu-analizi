import cv2

class FaceDetection:
    def adjusted_detect_face(self,img1, face_cascade):
        face_img = img1.copy()
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        self.face_rect = face_cascade.detectMultiScale(gray, 
                                                  scaleFactor=1.2, 
                                                  minNeighbors=5)
        if len(self.face_rect) == 0:
            return face_img
        
        for (x, y, w, h) in self.face_rect:
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
        self.center_x = x + w // 2
        self.center_y = y + h // 2
            
        return face_img
    
    def __init__(self):
        self.face_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    
    def main(self, img):
            
        face_img = self.adjusted_detect_face(img, self.face_cascade)
        return face_img
         
    