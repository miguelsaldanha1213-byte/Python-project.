import cv2
import sys
import os

def process_identity(image_path):
    if not os.path.exists(image_path):
        print("FALSE")
        return

    face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    
    if image is None:
        print("FALSE")
        return

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = face_data.detectMultiScale(grayscale, 1.3, 5)

    if len(detections) > 0:
        print("TRUE")
    else:
        print("FALSE")

if __name__ == "__main__":
    process_identity(sys.argv[1])
