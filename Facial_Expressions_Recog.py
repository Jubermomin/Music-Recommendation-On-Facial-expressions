from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\Juber Momin\Downloads\Facial Expression project\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Juber Momin\Downloads\Facial Expression project\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"C:\Users\Juber Momin\Downloads\WhatsApp Video 2022-06-12 at 2.03.47 PM.mp4")


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import webbrowser

url = "https://www.youtube.com/search?q={} mood songs and shayari".format(label)
print("Your current mood is {} ".format(label))
webbrowser.open(url)

import random
import os

if label == 'Angry':
    n = random.randint(0, 20)
    print(n)

    music_dir = r'C:\Users\Juber Momin\Downloads\Facial Expression project\songs\angry'
    song = os.listdir(music_dir)
    print(song)

    os.startfile(os.path.join(music_dir, song[n]))
    print("Currently Playing Song:  ",song[n])
elif label == 'Happy':
    n = random.randint(0, 30)
    print(n)

    music_dir = r'C:\Users\Juber Momin\Downloads\Facial Expression project\songs\happy'
    song = os.listdir(music_dir)
    print(song)

    os.startfile(os.path.join(music_dir, song[n]))
    print("Currently Playing Song:  ",song[n])
elif label == 'Neutral':
    n = random.randint(0, 20)
    print(n)

    music_dir = r'C:\Users\Juber Momin\Downloads\Facial Expression project\songs\neutral'
    song = os.listdir(music_dir)
    print(song)

    os.startfile(os.path.join(music_dir, song[n]))
    print("Currently Playing Song:  ",song[n])
elif label == 'Sad':
    n = random.randint(0, 20)
    print(n)

    music_dir = r'C:\Users\Juber Momin\Downloads\Facial Expression project\songs\sad'
    song = os.listdir(music_dir)
    print(song)

    os.startfile(os.path.join(music_dir, song[n]))
    print("Currently Playing Song:  ",song[n])
elif label == 'Surprise':
    n = random.randint(0, 20)
    print(n)

    music_dir = r'C:\Users\Juber Momin\Downloads\Facial Expression project\songs\surprise'
    song = os.listdir(music_dir)
    print(song)

    os.startfile(os.path.join(music_dir, song[n]))
    print("Currently Playing Song:  ",song[n])
else:

    print("NO")

cap.release()
cv2.destroyAllWindows()


























