import numpy as np
import time
import imutils
import cv2

face_cascade = cv2.CascadeClassifier('C:/Program Files/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
low_cascade = cv2.CascadeClassifier('C:/Program Files/Python37/Lib/site-packages/cv2/data/haarcascade_lowerbody.xml')

video = cv2.VideoCapture("1.mp4")
width = 800
avg = None
while 1:
    ret, frame = video.read()
    flag = True
    text=""
    frame = imutils.resize(frame, width=width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    cv2.accumulateWeighted(gray, avg, 0.000001)
    
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    
#     faces = face_cascade.detectMultiScale(frameDelta,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30))
    low = low_cascade.detectMultiScale(frame, 1.1 , 3)
    
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (12,150,100),2)
    for (x,y,w,h) in low:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (12,150,100),2)
    
#     cv2.line(frame, (0, 110), (width * 3//4,170), (0,255,0), 2)
#     cv2.line(frame, (0, 160), (width * 3//4,220), (0,255,0), 2)	
    cv2.imshow("Frame",frame)
    cv2.imshow("Gray", gray)
    cv2.imshow("FrameDelta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
#     time.sleep(0.03)
    if key == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
