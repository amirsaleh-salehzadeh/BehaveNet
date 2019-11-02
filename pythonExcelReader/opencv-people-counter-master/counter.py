import numpy as np
import time
import imutils
import cv2

avg = None
video = cv2.VideoCapture("1.mp4")
xvalues = list()
yvalues = list()
motion = list()
count1 = 0
count2 = 0


def find_majority(k):
    myMap = {}
    maximum = ('', 0)  # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n, myMap[n])

    return maximum


face_cascade = cv2.CascadeClassifier('C:/Program Files/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

width = 500
while 1:
    ret, frame = video.read()
    flag = True
    text = ""
    frame = imutils.resize(frame, width=width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    cv2.accumulateWeighted(gray, avg, 0.000001)
    
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
    
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for c in cnts:
        print(c)
        if cv2.contourArea(c) < 12000:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        xvalues.append(x)
        yvalues.append(y)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rectagleCenterPont = ((x + x + w) // 2, (y + y + h) // 2)
        cv2.circle(frame, rectagleCenterPont, 1, (0, 0, 255), 5)
        flag = False
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, inNeighbors=5, minSize=(30, 30))
    no_x = len(xvalues)
    no_y = len(yvalues)
    
    if (no_y > 2):
        difference = yvalues[no_y - 2] - yvalues[no_y - 1]
        if(difference > 0):
            motion.append(1)
        else:
            motion.append(0)

    if flag is True:
        if (no_y > 3):
            val, times = find_majority(motion)
            if val == 1 and times >= 3:
                count1 += 1
            else:
                count2 += 1
        yvalues = list()
        motion = list()
    
    cv2.line(frame, (0, 110), (width * 3 // 4, 110), (0, 255, 0), 2)
    cv2.line(frame, (0, 160), (width * 3 // 4, 160), (0, 255, 0), 2)	
    cv2.putText(frame, "In: {}".format(count1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Out: {}".format(count2), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Gray", gray)
    cv2.imshow("FrameDelta", frameDelta)
    
    key = cv2.waitKey(1) & 0xFF
#     time.sleep(0.03)
    if key == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
