import pyttsx3
import keyboard
import numpy as np
import cv2
import os
import face_recognition
from datetime import datetime, date

studImg = []
studNames = []

path = r"/home/pi/project/data"
print(path)
studImageList = os.listdir(path)

for file in studImageList:
    curStudImg = cv2.imread(f'{path}/{file}')
    studImg.append(curStudImg)
    studNames.append(os.path.splitext(file)[0])


def findEncodings(studImg):
    encodeList = []
    print(len(studImg))
    for img in studImg:
        # print(img)
        if (img is not None):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        if len(encode) > 0:
            encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance/Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            today = date.today()
            d2 = today.strftime("%B %d, %Y")
            today = str(today)
            f.writelines(f'\n{name},{dtString},{d2}')

encodedStudImg = findEncodings(studImg)
Vcap = cv2.VideoCapture(1)
past = ""
if not Vcap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    print(Vcap)
    success, frame = Vcap.read()
    print(Vcap.read())
    #cv2.imshow("show", frame)
    sFrame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    sFrame = cv2.cvtColor(sFrame, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(sFrame)
    encodedCurFrame = face_recognition.face_encodings(sFrame, facesCurFrame)
    for encodedFace, faceLoc in zip(encodedCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodedStudImg, encodedFace)
        distance = face_recognition.face_distance(encodedStudImg, encodedFace)
        matchIndex = np.argmin(distance)

        if matches[matchIndex]:
            name = studNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 + 4, x2 + 4, y2 + 4, x1 + 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
            present = name
            if present != past:
                engine = pyttsx3.init()
                engine.say(name)
                engine.runAndWait()
                past = present

        #cv2.imshow("show", frame)
    #cv2.waitKey(1)
    if keyboard.is_pressed("q"):
        exit()
