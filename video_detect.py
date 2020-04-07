# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:57:08 2018
@author: 64186_GYN
tensorflow-gpu1.2.0
keras 2.1.1
cuda8
cudnn5.1
"""
import os
import numpy as np
import cv2
from train_CNN import Model
from NameOfSigns import SignNames
import time
from timeit import default_timer as timer
import xlwt
import xlsxwriter


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CTWSCascade = cv2.CascadeClassifier("cascade.xml")
CTWSCascade.load('./cascade/cascade.xml')
model = Model()
model.load_model(
    file_path='./warningSigns.model8.h5')

numofsigs=0
totalnum=0
avgtime=0
cap = cv2.VideoCapture("./testVideo.mp4")
index=0
a=0
totaltime=0
curr_fps=0
xl = xlsxwriter.Workbook(r'D:\1.xlsx')
sheet = xl.add_worksheet('sheet1')

video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter("./videoResult.mp4", video_FourCC, video_fps, video_size)

while True:
        index+=1
        a+=1
        ret, frame = cap.read()
        if ret!=True:
            break
        start = time.time()
        #gray = cv2.GaussianBlur(frame, (3, 3), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = CTWSCascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=1, minSize=(40, 40), maxSize=(80,80))
        for (x, y, w, h) in rect:
            image = frame[y - 5: y + h + 5, x - 5: x + w + 5]
            prob, ID = model.predict(image)
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
            if(prob[ID]>0.85):
                numofsigs=numofsigs+1
                signname = SignNames()
                result = str(ID) + signname.getName(ID)
                cv2.putText(frame, result,
                        (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255), 
                        2)
        end = time.time()
        totaltime+=end - start
        if a==30:
            curr_fps = 1 / (totaltime / 30)
            totaltime = 0
            a = 0
        curr_fps=int(curr_fps)
        fps = "FPS: " + str(curr_fps)
        print(curr_fps,' ',end - start)
        FPSwriter=curr_fps
        cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.50, color=(255, 0, 0), thickness=2)
        sheet.write(0, index, FPSwriter)

        if ret == True:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", frame)
        else:
            break

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end = time.time()

        timegap=end-start
        if (timegap < 1):
            avgtime+=timegap

xl.close()



