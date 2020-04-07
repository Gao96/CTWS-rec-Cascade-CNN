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
from chineseNameOfSigns import SignNames
import time

# 加载分类器
CTWSCascade = cv2.CascadeClassifier("cascade.xml")
CTWSCascade.load('./Cascade/cascade.xml')
model = Model()
model.load_model(
    file_path='./warningSigns.model8.h5')

numofsigs=0
totalnum=0
avgtime=0

allright=0
allwrong=0
allno=0

for xxx in range(0,42):
  right = 0
  rightimgname = []
  wrong = 0
  wrongimgname=[]
  no = 0
  noimgname=[]

  print("##############",xxx)
  imgdir='D:\\TSRtest\\image\\'+str(xxx)+'\\'
  GTdir='D:\\TSRtest\\grandTruth\\'+str(xxx)+'\\'
  dst='D:\\TSRtest\\res_cas\\'+str(xxx)+'\\'
  if not os.path.exists(dst):
      os.makedirs(dst)
  for root, dirs, files in os.walk(imgdir):
    for file in files:
        img_path = root + file
        print(img_path)
        name=file.split('.')[0]
        GTpath=GTdir+name+'.txt'
        with open(GTpath, "r") as f:
            GT=f.readline()
        GTclass=GT.split(';')[0]
        GTx=GT.split(';')[1]
        GTy=GT.split(';')[2]
        GTw=GT.split(';')[3]
        GTh=GT.split(';')[4]
        print(GTclass,GTx,GTy,GTw,GTh)
        img = cv2.imread(img_path)
        frame = img
        start = time.time()
        #gray = cv2.GaussianBlur(frame, (3, 3), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = CTWSCascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=2, minSize=(2, 2))
        index = 0
        print(rects)
        if len(rects)==0:
            no +=1
            noimgname.append(name)
            continue

        maxprob=0
        finalloc=[0,0,0,0]
        finalclass=0
        finalprob=0

        for rect in rects:
            x=rect[0]
            y=rect[1]
            w=rect[2]
            h=rect[3]
            index+=1
            image = frame[y - 5: y + h + 5, x - 5: x + w + 5]
            prob, ID = model.predict(image)

            if prob[ID]>maxprob:
                finalloc=rect
                finalclass=ID
                finalprob=prob[ID]

        x = finalloc[0]
        y = finalloc[1]
        w = finalloc[2]
        h = finalloc[3]

        if(finalprob>0):
                cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
                numofsigs=numofsigs+1
                signname = SignNames()
                result = str(finalclass) + signname.getName(finalclass)+str(finalprob)
                cv2.putText(frame, result,  # 显示标签
                        (x + 10, y + 10),  # 坐标
                        cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                        0.5,  # 字号
                        (255, 0, 255),  # 颜色
                        1)  # 字的线宽
                if finalclass==int(GTclass):
                    right+=1
                else:
                    wrong+=1
                    wrongimgname.append(name)
        else:
            no += 1
            noimgname.append(name)



        end = time.time()

        timegap=end-start
        if (timegap < 1):
            avgtime+=timegap
        print("has recognized num:"+str(numofsigs))
        print(timegap)
        cv2.imwrite(dst+str(file),frame)
  acc=right/(right+wrong+no)
  rs="Acc 0f sign "+str(xxx)+" is:"+str(acc)+' and rightnum is '+str(right)+str(rightimgname)+" wrongnum is "+str(wrong)+str(wrongimgname)+" noimgnum is "+str(no)+str(noimgname)+'\n'
  print(rs)
  with open('D:\\TSRtest\\res_cas\\log.txt', "a") as f:
      f.write(rs)
      f.close()
  allright+=right
  allwrong+=wrong
  allno+=no

allacc=allright/( allwrong+allno+allright)
ars="$$$$$$$$$$$$$ totall acc is "+str(allacc)+" and allright is "+str(allright)+" allwrong is "+str(allwrong)+" allno is "+str(allno)+'\n'
print("$$$$$$$$$$$$$ totall acc is "+str(allacc)+" and allright is "+str(allright)+" allwrong is "+str(allwrong)+" allno is "+str(allno))
with open('D:\\TSRtest\\res_cas\\log.txt', "a") as f:
    f.write(ars)
    f.write('\n')
    f.close()
avgtime=avgtime/2100
print("avgtime: ",avgtime)

cv2.destroyAllWindows()
