import cv2
import numpy as np
import math as mt

def findMaxContour(contours):
    #Max alanı kaplayan contours u bulup gereksiz kısımları kaldırıcaz
    max_i=0#indice
    max_area=0#alan

    for i in range(len(contours)):
        area_face=cv2.contourArea(contours[i])
        if max_area<area_face:
            max_area=area_face
            max_i=i
    try:
        c=contours[max_i]
    except:
        contours=[0]
        c=contours[0]
        print("fonskyion hata")

    return c

cap=cv2.VideoCapture(0)

while 1:
    _,frame=cap.read()
    
    #frame imizi 1 yani y eksenine göre yansıttık
    frame=cv2.flip(frame,1)
    
    #Koordinatlar
    y1=100
    y2=350
    x1=225
    x2=400
    
    #Region of interest alanımız 
    roi=frame[y1:y2,x1:x2]

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),0)  
    
    #Skin Color u daha iyi yakalayabilmek için hsv renk uzayına çevirdik
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    #Upper and Lower hue saturation value
    #Kendi renk teninize göre oynamalar yapabilirsiniz
    lower = np.array([0, 30, 60], dtype="uint8")
    upper = np.array([23, 150, 255], dtype="uint8")
    
    #Frame i maskeledik
    mask=cv2.inRange(hsv,lower,upper)
    
    #Kernel 1 lerden oluşan dönüşüm dizeyi matris boyutu arttıkça beyazları yayar
    kernel=np.ones((9,9),dtype=np.uint8)
    
    mask=cv2.dilate(mask,kernel,iterations=1)
    
    #Yumuşatma
    mask=cv2.medianBlur(mask,15)
    
    #Noktalardan oluşan sınır çizgilerimiz bulur
    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    if len(contours)>0:
        
        try:
            #Max alanı kaplayan contours u bulup gereksiz kısımları kaldırıcaz
            c=findMaxContour(contours)
            
            #extLeft=tuple(c[c[::,0].argmin()][0]) x i en küçük olan x ve y leri döner
            extLeft=tuple(c[c[::,0].argmin()][0])
            extRight = tuple(c[c[:,:, 0].argmax()][0])
            extTop = tuple(c[c[:,:, 1].argmin()][0])#0 x lere 1 y lere bakar
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            cv2.circle(roi,extLeft,5,(0,255,0),2)
            cv2.circle(roi, extRight, 5, (0, 255, 0), 2)
            cv2.circle(roi, extTop, 5, (0, 255, 0), 2)
            cv2.circle(roi, extBot, 5, (0, 255, 0), 2)

            cv2.line(roi,extLeft,extTop,(0,0,255),2,3)
            cv2.line(roi, extTop, extRight, (0, 0, 255), 2, 3)
            cv2.line(roi, extRight, extBot, (0, 0, 255), 2, 3)
            cv2.line(roi, extBot, extLeft, (0, 0, 255), 2, 3)
   
            #Kenarları hesaplama açı bulmak için
            a=mt.sqrt((extRight[0]-extTop[0])**2+(extRight[1]-extTop[1])**2)
            b = mt.sqrt((extBot[0] - extRight[0]) ** 2 + (extRight[1] - extBot[1]) ** 2)
            c = mt.sqrt((extBot[0] - extTop[0]) ** 2 + (extBot[1] - extTop[1]) ** 2)
            d = mt.sqrt((extLeft[0] - extTop[0]) ** 2 + (extLeft[1] - extTop[1]) ** 2)
            
            #Kenarların 0 olma ihtimaline karşılık try excep bloğu
            try:
                angle_ab = mt.acos((a ** 2 + b ** 2 - 2*c ** 2) / (a * b * c))*57
                cv2.putText(roi,str(angle_ab),(extRight[0]-100,extRight[1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2,cv2.LINE_AA)
            except:
                print("kenarlardan biri sıfır")
        except:
            print("hata")
    else:
        print("no contour")
        
    cv2.imshow("frame", frame)
    cv2.imshow("mask",mask)
    cv2.imshow("roi",roi)
    
    if cv2.waitKey(10)&0xFF==ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()
