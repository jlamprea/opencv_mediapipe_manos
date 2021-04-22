import cv2

import numpy as np
import mediapipe as mp
import time

## vista de la camara

cap= cv2.VideoCapture(0)


## carga el modelo mediapipe

mpmanos= mp.solutions.hands

manos = mpmanos.Hands()

mppuntos= mp.solutions.drawing_utils

currentTime=0

previousTime=0


while True:

    sucssess, imagencamara= cap.read()

    imagenRGB= cv2.cvtColor(imagencamara, cv2.COLOR_BGR2RGB)

    resultados= manos.process(imagenRGB)

    ## muestra las ubicaciones de los puntos de las manos

    ##print(resultados.multi_hand_landmarks) de las manos y dedos

    if resultados.multi_hand_landmarks:

        for hands in resultados.multi_hand_landmarks:
            for id,lam in enumerate(hands.landmark):
                h,w,c = imagencamara.shape
                cx,cy = int(lam.x*w),int(lam.y*h)
                #print(id,lam)
                #print(id,cx,cy)
                if id==8:
                    cv2.circle(imagencamara, (cx,cy), 9, (255,0,0),cv2.FILLED)

            mppuntos.draw_landmarks(imagencamara,hands,mpmanos.HAND_CONNECTIONS)

    ## calcula cuantos frames por segundo tiene la imagen

    currentTime=time.time()

    fsp= 1/(currentTime-previousTime)

    previousTime= currentTime

    cv2.putText(imagencamara, str(int(fsp)),(10,78), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255),3)

    cv2.imshow("imagen", imagencamara)

    cv2.waitKey(1)
    