import cv2

import numpy as np
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode=False, maxHands=2, detectionconf=0.5,trackcon=0.5):
        self.mode= mode
        self.maxHands= maxHands
        self.detectionCon= detectionconf
        self.trackCon = trackcon
        self.mpmanos= mp.solutions.hands

        self.manos = self.mpmanos.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)

        self.mppuntos= mp.solutions.drawing_utils

    def findHands(self, imagencamara, Draw=True):
        imagenRGB= cv2.cvtColor(imagencamara, cv2.COLOR_BGR2RGB)

        self.resultados= self.manos.process(imagenRGB)

    
        if self.resultados.multi_hand_landmarks:

            for hands in self.resultados.multi_hand_landmarks:
                    #for id,lam in enumerate(hands.landmark):
                     #   h,w,c = imagencamara.shape
                     #   cx,cy = int(lam.x*w),int(lam.y*h)
                
                      #  if id==8:
                       #     cv2.circle(imagencamara, (cx,cy), 9, (255,0,0),cv2.FILLED)
                if Draw:
                    self.mppuntos.draw_landmarks(imagencamara,hands,self.mpmanos.HAND_CONNECTIONS)
        return imagencamara
    def findPosicion(self,imagencamara,handNu=0,draw=True):
        lamlist=[]
        if self.resultados.multi_hand_landmarks:
            misHands= self.resultados.multi_hand_landmarks[handNu]
            for id,lam in enumerate(misHands.landmark):
                h,w,c = imagencamara.shape
                cx,cy = int(lam.x*w),int(lam.y*h)
                lamlist.append([id,cx,cy])
                
                if draw:
                  cv2.circle(imagencamara, (cx,cy), 9, (255,0,0),cv2.FILLED)
        return lamlist



### estas lineas no se usan, solo para prueba del modulo


def main():
    
## codigo de prueba en el ciclo principal del modulo
    if __name__ == "__main__":
        main()
