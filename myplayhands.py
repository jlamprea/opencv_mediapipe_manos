## muestra los puntos de las manos con una clase aparte
## archivo de corre en entorno virtual (manos), no funciona en consola el .py
## usa python 3.7.8 64 bit como interprete
## se debe crear primero en env, luedo dentro de la carpeta activarlo con \scripts.\activate
## se desactiva el env con deactivate
## si tiene problemas con politicas de usuario
##ejecutar comando en consola set-executionPolicy - scope currentuser unrestricted si 
## se confirma con get-executionpolicy se debe ver unrestricted
## se inicia GIT en el icono, lo mejor es antes de hacer el archivo

import cv2

import numpy as np
##se importa el modulo mde google mediapipe que se baja con pip mediapipe
import mediapipe as mp
import time
## se importa el modulo creado que es un archivo .py con varias funciones
import Hand_Module as HM


currentTime=0

previousTime=0
cap= cv2.VideoCapture(0)
## se crea una instancia al modulo creado
detector= HM.HandDetector()


while True:

    sucssess, imagencamara= cap.read()
    ## se usa la funcion del modulo creado para detectar manos
    imagencamara= detector.findHands(imagencamara, Draw=True)
    ## se hace una lista de los puntos de las manos para dibujarlas
    lamlist= detector.findPosicion(imagencamara, draw=True)
     ## se indica el punto que se quiere imprimir con su ubucacion   
    if len(lamlist) !=0:
       print(lamlist[8])
    ## se calcula el tiempo de lo FPS y se imprime
    currentTime=time.time()

    fsp= 1/(currentTime-previousTime)

    previousTime= currentTime

    cv2.putText(imagencamara, str(int(fsp)),(10,78), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255),3)
    # se muestra la imagen con los datos
    cv2.imshow("imagen", imagencamara)
    # se espera a cancelar con una tecla o Ctrl + C en consola
    cv2.waitKey(1)