import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from collections import deque
from parpadeos import detect_blinks


def init_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")    
    return cap

def mouth_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5])) 
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))  
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))  
    return (d_A + d_B) / (2 * d_C)


mp_face_mesh = mp.solutions.face_mesh# Inicializar MediaPipe Face Mesh
index_mouth = [61, 37, 267, 291, 314, 84]#Puntos de la boca
cap = init_camera()# Inicializar la cámara
counter= 0

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                mounth_pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in index_mouth]

                # Calcular MAR
                mar = mouth_aspect_ratio(mounth_pts)
                #Detectar Bostezo
                if mar > 0.6:
                    if yawn_start_time is None:  # Inicio del bostezo
                        yawn_start_time = time.time()
                    elif time.time() - yawn_start_time > 3:  # Más de 3 segundos
                        counter += 1
       
                        yawn_start_time = None  # Reiniciar tiempo de bostezo
                else:
                    yawn_start_time = None  # Reiniciar si no está por encima del umbral
 
                # Mostrar los valores en la ventana de la cámara
                cv2.putText(frame, f"Bostezos: {counter}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        # Mostrar el frame con OpenCV
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

