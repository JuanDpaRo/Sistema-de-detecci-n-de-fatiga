import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from collections import deque
from parpadeos import detect_blinks

"""Función para iniciar camara """
def init_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
    return cap
    
"""Función para Calcular la relación de aspecto del ojo (EAR) """
def eye_aspect_ratio(coordinates):

    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)
    
"""Calcula un nuevo EAR_THRESH basado en el promedio de los últimos valores de EAR."""
def update_ear_thresh(ear_history):

    if len(ear_history) > 0:
        return np.mean(ear_history)
    else:
        return 0.22  # Umbral predeterminado si no hay datos aún
        

mp_face_mesh = mp.solutions.face_mesh # Inicializar MediaPipe Face Mesh       
# Índices de los ojos en MediaPipe
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
cap = init_camera()# Inicializar la cámara
pts_ear = deque(maxlen=64)
start_time = time.time()
MAX_EAR_LIMIT = 0.37
i = 0
blink_counter = 0
blink_durations = []
ear_history = deque(maxlen=5)  # Últimos valores de EAR
blink_start_time = None  # Tiempo de inicio del parpadeo
aux_counter=0
ear_history = deque(maxlen=5) # Cola para los últimos 5 valores de EA

"""Bucle principal """
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
                left_eye_pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in index_left_eye]
                right_eye_pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in index_right_eye]
                
                # Calcular EAR
                ear_left = eye_aspect_ratio(left_eye_pts)
                ear_right = eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2

                # Detectar parpadeos
                blink_counter, blink_durations = detect_blinks(ear)
                ear_history.append(ear)
                EAR_THRESH2 = update_ear_thresh(ear_history)
                EAR_THRESH = EAR_THRESH2 - 0.035  

                if ear > MAX_EAR_LIMIT:
                    EAR_THRESH = 0.1

                if ear < EAR_THRESH:               
                    if n == 1:  # Guardar el valor de EAR_LIM como umbral para el tiempo de parpadeo
                        aux_counter = EAR_THRESH
                        EAR_LIM = aux_counter
                        n = 0
                
                   if blink_start_time is None:
                        blink_start_time = cv2.getTickCount()  # Guardar tiempo de inicio

                if ear > aux_counter:
                    n = 1                   
                    if blink_start_time is not None:
                        blink_duration = (cv2.getTickCount() - blink_start_time) / cv2.getTickFrequency()
                        if 0.100 <= blink_duration <= 0.5:
                             blink_counter += 1
                             blink_durations.append(blink_duration)               
                        blink_start_time = None

                if len(blink_durations) > 0:
                    avg_blink_duration = sum(blink_durations) / len(blink_durations)
                else:
                    avg_blink_duration = 0

                cv2.putText(frame, f"Parpadeos: {blink_counter}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Duracion Prom. Parpadeo: {avg_blink_duration:.4f}s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presionar 'ESC' para salir
            break

cap.release()
cv2.destroyAllWindows()

