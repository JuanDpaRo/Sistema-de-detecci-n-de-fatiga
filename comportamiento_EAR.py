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
""" Funcion para Calcular la relación de aspecto del ojo (EAR) """
def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)
    
"""Funcion para generar grafica"""
def plotting_ear(timestamps, pts_ear, amps, line1, line2, thresh_line=None):
    global figure  
    if not line1:  # Si la gráfica no está inicializada
        plt.style.use("ggplot")
        plt.ion()
        figure, ax = plt.subplots()
        line1, = ax.plot(timestamps, pts_ear, label="EAR", color='r')
        line2, = ax.plot(timestamps, amps, label="Amp", color='b')
        plt.ylim(0, 0.32)
        plt.ylabel("EAR / Amp", fontsize=14)
        plt.xlabel("Tiempo (s)", fontsize=14)
        plt.legend(loc="upper right")
    else:  # Si ya está inicializada, actualizar datos
        line1.set_xdata(timestamps)
        line1.set_ydata(pts_ear)
        line2.set_xdata(timestamps)
        line2.set_ydata(amps)
        plt.xlim(min(timestamps), max(timestamps))  # Ajustar el eje X dinámicamente
        figure.canvas.draw()
        figure.canvas.flush_events()

    return line1, line2, thresh_line

mp_face_mesh = mp.solutions.face_mesh # Inicializar MediaPipe Face Mesh
cap = init_camera()# Inicializar la cámara

"""Declarar variables"""
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
pts_ear = deque(maxlen=64)
amps = deque(maxlen=64)  
timestamps = deque(maxlen=64)  
start_time = time.time()
i = 0
line1 = []
line2 = []
thresh_line = None
figure = None
ear_max = 0
ear_min = 0
amp = 0
pts_ear2 = deque(maxlen=5)

"""Bucle principal """
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time_fps = time.time()
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_eye_pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in index_left_eye]
                right_eye_pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in index_right_eye]

                # Dibujar los puntos de los ojos 
                for pt in left_eye_pts + right_eye_pts:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

                # Calcular EAR
                ear_left = eye_aspect_ratio(left_eye_pts)
                ear_right = eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2

                # Agregar datos a las listas
                pts_ear2.append(ear)
                pts_ear.append(ear)
                timestamps.append(time.time() - start_time)  # Tiempo transcurrido en segundos

                # Calcular amp y agregar a historial
                ear_max = max(pts_ear2)
                ear_min = min(pts_ear2)
                amp = ear_max - ear_min
                amps.append(amp)
                print(f"EAR: {ear:.4f}, Amp: {amp:.4f}")

                # Actualizar la gráfica
                if i > 20:
                    line1, line2, thresh_line = plotting_ear(timestamps, pts_ear, amps, line1, line2, thresh_line)
                i += 1

                end_time_fps = time.time()
                fps = 1 / (end_time_fps - start_time_fps)
                print(f"FPS: {fps:.2f}")

        # Mostrar el frame con OpenCV
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break

cap.release()
cv2.destroyAllWindows()
