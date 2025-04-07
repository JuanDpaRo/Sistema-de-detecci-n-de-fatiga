import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from collections import deque
from parpadeos import detect_blinks

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        raise RuntimeError("Failed to open camera.")
    
    return cap

def mouth_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))  # Parte superior-inferior
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))  # Parte central-superior
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))  # Izquierda-derecha
    return (d_A + d_B) / (2 * d_C)
    
def plotting_ear(timestamps, pts_ear, amps, line1, line2, thresh_line=None):
    """ Genera la gráfica en tiempo real del EAR y la amplitud histórica """
    global figure  

    if not line1:  
        plt.style.use("ggplot")
        plt.ion()
        figure, ax = plt.subplots()
        line1, = ax.plot(timestamps, pts_ear, label="MAR", color='r')
        #line2, = ax.plot(timestamps, amps, label="Amp", color='b')
        plt.ylim(0.1, 0.9)
        plt.ylabel("MAR", fontsize=14)
        plt.xlabel("Tiempo (s)", fontsize=14)
        plt.legend(loc="upper right")
    else:  # Si ya está inicializada, actualizar datos
        line1.set_xdata(timestamps)
        line1.set_ydata(pts_ear)
        #line2.set_xdata(timestamps)
        #line2.set_ydata(amps)
        plt.xlim(min(timestamps), max(timestamps))  # Ajustar el eje X dinámicamente
        figure.canvas.draw()
        figure.canvas.flush_events()

    return line1, line2, thresh_line

index_mouth = [61, 37, 267, 291, 314, 84]# Índices de la boca en MediaPipe
cap = init_camera()# Inicializar la cámara
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
             

                # Dibujar los puntos de los boca 
                for pt in mounth_pts:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

                # Calcular MAR
                mar = mouth_aspect_ratio(mounth_pts)

                # Agregar datos a las listas
                pts_ear2.append(mar)
                pts_ear.append(mar)
                timestamps.append(time.time() - start_time)  # Tiempo transcurrido en segundos
                amp = 0
                amps.append(0)
                
                # Actualizar la gráfica
                if i > 20:
                    line1, line2, thresh_line = plotting_ear(timestamps, pts_ear, amps, line1, line2, thresh_line)
                i += 1
                
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

cap.release()
cv2.destroyAllWindows()
