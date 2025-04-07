import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import csv
import time
import os
import pandas as pd
from collections import deque
from parpadeos import detect_blinks
from datetime import datetime, timedelta
from prophet import Prophet

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Definir la ruta del archivo CSV
file_path = "/home/juan/Escritorio/TESIS/scripts_Tesis/detected_prophet.csv"
prediction_file = "/home/juan/Escritorio/TESIS/scripts_Tesis/predicciones.csv"
 

def hacer_prediccion():
    """Función que ejecuta la predicción."""

    
        # Cargar datos
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Archivo no encontrado. Esperando más datos...")
        

        # Verificar cantidad de datos
    if len(df) > 5:
        print("Iniciando predicción...")

          

        
        df['ds'] = pd.to_datetime(df['Timestamp'])  
        df['y'] = df['fatigue_percentage']
        # Resamplear los datos a una frecuencia de 1 minuto

        df['ds'] = pd.to_datetime(df['ds'])  # Convertir a formato datetime

        # Agrupar cada 1 minuto y calcular la media
        df_resampled = df.resample('1T', on='ds').agg({'y': 'mean'}).reset_index()
 
        # Preprocesamiento
        df_resampled['cap'] = 100  
        df_resampled['floor'] = 0  
     
        # Configurar el modelo Prophet
        model = Prophet(changepoint_prior_scale=0.1,growth='logistic')  
        model.fit(df_resampled[['ds', 'y', 'cap', 'floor']])

        # Hacer predicción
        future = model.make_future_dataframe(periods=5, freq='1min')
        future['cap'] = 100  
        future['floor'] = 0  
        forecast = model.predict(future)

        # Guardar predicciones en CSV
        predicciones = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        predicciones.to_csv(prediction_file, mode='w', index=False)

        print("Predicción calculada y guardada en CSV.")

        # Guardar gráfica
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 5))
        model.plot(forecast, ax=ax)
        plt.legend()
        plt.savefig("prediccion.png")
    else:
        print("No hay datos para entrenar")



def detect_yawns(mar, MAR_THRESH, yawn_start_time, counter):
    if mar > MAR_THRESH:
        if yawn_start_time is None:  # Inicio del bostezo
            yawn_start_time = time.time()
        elif time.time() - yawn_start_time > 3:  # Más de 3 segundos
            counter += 1
            yawn_start_time = None  # Reiniciar tiempo de bostezo
    else:
        yawn_start_time = None  # Reiniciar si no está por encima del umbral

    return yawn_start_time, counter



# Cola para los últimos 5 valores de EAR
ear_history = deque(maxlen=5)  # Máximo 6 elementos

def update_ear_thresh(ear_history):
    """
    Calcula un nuevo EAR_THRESH basado en el promedio de los últimos valores de EAR.
    """
    if len(ear_history) > 0:
        return np.mean(ear_history)
    else:
        return 0.22  # Umbral predeterminado si no hay datos aún
 
 
n = 1
blink_counter = 0
blink_durations = []
ear_history = deque(maxlen=5)  # Últimos valores de EAR
blink_start_time = None  # Tiempo de inicio del parpadeo
aux_counter=0
amp=0 
#funcion para calcular parpadeos
def detect_blinks(ear,blink_durations):
    global n, blink_counter, blink_start_time,aux_counter
    MAX_EAR_LIMIT = 0.37


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
                print(f"Parpadeo {blink_counter}: {blink_duration:.3f} segundos")

            blink_start_time = None

    return blink_counter, blink_durations




def save_data_to_csv(output_file,start_time,blink_counter, yawn_counter,avg_blink_duration,fatigue_state, fatigue_percentage):
    """
    Guarda los datos en el archivo CSV cada 15 segundos.
    
    """
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            blink_counter,
            avg_blink_duration,
            yawn_counter,
            fatigue_state,
            fatigue_percentage
        ])

    # Resetear datos acumulativos después de guardar
    blink_counter = 0
    blink_durations=0
    yawn_counter = 0
    fatigue_percentage = 0


def initialize_csv_file(output_file):
    """
    Inicializa el archivo CSV con la cabecera.
    :param output_file: Nombre del archivo CSV.
    """
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Num_Blinks", "Avg_Blink_Duration", "Num_Yawns", "Fatigue_State","fatigue_percentage"])


def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        raise RuntimeError("Failed to open camera.")
    
    return cap

def eye_aspect_ratio(coordinates):
    """ Calcula la relación de aspecto del ojo (EAR) """
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)
    
def mouth_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))  # Parte superior-inferior
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))  # Parte central-superior
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))  # Izquierda-derecha
    return (d_A + d_B) / (2 * d_C)

def plotting_ear(timestamps, pts_ear, amps, line1, line2, thresh_line=None):
    """ Genera la gráfica en tiempo real del EAR y la amplitud histórica """
    global figure  # Mantiene la misma figura en todas las iteraciones

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

def calculate_fatigue_score(blink_count, avg_blink_duration, yawn_count):
    """Calcula la puntuación de fatiga y el estado del conductor."""
    
    # Puntuación de Frecuencia de Parpadeo (F_p)
    F_p=0
    D_p=0
    F_b=0
    
    valores_fatiga = {
        3: 100, 4: 95, 5: 90, 6: 85, 7: 80, 8: 75, 9: 50, 10: 45, 11: 40, 12: 35, 
        13: 30, 14: 25, 15: 10, 16: 10, 17: 15, 18: 20, 19: 24, 20: 50, 21: 55, 
        22: 60, 23: 65, 24: 70
    }
    
    if blink_count < 3:
        F_p = 100
    else:
        F_p = valores_fatiga.get(blink_count, 75)   
        

    # Puntuación de Duración de Parpadeo (D_p)
    if 0.100 < avg_blink_duration <= 0.200:  # ≤ 275 ms
        D_p = 230* avg_blink_duration  -22
    elif 0.200 < avg_blink_duration <= 0.275:  # > 275 ms y ≤ 300 ms
        D_p = 333.33 * avg_blink_duration - 42.666
    elif 0.275 < avg_blink_duration <= 0.350:  # > 300 ms y ≤ 400 ms
        D_p = 320 * avg_blink_duration -38
    elif 0.350 < avg_blink_duration <= 0.500:
        D_p = 166*avg_blink_duration +16.666
    else:  # = 0ms
        D_p = 40

    # Puntuación de Frecuencia de Bostezos (F_b)
    if yawn_count >= 1:
        F_b = 100
    else:
        F_b = 10
    # Cálculo de la puntuación total de fatiga (PT_F)
    PT_F = (F_p * 0.5) + (D_p * 0.25) + (F_b * 0.25)

    # Determinar estado de fatiga
    if PT_F >= 75:
        fatigue_state = "Fatiga Alta"# Es necesario descanso inmediato o cambio de conductor."
    elif 50 <= PT_F < 75:
        fatigue_state = "Fatiga Moderada" # Se recomienda tomar un breve descanso o realizar actividades que aumenten el estado de alerta."
    elif 25 <= PT_F < 50:
        fatigue_state = "Fatiga Baja" # Se recomienda mantener la atención y evitar distracciones. Si los síntomas de fatiga aumentan, considere tomar un descanso breve "
    else:
        fatigue_state = "Fatiga Nula" #Continúe con la conducción segura. Mantenga hábitos saludables como buena postura, hidratación y descansos regulares para prevenir la aparición de fatiga."

    # Cálculo del porcentaje de fatiga
    fatigue_percentage = PT_F
    return fatigue_state, fatigue_percentage

# Definir el nombre del archivo CSV
output_file = "detected_prophet.csv"
# Inicialización del archivo CSV
initialize_csv_file(output_file)

# Índices de los ojos y boca en MediaPipe
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
index_mouth = [61, 37, 267, 291, 314, 84]
# Inicializar la cámara
cap = init_camera()
pts_ear = deque(maxlen=64)
amps = deque(maxlen=64)  # Para almacenar valores históricos de amp
timestamps = deque(maxlen=64)  # Para almacenar los tiempos transcurridos
start_time = time.time()

i = 0
line1 = []
line2 = []
thresh_line = None
figure = None

ear_max = 0
ear_min = 0
amp = 0

yawn_start_time = None
yawn_counter = 0
pts_ear2 = deque(maxlen=5)

start_time = datetime.now()
blink_dura=0
t=0

avg_blink_duration = 0
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
                mounth_pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in index_mouth]

                # Dibujar los puntos de los ojos 
                #for pt in left_eye_pts + right_eye_pts:
                    #cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

                # Calcular EAR
                ear_left = eye_aspect_ratio(left_eye_pts)
                ear_right = eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2

                # Calcular MAR
                mar = mouth_aspect_ratio(mounth_pts)
                
                # Detectar parpadeos
                blink_counter, blink_durations = detect_blinks(ear,blink_durations)
                #Detectar bozteso
                yawn_start_time, yawn_counter = detect_yawns(mar, 0.6, yawn_start_time, yawn_counter)
                
                
                

                current_time = datetime.now()
                if current_time - start_time >= timedelta(minutes=1):
                    avg_blink_duration = sum(blink_durations) / len(blink_durations)
                    fatigue_state, fatigue_percentage = calculate_fatigue_score(blink_counter, avg_blink_duration, yawn_counter)
                    save_data_to_csv(output_file,start_time,blink_counter, yawn_counter,avg_blink_duration,fatigue_state, fatigue_percentage)
                    
                    if t>4:
                        hacer_prediccion()
                        t=0
                    t=t+1
                    
                    start_time = current_time
                    yawn_counter=0
                    blink_durations.clear()
                    blink_counter= 0

                # Mostrar los valores en la ventana de la cámara
                cv2.putText(frame, f"Parpadeos: {blink_counter}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Bostezos: {yawn_counter}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Duracion Prom. Parpadeo: {avg_blink_duration:}s", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
