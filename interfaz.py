import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from estado import estado_ejecucion, lock  
import wiringpi
import time

# Configurar WiringPi en modo GPIO
wiringpi.wiringPiSetup()

# Definir los pines de los LEDS
LED_PINS = [3, 4, 6, 9, 10]

# Configurar los pines como salida
for pin in LED_PINS:
    wiringpi.pinMode(pin, 1)  # 1 = OUTPUT

def alarmas():
    valor = estado_ejecucion["indicador"]
    print(valor)
    if valor >= 75:
        wiringpi.digitalWrite(3, 0)
        wiringpi.digitalWrite(4, 1)
        wiringpi.digitalWrite(6, 1)
        wiringpi.digitalWrite(9, 0)
        wiringpi.digitalWrite(10, 0)
        time.sleep(0.1)
        wiringpi.digitalWrite(6, 0)
        time.sleep(0.7)
        
    elif 50 <= valor < 75:
        wiringpi.digitalWrite(3, 1)
        wiringpi.digitalWrite(4, 0)
        wiringpi.digitalWrite(6, 0)
        wiringpi.digitalWrite(9, 0)
        wiringpi.digitalWrite(10, 0)
    elif 25 <= valor < 50:
        wiringpi.digitalWrite(3, 0)
        wiringpi.digitalWrite(4, 0)
        wiringpi.digitalWrite(6, 0)
        wiringpi.digitalWrite(9, 1)
        wiringpi.digitalWrite(10, 0)
    else:
        wiringpi.digitalWrite(3, 0)
        wiringpi.digitalWrite(4, 0)
        wiringpi.digitalWrite(6, 0)
        wiringpi.digitalWrite(9, 0)
        wiringpi.digitalWrite(10, 1)

def actualizar_color_progreso(progress_bar, valor, style):
    if valor <= 25:
        color = "#2ecc71"
    elif 26 <= valor <= 50:
        color = "#f1c40f"
    elif 51 <= valor <= 75:
        color = "#e67e22"
    else:
        color = "#e74c3c"

    progress_style = f"custom_{id(progress_bar)}.Horizontal.TProgressbar"
    style.configure(progress_style, troughcolor="#FFFFFF", background=color)
    progress_bar.config(style=progress_style)
    progress_bar["value"] = valor

def cambiar_estado(valor):
    with lock:
        estado_ejecucion["valor"] = valor

def actualizar_valor(label, progress_bar, canvas, semaforo_luces,style):
    with lock:
        valor = estado_ejecucion["indicador"]
    label.config(text=f"Nivel de fatiga actual: {valor}%")
    actualizar_color_progreso(progress_bar, valor,style)  
    actualizar_semaforo(canvas, semaforo_luces, valor)
    alarmas() 
    label.after(1000, lambda: actualizar_valor(label, progress_bar, canvas, semaforo_luces,style))


def actualizar_valor3(label, progress_bar, canvas, semaforo_luces,style):
    with lock:
        valor = estado_ejecucion["estado_fatiga_futuro"]
    label.config(text=f"Predicción de fatiga en 5 minutos: {valor}%")
    progress_bar["value"] = valor
    actualizar_color_progreso(progress_bar, valor,style)
    actualizar_semaforo(canvas, semaforo_luces, valor)
    label.after(1000, lambda: actualizar_valor3(label, progress_bar, canvas, semaforo_luces,style))

def cargar_imagen(label, ruta):
    imagen = Image.open(ruta)
    imagen = imagen.resize((130, 46), Image.Resampling.LANCZOS)
    imagen_tk = ImageTk.PhotoImage(imagen)
    label.config(image=imagen_tk)
    label.image = imagen_tk

def actualizar_semaforo(canvas, luces, valor):
    colores = ["#bdc3c7"] * 4
    if valor <= 25:
        colores[0] = "#2ecc71"
    elif 26 <= valor <= 50:
        colores[1] = "#f1c40f"
    elif 51 <= valor <= 75:
        colores[2] = "#e67e22"
    elif 75 < valor <= 100:
        colores[3] = "#e74c3c"

    for i, luz in enumerate(luces):
        canvas.itemconfig(luz, fill=colores[i])

def crear_interfaz():
    root = tk.Tk()
    root.title("Monitor de Fatiga")
    root.geometry("800x535")
    root.configure(bg="#DCDCDC")
    # Inicializa el estilo después de crear root
    style = ttk.Style(root)

    frame_principal = tk.Frame(root, bg="#DCDCDC", padx=10, pady=10)
    frame_principal.pack(fill="both", expand=True)

    frame_imagen = tk.Frame(frame_principal, bg="#DCDCDC", height=100)
    frame_imagen.pack(side="bottom", fill="x")

    label_imagen = tk.Label(frame_imagen, bg="#DCDCDC")
    label_imagen.pack(expand=True)
    cargar_imagen(label_imagen, "/home/orangepi/Escritorio/CODIGO_FINAL/01.png")

    frame_controles = tk.Frame(frame_principal, bg="#DCDCDC", padx=10, pady=10)
    frame_controles.pack(side="left", fill="y")

    frame_indicadores = tk.Frame(frame_principal, bg="#DCDCDC", padx=20, pady=20)
    frame_indicadores.pack(side="left", expand=True, fill="both")

    def iniciar():
        cambiar_estado(1)
        print("Iniciado")

    def detener():
        cambiar_estado(0)
        print("Detenido")

    def pausar():
        cambiar_estado(2)
        print("Pausa")

    btn_style = {"font": ("Helvetica", 16, "bold"), "fg": "white", "width": 15, "height": 3}

    btn_iniciar = tk.Button(frame_controles, text="Iniciar", command=iniciar, bg="#008B8B", **btn_style)
    btn_iniciar.grid(row=0, column=0, sticky="nsew", pady=(75, 15))

    btn_detener = tk.Button(frame_controles, text="Detener", command=detener, bg="#FFA500", **btn_style)
    btn_detener.grid(row=1, column=0, sticky="nsew", pady=(15, 15))

    btn_pausar = tk.Button(frame_controles, text="Pausar", command=pausar, bg="#4682B4", **btn_style)
    btn_pausar.grid(row=2, column=0, sticky="nsew", pady=(15, 20))

    label_style = {"font": ("Helvetica", 16, "normal"), "bg": "#FFFFFF", "fg": "black", "pady": 0}


    label_01 = tk.Label(root, text="Fatiga Actual", font=("Helvetica", 13), bg="#DCDCDC", fg="black")
    label_01.place(relx=1.0, rely=1.0, anchor="se", x=-430, y=-500)

    label_02 = tk.Label(root, text="Fatiga Futura", font=("Helvetica", 13), bg="#DCDCDC", fg="black")
    label_02.place(relx=1.0, rely=1.0, anchor="se", x=-430, y=-280)

    
    frame_actual = tk.LabelFrame(frame_indicadores, bg="#FFFFFF", fg="black", padx=10, pady=10, font=("Helvetica", 12), height=150)
    frame_actual.pack(fill="x", pady=(5, 12))


    label_valor = tk.Label(frame_actual, text="Nivel de fatiga actual: 0%", height=0, **label_style)
    label_valor.pack(fill="both", expand=True, pady=0)

     
    progress_bar = ttk.Progressbar(frame_actual, length=400, mode="determinate", maximum=100,style="custom.Horizontal.TProgressbar")
    progress_bar.pack(fill="both", expand=True, pady=5)


    label_texto = tk.Label(frame_actual, text="Estado de fatiga:", **label_style, height = 1)
    label_texto.pack(fill="both", expand=True, pady=5)

    canvas = tk.Canvas(frame_actual, width=300, height=80, bg="#FFFFFF", highlightthickness=0)
    canvas.pack(pady=1)

    semaforo_luces = []
    etiquetas = ["Nula", "Baja", "Moderada", "Alta"]
    for i in range(4):
        x0 = 10 + i * 80
        y0 = 10
        x1 = x0 + 40
        y1 = y0 + 40
        luz = canvas.create_oval(x0, y0, x1, y1, fill="#bdc3c7", outline="black", width=2)
        semaforo_luces.append(luz)
        canvas.create_text((x0 + x1) / 2, y1 + 15, text=etiquetas[i], fill="black", font=("Helvetica", 11))

    frame_futuro = tk.LabelFrame(frame_indicadores, bg="#FFFFFF", fg="black", padx=10, pady=10, font=("Helvetica", 12),height=150)
    frame_futuro.pack(fill="x", pady=(15, 10))

    label_valor_futuro = tk.Label(frame_futuro, text="Estimacion de fatiga en 5 minutos: 0%", **label_style)
    label_valor_futuro.pack(fill="both", expand=True, pady=5)


    # Crear la barra de progreso con el nuevo estilo
    progress_bar_futuro = ttk.Progressbar(frame_futuro, length=400, mode="determinate", maximum=100, style="custom.Horizontal.TProgressbar")
    progress_bar_futuro.pack(fill="both", expand=True, pady=0)
    valor = 50
    actualizar_color_progreso(progress_bar, valor, style)
    
    label_texto_futuro = tk.Label(frame_futuro, text="Estado de fatiga en 5 minutos:", **label_style, height = 1)
    label_texto_futuro.pack(fill="both", expand=True, pady=5)
    
    canvas_futuro = tk.Canvas(frame_futuro, width=300, height=80, bg="#FFFFFF", highlightthickness=0)
    canvas_futuro.pack(pady=5)

    semaforo_luces_futuro = []
    for i in range(4):
        x0 = 10 + i * 80
        y0 = 10
        x1 = x0 + 40
        y1 = y0 + 40
        luz = canvas_futuro.create_oval(x0, y0, x1, y1, fill="#bdc3c7", outline="black", width=2)
        semaforo_luces_futuro.append(luz)
        canvas_futuro.create_text((x0 + x1) / 2, y1 + 15, text=etiquetas[i], fill="black", font=("Helvetica", 11))
    
    actualizar_valor(label_valor, progress_bar, canvas, semaforo_luces,style)
    actualizar_valor3(label_valor_futuro, progress_bar_futuro, canvas_futuro, semaforo_luces_futuro, style)
    alarmas()
    root.mainloop()
    