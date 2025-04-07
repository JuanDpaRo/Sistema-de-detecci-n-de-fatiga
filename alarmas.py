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
