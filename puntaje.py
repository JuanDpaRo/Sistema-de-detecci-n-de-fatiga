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
