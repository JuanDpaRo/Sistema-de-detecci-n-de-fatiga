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

       
    else:
        print("No hay datos para entrenar")
