import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simular el primer dataset (Calidad del Aire)
# Contiene mediciones de contaminantes y temperatura en Celsius
np.random.seed(42) # Para reproducibilidad

dates_air = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(168)]
data_air = {
    'Fecha': dates_air,
    'CO2_ppm': np.random.normal(420, 15, 168),
    'PM2.5_ugm3': np.random.normal(25, 8, 168),
    'temp_celsius': np.random.normal(15, 5, 168)
}
# Introducir valores faltantes
data_air['PM2.5_ugm3'][10:20] = np.nan
data_air['CO2_ppm'][50:60] = np.nan

df_calidad_aire = pd.DataFrame(data_air)
print("--- Primer DataFrame (df_calidad_aire) ---")
print(df_calidad_aire.head())

print("\n-------------------------------------------")

# Simular el segundo dataset (Datos Meteorológicos)
# Contiene humedad, velocidad del viento y temperatura en Fahrenheit
dates_meteo = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(168)]
data_meteo = {
    'date': dates_meteo,
    'Humidity_%': np.random.normal(60, 10, 168),
    'Wind_Speed_kph': np.random.normal(10, 3, 168),
    'temp_fahrenheit': np.random.normal(60, 10, 168)
}
# Introducir valores atípicos
data_meteo['Wind_Speed_kph'][75] = 100 # Outlier
data_meteo['Humidity_%'][100] = 5 # Outlier

df_meteo = pd.DataFrame(data_meteo)
print("--- Segundo DataFrame (df_meteo) ---")
print(df_meteo.head())

# Guardar datasets como CSV
df_calidad_aire.to_csv("calidad_aire.csv", index=False)
df_meteo.to_csv("datos_meteorologicos.csv", index=False)
