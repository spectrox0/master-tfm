'''
1.0V

Script orientado al PRE-PROCESADO de datos. 

Acciones realizadas:
    1. Limpieza de Datos:
        - Identificación y eliminación de valores NaN
        - Interpolación lineal para valores faltantes

    2. Análisis de Tendencias y Patrones Estacionales
        - Resampleo diario y mensual

    3. Creación de Nuevas Características

    4. Correlación de Características

    5. Análisis de Outliers

    6. Generación de nuevo csv de datos preprocesados y normalizados.

Alberto Bria
'''



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DEFINIR NOMBRE DEL ARCHIVO CSV A GENERAR (CAMBIAR NOMBRE SI HYA MODIFICACIONES DE CÓDIGO)
n = '0'
 
# Cargar los datos proporcionados
data = pd.read_csv('./Datasets/time_series_60min_singleindex.csv', parse_dates=['utc_timestamp'], index_col='utc_timestamp')

# Filtrar las columnas relevantes
columns_of_interest = ['DE_solar_generation_actual', 'DE_wind_generation_actual', 'DE_load_actual_entsoe_transparency']
data = data[columns_of_interest]

# 1. Limpieza de Datos: Identificación y eliminación de valores NaN
nan_counts = data.isna().sum()

# Gráfico 1: Gráfica de valores NaN antes de la limpieza
plt.figure(figsize=(10, 6))
nan_counts.plot(kind='bar', color='skyblue')
plt.title('Valores NaN en cada columna antes de la limpieza')
plt.xlabel('Columnas')
plt.ylabel('Cantidad de valores NaN')
plt.show()

# Limpieza: Interpolación lineal para valores faltantes
data = data.interpolate(method='linear')

# 2. Análisis de Tendencias y Patrones Estacionales
# Gráfico 2: Tendencia de la demanda total de energía en Alemania
plt.figure(figsize=(14, 7))
data['DE_load_actual_entsoe_transparency'].plot(title='Tendencia de la Demanda Total de Energía en Alemania', color='blue')
plt.xlabel('Fecha')
plt.ylabel('Demanda Total de Energía (MW)')
plt.show()

# Resampleo diario y mensual
daily_data = data['DE_load_actual_entsoe_transparency'].resample('D').mean()
monthly_data = data['DE_load_actual_entsoe_transparency'].resample('M').mean()

# Gráfico 3: Estacionalidad diaria de la demanda de energía en Alemania
plt.figure(figsize=(14, 7))
daily_data.plot(title='Estacionalidad Diaria de la Demanda de Energía en Alemania', color='green')
plt.xlabel('Fecha')
plt.ylabel('Demanda Promedio Diaria (MW)')
plt.show()

# Gráfico 4: Estacionalidad mensual de la demanda de energía en Alemania
plt.figure(figsize=(14, 7))
monthly_data.plot(title='Estacionalidad Mensual de la Demanda de Energía en Alemania', color='purple')
plt.xlabel('Fecha')
plt.ylabel('Demanda Promedio Mensual (MW)')
plt.show()

# 3. Creación de Nuevas Características
data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek
data['quarter'] = data.index.quarter
data['month'] = data.index.month
data['year'] = data.index.year
data['dayofyear'] = data.index.dayofyear
data['weekend'] = data['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# Gráfico 5: Visualización de las nuevas características temporales creadas
plt.figure(figsize=(14, 7))
data[['hour', 'dayofweek', 'quarter', 'month', 'weekend']].sample(1000).plot(kind='box', figsize=(14, 7))
plt.title('Visualización de Nuevas Características Temporales')
plt.show()

# 4. Correlación de Características
correlation_matrix = data.corr()

# Gráfico 6: Matriz de Correlación entre las características
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación entre Características')
plt.show()

# 5. Análisis de Outliers
# Gráfico 7: Boxplot de la demanda total de energía en Alemania
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['DE_load_actual_entsoe_transparency'], color='orange')
plt.title('Boxplot de la Demanda Total de Energía en Alemania')
plt.xlabel('Demanda Total de Energía (MW)')
plt.show()

data.to_csv(f'./Datasets/Processed_Data/processed_data_{n}.csv')