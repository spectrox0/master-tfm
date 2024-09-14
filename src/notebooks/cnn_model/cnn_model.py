import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib

from src.utils.load_dataframe import load_time_series_60min


def main():
    print(device_lib.list_local_devices())
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        # Activate GPU by default
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

    # %%
    # 1. Cargar los datos
    df = load_time_series_60min()

    # %%
    # Seleccionar la columna de demanda de energía para Alemania
    target_column = 'DE_load_actual_entsoe_transparency'
    y = df[target_column].values


    # %%
    # Predecir usando solo la columna objetivo
    X = y

    # %%
    # Asegurarse de que no haya valores nulos
    X = X[~np.isnan(X)]
    y = y[~np.isnan(y)]

    # %%
    # 2. Escalar los datos con MinMaxScaler
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X.reshape(-1, 1))

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # %%
    # Función para crear secuencias de tiempo
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # %%
    # Definir el tamaño de las secuencias de tiempo
    seq_length = 60

    # %%

    # Crear secuencias a partir de X y y escalados
    X_seq, y_seq = create_sequences(X_scaled, seq_length)
    y_seq = y_seq.reshape(-1, 1)

    # %%
    # Redimensionar X para que sea compatible con la entrada de la CNN
    X_seq = np.reshape(X_seq, (X_seq.shape[0], X_seq.shape[1], 1))

    # %%
    # 3. Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # %%
    # 4. Crear el modelo CNN con las mejoras
    model = Sequential()

    # Primera capa de convolución y max pooling
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))

    # Segunda capa de convolución y max pooling
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Añadir capa de aplanado y densas
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    # %%
    # 5. Compilación del modelo usando Adam optimizer
    optimizer = Adam(learning_rate=0.0005)  # Tasa de aprendizaje más baja
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # %%
    # Añadir early stopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # %%
    # 6. Entrenar el modelo
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Incrementar las épocas para mejorar la precisión
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    # %%
    # 7. Graficar la evolución de la pérdida durante el entrenamiento
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Evolución de la pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

    # %%
    # 8. Hacer predicciones y revertir el escalado
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test)

    # 9. Evaluar el rendimiento del modelo
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')

    # %%
    # 10. Graficar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color='blue', label='Demanda Real')
    plt.plot(y_pred, color='red', label='Demanda Predicha')
    plt.title('Predicción de Demanda de Energía')
    plt.xlabel('Tiempo')
    plt.ylabel('Demanda de Energía (MW)')
    plt.legend()
    plt.show()

    # %%
    # 12. Mejorar predicciones mensuales para 2020
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df_2020 = df[df['utc_timestamp'].dt.year == 2020]
    y_2020 = df_2020[target_column].values
    y_2020_scaled = scaler_y.transform(y_2020.reshape(-1, 1))

    # Crear secuencias y hacer predicciones para 2020
    X_2020_seq, _ = create_sequences(y_2020_scaled, seq_length)
    X_2020_seq = np.reshape(X_2020_seq, (X_2020_seq.shape[0], X_2020_seq.shape[1], 1))
    predicciones_2020_scaled = model.predict(X_2020_seq)
    predicciones_2020 = scaler_y.inverse_transform(predicciones_2020_scaled)

    # Crear un DataFrame con predicciones y fechas
    predicciones_2020_df = pd.DataFrame({
        'utc_timestamp': df_2020['utc_timestamp'][seq_length:],
        'predicciones': predicciones_2020.flatten()
    })

    # Agrupar por mes para predicciones mensuales
    predicciones_mensuales_2020 = predicciones_2020_df.resample('M', on='utc_timestamp').sum()

    # %%
    # Graficar predicciones mensuales
    plt.figure(figsize=(12, 6))
    plt.plot(predicciones_mensuales_2020.index, predicciones_mensuales_2020['predicciones'], color='blue', marker='o', label='Predicciones Mensuales de Demanda')
    plt.title('Predicciones Mensuales de Demanda de Energía para 2020')
    plt.xlabel('Tiempo')
    plt.ylabel('Demanda de Energía (MW)')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()