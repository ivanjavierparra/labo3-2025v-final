import pandas as pd
import gc
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
from tensorflow.keras.optimizers import Adam, RMSprop


ESTOY_EN_KAGGLE = True

config = {
        # Arquitectura
        "lstm_units_1": 64,
        "lstm_units_2": 32,
        "dense_units": 16,
        "dropout_rate": 0.3,

        # Entrenamiento
        "epochs": 150,
        "batch_size": 250,
        "early_stopping_patience": 20,
        "learning_rate": None,  # Si querés usar Adam con tasa específica

        # Regularización
        "l2_lambda": 0.001,

        # Optimizador
        "optimizer": "rmsprop",  # adam, sgd, rmsprop, etc.

        # Preprocesamiento
        "scaler_name": "robust",  # standard o robust

        # Ventana temporal
        "window_size": 3,
        "feature_cols" : ['tn', 'tn_lag1', 
                          #'tn_lag6', 
                          #'tn_lag12', 
                          'tn_diff1', 
                          #'tn_diff6',
                'rolling_mean3', 'rolling_std3', 'rolling_max3', 'rolling_min3','rolling_max6','rolling_min6',"size"],
    }


if ESTOY_EN_KAGGLE:
    df = pd.read_csv("../entregable/datasets/periodo_x_producto_con_target_transformado_con_feature_engineering_201912.csv", sep=',', encoding='utf-8')
else:
    # Cargar de nuevo por claridad
    df = pd.read_csv("../../data/raw/sell-in.csv", sep="\t")

df["periodo"] = pd.to_datetime(df["periodo"], format="%Y%m")

# Agregación mensual por producto
df = df.groupby(["product_id", "periodo"]).agg({"tn": "sum"}).sort_values(["product_id", "periodo"]).reset_index()

# Crear características por producto
def agregar_features(df):
    df = df.copy()
    df = df.sort_values(["product_id", "periodo"])
    
    # Crear features con groupby + transform
    df["tn_lag1"] = df.groupby("product_id")["tn"].shift(1)
    #df["tn_lag6"] = df.groupby("product_id")["tn"].shift(6)
    #df["tn_lag12"] = df.groupby("product_id")["tn"].shift(12)

    df["tn_diff1"] = df["tn"] - df["tn_lag1"]
    #df["tn_diff6"] = df["tn"] - df["tn_lag6"]
    df["size"] = df.groupby("product_id")["tn"].transform("size")
    

    df["rolling_mean3"] = df.groupby("product_id")["tn"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["rolling_std3"] = df.groupby("product_id")["tn"].transform(lambda x: x.shift(1).rolling(3).std())
    
    df["rolling_max3"] = df.groupby("product_id")["tn"].transform(lambda x: x.shift(1).rolling(3).max())
    df["rolling_min3"] = df.groupby("product_id")["tn"].transform(lambda x: x.shift(1).rolling(3).min())
    df["rolling_max6"] = df.groupby("product_id")["tn"].transform(lambda x: x.shift(1).rolling(6).max())
    df["rolling_min6"] = df.groupby("product_id")["tn"].transform(lambda x: x.shift(1).rolling(6).min())

    return df

df_features = agregar_features(df).fillna(0)





# Último período disponible
ultimo_mes = df_features["periodo"].max()

# Definir los 3 meses anteriores
ultimos_3_meses = pd.date_range(end=ultimo_mes - pd.DateOffset(months=1), periods=3, freq='MS')

# Filtrar productos con datos en al menos 3 de esos meses
df_filtrado = df_features[df_features["periodo"].isin(ultimos_3_meses)]

# Contar cuántos meses tiene cada producto
conteo_por_producto = df_filtrado[df_filtrado["tn"] > 0].groupby("product_id").size()

# Seleccionar productos válidos
productos_validos = conteo_por_producto[conteo_por_producto >= 3].index

# Filtrar el dataframe original
df_features = df_features[df_features["product_id"].isin(productos_validos)].copy()
print(df_features.shape)




if ESTOY_EN_KAGGLE:
    df_test = df_features[df_features["periodo"] == pd.to_datetime(201912, format="%Y%m")].copy()
    df_features[df_features["periodo"].isin(pd.to_datetime([201911, 201912], format="%Y%m")) ]
else:
    df_test = df_features[df_features["periodo"] == pd.to_datetime(201910, format="%Y%m")].copy()
    df_features[df_features["periodo"].isin(pd.to_datetime([201910, 201911, 201912], format="%Y%m")) ]
    
    
    


window_size = config["window_size"]
scaler_name = config["scaler_name"]

feature_cols = config["feature_cols"]

# Agrupar por producto
productos = df_features["product_id"].unique()
scalers = {}  # Guardamos los scalers por producto

X, y, productos_list = [], [], []

for producto in productos:
    df_prod = df_features[df_features["product_id"] == producto].copy()

    if len(df_prod) < window_size + 2:
        continue  # No tiene suficientes datos

    # Escalado por producto
    
    scaler = StandardScaler() if scaler_name == "standard" else RobustScaler()
    scaled_features = scaler.fit_transform(df_prod[feature_cols])
    scalers[producto] = scaler

    for i in range(window_size, len(df_prod) - 2):
        X.append(scaled_features[i - window_size:i])  # (window_size, n_features)
        y.append(scaled_features[i + 2][0])  # Target: tn escalado en t+2
        productos_list.append(producto)

X = np.array(X)
y = np.array(y).reshape(-1, 1)





periodos = []

for producto in productos:
    df_prod = df_features[df_features["product_id"] == producto].copy()

    if len(df_prod) < window_size + 2:
        continue

    for i in range(window_size, len(df_prod) - 2):
        # El periodo objetivo (de y) es en t+2, entonces corresponde a:
        periodo_target = df_prod.iloc[i + 2]["periodo"]
        periodos.append(periodo_target)

# Convertimos a numpy para facilitar indexado
periodos = np.array(periodos)

# Definimos el umbral de corte (por ejemplo, predecimos diciembre 2019)
fecha_corte = pd.to_datetime("2019-12-01")

# Creamos máscaras según el periodo de y
train_mask = periodos < fecha_corte
test_mask = periodos == fecha_corte  # opcional: podrías usar > fecha_corte para más test

# Aplicamos la máscara
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
productos_test = np.array(productos_list)[test_mask]








# Configuración y semillas
semillas = [42, 101, 202, 303, 404]  # Tus semillas
l2_lambda = config["l2_lambda"]
optimizer_name = config["optimizer"]
epochs = config["epochs"]
batch_size = config["batch_size"]
early_stopping_patience = config["early_stopping_patience"]

# Función para crear modelo (con semilla como parámetro)
def crear_modelo(semilla, window_size, n_features):
    tf.keras.utils.set_random_seed(semilla)  # Fija semilla para TensorFlow
    
    model = Sequential([
        LSTM(200, activation='tanh', return_sequences=True, 
             input_shape=(window_size, n_features),
             kernel_regularizer=l2(l2_lambda) if l2_lambda > 0 else None),
        Dropout(0.2, seed=semilla),
        LSTM(32, activation='tanh'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Configurar optimizer con semilla
    if optimizer_name.lower() == 'adam':
        optimizer = Adam(learning_rate=0.001)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = RMSprop(learning_rate=0.001)
    else:
        optimizer = optimizer_name
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Entrenamiento con semillerio
def entrenar_con_semillerio(X_train, y_train, X_test, y_test):
    modelos = []
    historiales = []
    window_size = X_train.shape[1]
    n_features = X_train.shape[2]
    
    callbacks = [
        EarlyStopping(patience=early_stopping_patience, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=15)
    ]
    
    for i, semilla in enumerate(semillas):
        print(f"\nEntrenando modelo con semilla {semilla} ({i+1}/{len(semillas)})")
        
        model = crear_modelo(semilla, window_size, n_features)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        modelos.append(model)
        historiales.append(history)
    
    return modelos, historiales



modelos, historiales = entrenar_con_semillerio(X_train, y_train, X_test, y_test)




def predecir_todos_t2_semillerio(df_features_full, modelos, scalers, feature_cols, window_size):
    """
    Predice tn en t+2 para todos los productos usando un ensemble de modelos con semillerio.

    Args:
        df_features_full (pd.DataFrame): dataframe completo con features
        modelos (list): lista de modelos LSTM entrenados con diferentes semillas
        scalers (dict): diccionario con los StandardScaler por producto
        feature_cols (list): lista de columnas de features
        window_size (int): tamaño de la ventana temporal

    Returns:
        pd.DataFrame: dataframe con columnas ["product_id", "tn_t2_pred"]
    """
    productos = df_features_full["product_id"].unique()
    resultados = []

    for pid in productos:
        df_prod = df_features_full[df_features_full["product_id"] == pid]
        
        if len(df_prod) < window_size:
            continue  # no hay suficientes datos

        try:
            # Preparar datos
            ultimos = df_prod[feature_cols].iloc[-window_size:]
            scaler = scalers[pid]
            ultimos_scaled = scaler.transform(ultimos)
            X_new = ultimos_scaled.reshape(1, window_size, len(feature_cols))
            
            # Predecir con todos los modelos y promediar
            predicciones = []
            for model in modelos:
                y_pred_scaled = model.predict(X_new, verbose=0)
                tn_mean = scaler.center_[0]
                tn_std = scaler.scale_[0]
                y_pred = y_pred_scaled[0][0] * tn_std + tn_mean
                predicciones.append(y_pred)
            
            y_pred_final = np.mean(predicciones)
            
            # Opcional: eliminar outliers extremos antes de promediar
            # y_pred_final = np.mean(np.clip(predicciones, 
            #                           np.percentile(predicciones, 10),
            #                           np.percentile(predicciones, 90)))

        except Exception as e:
            print(f"Error al predecir producto {pid}: {e}")
            y_pred_final = np.mean(df_prod["tn"])  # Valor por defecto si falla
            
        # Asegurar predicción positiva
        y_pred_final = max(0, y_pred_final)
        resultados.append({"product_id": pid, "tn_t2_pred": y_pred_final})

    return pd.DataFrame(resultados)


df_preds_t2 = predecir_todos_t2_semillerio(
    df_features,
    modelos,
    scalers,
    feature_cols,
    window_size
)

df_final = df_test.merge(df_preds_t2, on="product_id", how="left")


numerador = (df_final["tn"]- df_final["tn_t2_pred"]).abs().sum()
denominador = df_final["tn"].sum()
porcentaje_error = (numerador / denominador)
porcentaje_error