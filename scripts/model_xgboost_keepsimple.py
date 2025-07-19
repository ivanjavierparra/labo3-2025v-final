import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from xgboost.callback import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from optuna.storages import RDBStorage
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
import os
import json
import sqlite3






training = [
    201701, 201702, 201703, 201704, 201705, 201706, 201707, 201708, 201709,
    201710, 201711, 201712, 201801, 201802, 201803, 201804, 201805,
    201806, 201807, 201808, 201809, 201810, 201811, 201812,
    201901, 201902, 201903, 201904, 201905, 201906
]

validation = [
    201907, 201909
]


testing = [
    201910
]






def guardar_hiperparametros(best_params, name='lgb_v1'):
    """
    Guarda los mejores hiperparámetros en un archivo JSON.
    """
    # Guardar best_params en un archivo JSON
    with open(f'./best_params_{name}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
        
        

def levantar_hiperparametros(nombre):
    """
    Levanta los hiperparámetros guardados en un archivo JSON.
    
    Args:
        nombre (str): Nombre del archivo (sin extensión .json).
    
    Returns:
        dict: Diccionario con los hiperparámetros. None si hay error.
    """
    try:
        with open(f'./best_params_{nombre}.json', 'r') as f:
            best_params = json.load(f)  # ¡Usar json.load() en lugar de json.dump()!
        return best_params
    except FileNotFoundError:
        print(f"Error: Archivo './best_params/{nombre}.json' no encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo './best_params/{nombre}.json' no tiene formato JSON válido.")
        return None   




   
def optimizar_con_optuna_sin5FCV_con_semillerio_db(df_train, df_val, semillas=[42, 101, 202, 303, 404], version="v1", n_trials=100):
    """
    Optimiza hiperparámetros de XGBoost con Optuna usando semillerío.
    Guarda trials en SQLite y permite visualización en tiempo real con optuna-dashboard.
    """

    db_name = f"optuna_studies_{version}.db"
    storage_url = f"sqlite:///{db_name}"

    print("\nPara visualizar los resultados en tiempo real:")
    print("1. Abre otra terminal y ejecuta:")
    print(f"   optuna-dashboard sqlite:///{db_name}")
    print("2. Abre en tu navegador: http://127.0.0.1:8080/")

    # ==================== Preparación de datos ====================
    # --- FEATURES DIVISION
    target_col = 'target'
    feature_cols = [col for col in df_train.columns if col != target_col]
    datetime_cols = df_train.select_dtypes(include=['datetime', 'datetime64', 'object', 'category']).columns.tolist()
    
    # --- TRAIN y VAL
    X_tr = df_train[feature_cols]
    X_tr = X_tr.drop(columns=[*datetime_cols])
    y_tr = df_train[target_col]
    X_val = df_val[feature_cols]
    X_val = X_val.drop(columns=[*datetime_cols])
    y_val = df_val[target_col]
    
    # --- NaNs y Inf
    X_tr.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_tr.fillna(0, inplace=True)

    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.fillna(0, inplace=True)

    
    
    # --- PESOS
    # weights = np.log1p(y).replace([np.inf, -np.inf], 0).fillna(0).clip(lower=1e-3)

    

    # ==================== Métrica personalizada ====================
    def total_forecast_error(y_true, y_pred):
        numerador = np.sum(np.abs(y_true - y_pred))
        denominador = np.sum(y_true)
        if denominador == 0:
            return np.nan
        return numerador / denominador

    

    # ==================== Función objetivo para Optuna ====================
    def objective(trial):
        
        print(f"⏳ Ejecutando Trial #{trial.number}")
        
        params = {
            # 'objective': 'reg:squarederror',
            #'eval_metric': 'rmse',  # Se usa solo para early stopping
            # 'booster': 'gbtree',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            # 'tree_method': 'hist',
            'max_bin': trial.suggest_int('max_bin', 100, 512),
            # 'verbosity': 0,
            # 'early_stopping' : trial.suggest_int('early_stopping_rounds', 20, 100)
        }

    
        tfe_seeds = []

        for seed in semillas:
            model = xgb.XGBRegressor(
                **params,
                # n_estimators=1000,
                random_state=seed
            )

            model.fit(
                X_tr, y_tr,
                # sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                # sample_weight_eval_set=[w_val],
                # early_stopping_rounds=30,
                #verbose=False
            )

            y_pred = model.predict(X_val)
            tfe = total_forecast_error(y_val, y_pred)
            tfe_seeds.append(tfe)

        return np.mean(tfe_seeds)

    # ==================== Callback ====================
    def print_best_trial(study, trial):
        print(f"Mejor trial hasta ahora: TFE={study.best_value:.6f}, Parámetros={study.best_trial.params}")

    # ==================== Crear estudio Optuna ====================
    try:
        study = optuna.create_study(
            direction='minimize',
            study_name=f"xgboost_optimization_{version}",
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
    except sqlite3.OperationalError:
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        print("Advertencia: No se pudo conectar a SQLite. Usando almacenamiento en memoria")

    # ==================== Ejecutar optimización ====================
    study.optimize(objective, n_trials=n_trials, callbacks=[print_best_trial], timeout=3 * 3600 * 24)

    # ==================== Resultados ====================
    if isinstance(study._storage, optuna.storages.InMemoryStorage):
        print("No se guardaron los trials al no usar SQLite")
    else:
        print(f"Estudio guardado en: {storage_url}")
        fig_history = plot_optimization_history(study)
        fig_params = plot_param_importances(study)
        fig_history.write_image(f"optimization_history_{version}.png")
        fig_params.write_image(f"param_importances_{version}.png")

    print("\nMejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    # ==================== Guardar hiperparámetros ====================
    best_params = study.best_params.copy()
    # best_params.update({
    #     'objective': 'reg:squarederror',
    #     'eval_metric': 'rmse',
    #     'booster': 'gbtree',
    #     'tree_method': 'hist',
    #     'verbosity': 0
    # })

    guardar_hiperparametros(best_params, version)  # Asegurate de tener esta función definida

    return study, best_params













