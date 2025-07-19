import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from optuna.storages import RDBStorage
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
import os
import json
import sqlite3
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.pruners import MedianPruner

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
    with open(f'./best_params/{name}.json', 'w') as f:
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
        with open(f'./best_params/{nombre}.json', 'r') as f:
            best_params = json.load(f)  # ¡Usar json.load() en lugar de json.dump()!
        return best_params
    except FileNotFoundError:
        print(f"Error: Archivo './best_params/{nombre}.json' no encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo './best_params/{nombre}.json' no tiene formato JSON válido.")
        return None




def optimizar_con_optuna_sin5FCV_con_semillerio_db(train, semillas=[42, 101, 202, 303, 404], version="v1", n_trials=500, pesos="log"):
    """
    Optimiza hiperparámetros de LightGBM con Optuna usando semillerío.
    Guarda trials en SQLite y permite visualización en tiempo real.
    
    Args:
        train: DataFrame con datos de entrenamiento
        semillas: Lista de semillas para el semillerío
        version: Identificador del estudio
        n_trials: Número de trials de optimización
    """
    # Configuración de la base de datos SQLite
    db_name = f"optuna_studies_{version}.db"
    storage_url = f"sqlite:///{db_name}"
    
    # Instrucciones para usar el dashboard:
    print("\nPara visualizar los resultados en tiempo real:")
    print("1. Abre otra terminal y ejecuta:")
    print(f"   optuna-dashboard sqlite:///optuna_studies_{version}.db")
    print("2. Abre en tu navegador: http://127.0.0.1:8080/")
    
    # Preparación de datos
    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64', 'category']).columns.tolist()
    X = train.drop(columns=[*datetime_cols, 'target'])
    X = X.replace([np.inf, -np.inf], np.nan)
    y = train['target'].replace([np.inf, -np.inf], np.nan)
    y = y.fillna(0)  # Rellenar NaNs con ceros

    if( pesos == "log"):
        weights = np.log1p(y)
        weights = weights.replace([np.inf, -np.inf], 0)
        weights = weights.fillna(0)
        weights = weights.clip(lower=1e-3)
    else:
        weights = (y / (y.max() + 1e-10))  # Pequeño épsilon para evitar división por cero
        weights = weights.replace([np.inf, -np.inf], 0).fillna(0).clip(lower=1e-3)
    # try:
    #     weights = y / y.max()  # Fórmula base  ########### Probar tambien: np.log1p(y)
    #     weights = weights.replace([np.inf, -np.inf], 0)  # Manejo de infinitos
    #     weights = weights.fillna(0)  # Manejo de NaNs
    # except ZeroDivisionError:
    #     weights = np.ones(len(y))  # Fallback a pesos unitarios

    
    def objective(trial):
        base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 100),  # Reducido de 200 a 100
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),  # Aumentado mínimo a 0.6
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),  # Aumentado mínimo a 0.7
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # Aumentado mínimo a 10
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Reducido de 15 a 10
            'max_bin': trial.suggest_int('max_bin', 100, 500),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),  # Aumentado mínimo a 20
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'verbosity': -1,
            
            # Hiperparámetros adicionales recomendados:
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50),  # Nuevo
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 1.0),  # Nuevo (suaviza divisiones)
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.5),  # Nuevo
        }

        rmse_scores_fold = []

       
        X_train_fold = X[X['periodo'].isin(training)]
        X_val_fold = X[X['periodo'].isin(validation)]

        y_train_fold = y.loc[X_train_fold.index]
        y_val_fold = y.loc[X_val_fold.index]

        w_train_fold = weights.loc[X_train_fold.index]
        w_val_fold = weights.loc[X_val_fold.index]

        
        rmse_seeds = []

        for seed in semillas:
            params = base_params.copy()
            params['seed'] = seed

            # Crear datasets con pesos
            train_data = lgb.Dataset(
                X_train_fold, 
                label=y_train_fold,
                weight=w_train_fold,  # Aquí se incluyen los pesos de entrenamiento
                free_raw_data=False
            )
            val_data = lgb.Dataset(
                X_val_fold, 
                label=y_val_fold,
                weight=w_val_fold,    # Aquí se incluyen los pesos de validación
                reference=train_data
            )

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )

            y_pred = model.predict(X_val_fold)
            rmse = mean_squared_error(y_val_fold, y_pred, sample_weight=w_val_fold)  # RMSE ponderado
            rmse_seeds.append(rmse)

        return np.mean(rmse_seeds)


    def print_best_trial(study, trial):
        print(f"Mejor trial hasta ahora: RMSE={study.best_value:.6f}, Parámetros={study.best_trial.params}")

    # Crear estudio con almacenamiento en SQLite
    try:
        study = optuna.create_study(
            direction='minimize',
            study_name=f"lightgbm_optimization_{version}",
            storage=storage_url,
            load_if_exists=True,
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=42))
    except sqlite3.OperationalError:
        # Si falla la conexión, crear estudio en memoria
        study = optuna.create_study(
            direction='minimize',
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=42))
        print("Advertencia: No se pudo conectar a SQLite. Usando almacenamiento en memoria")

    # Ejecutar optimización
    study.optimize(objective, n_trials=n_trials, callbacks=[print_best_trial], timeout=3600*24*3) # Límite de tiempo de 3 días

    # Guardar resultados y visualizaciones
    if isinstance(study._storage, optuna.storages.InMemoryStorage):
        print("No se guardaron los trials al no usar SQLite")
    else:
        print(f"Estudio guardado en: {storage_url}")
        
        # Generar visualizaciones
        fig_history = plot_optimization_history(study)
        fig_params = plot_param_importances(study)
        fig_history.write_image(f"optimization_history_{version}.png")
        fig_params.write_image(f"param_importances_{version}.png")

    print("\nMejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1
    })

    guardar_hiperparametros(best_params, version)  # Descomenta si tienes esta función
    
    
 
    
    return study, best_params




def optimizar_con_optuna_con5FCV_con_semillerio_db(train, semillas=[42, 101, 202, 303, 404], version="v1", n_trials=500, pesos="log"):
    """
    Optimiza hiperparámetros de LightGBM con Optuna usando semillerío.
    Guarda trials en SQLite y permite visualización en tiempo real.
    
    Args:
        train: DataFrame con datos de entrenamiento
        semillas: Lista de semillas para el semillerío
        version: Identificador del estudio
        n_trials: Número de trials de optimización
    """
    # Configuración de la base de datos SQLite
    db_name = f"optuna_studies_{version}.db"
    storage_url = f"sqlite:///{db_name}"
    
    # Instrucciones para usar el dashboard:
    print("\nPara visualizar los resultados en tiempo real:")
    print("1. Abre otra terminal y ejecuta:")
    print(f"   optuna-dashboard sqlite:///optuna_studies_{version}.db")
    print("2. Abre en tu navegador: http://127.0.0.1:8080/")
    
    # Preparación de datos
    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64', 'category']).columns.tolist()
    X = train.drop(columns=[*datetime_cols, 'target'])
    y = train['target']

    if( pesos == "log"):
        weights = np.log1p(y)
        weights = weights.replace([np.inf, -np.inf], 0)
        weights = weights.fillna(0)
        weights = weights.clip(lower=1e-3)
    else:
        weights = (y / (y.max() + 1e-10))  # Pequeño épsilon para evitar división por cero
        weights = weights.replace([np.inf, -np.inf], 0).fillna(0).clip(lower=1e-3)
    
    
    # Reemplazá infinitos por NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    weights = weights.replace([np.inf, -np.inf], np.nan)

    # Rellená NaN con ceros o interpolación según el caso
    X = X.fillna(0)
    y = y.fillna(0)
    weights = weights.fillna(1)  # Si usás pesos personalizados
    

    

    def objective(trial):
        base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 100),  # Reducido de 200 a 100
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),  # Aumentado mínimo a 0.6
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),  # Aumentado mínimo a 0.7
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # Aumentado mínimo a 10
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Reducido de 15 a 10
            'max_bin': trial.suggest_int('max_bin', 100, 500),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),  # Aumentado mínimo a 20
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'verbosity': -1,
            
            # Hiperparámetros adicionales recomendados:
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50),  # Nuevo
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 1.0),  # Nuevo (suaviza divisiones)
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.5),  # Nuevo
        }

        rmse_scores_fold = []
        
        folds_train = [
            training[0:12],   # 201701–201712
            training[0:15],   # 201701–201803
            training[0:18],   # ...
            training[0:21],
            training[0:24],   # 201701–201906
        ]
        
        
        

        for fechas_fold in folds_train:
            X_train_fold = X[X['periodo'].isin(fechas_fold)]
            X_val_fold = X[X['periodo'].isin(validation)]

            y_train_fold = y.loc[X_train_fold.index]
            y_val_fold = y.loc[X_val_fold.index]

            w_train_fold = weights.loc[X_train_fold.index]
            w_val_fold = weights.loc[X_val_fold.index]
                
            rmse_seeds = []

            for seed in semillas:
                params = base_params.copy()
                params['seed'] = seed

                # Crear datasets con pesos
                train_data = lgb.Dataset(
                    X_train_fold, 
                    label=y_train_fold,
                    weight=w_train_fold,  # Aquí se incluyen los pesos de entrenamiento
                    free_raw_data=False
                )
                val_data = lgb.Dataset(
                    X_val_fold, 
                    label=y_val_fold,
                    weight=w_val_fold,    # Aquí se incluyen los pesos de validación
                    reference=train_data
                )

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )

                y_pred = model.predict(X_val_fold)
                rmse = mean_squared_error(y_val_fold, y_pred, sample_weight=w_val_fold)  # RMSE ponderado
                rmse_seeds.append(rmse)

            rmse_scores_fold.append(np.mean(rmse_seeds))

        return np.mean(rmse_scores_fold)

    def print_best_trial(study, trial):
        print(f"Mejor trial hasta ahora: RMSE={study.best_value:.6f}, Parámetros={study.best_trial.params}")

    # Crear estudio con almacenamiento en SQLite
    try:
        study = optuna.create_study(
            direction='minimize',
            study_name=f"lightgbm_optimization_{version}",
            storage=storage_url,
            load_if_exists=True,
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=42))
    except sqlite3.OperationalError:
        # Si falla la conexión, crear estudio en memoria
        study = optuna.create_study(
            direction='minimize',
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=42))
        print("Advertencia: No se pudo conectar a SQLite. Usando almacenamiento en memoria")

    # Ejecutar optimización
    study.optimize(objective, n_trials=n_trials, callbacks=[print_best_trial], timeout=3600*24*3) # Límite de tiempo de 3 días

    # Guardar resultados y visualizaciones
    if isinstance(study._storage, optuna.storages.InMemoryStorage):
        print("No se guardaron los trials al no usar SQLite")
    else:
        print(f"Estudio guardado en: {storage_url}")
        
        # Generar visualizaciones
        fig_history = plot_optimization_history(study)
        fig_params = plot_param_importances(study)
        fig_history.write_image(f"optimization_history_{version}.png")
        fig_params.write_image(f"param_importances_{version}.png")

    print("\nMejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1
    })

    guardar_hiperparametros(best_params, version)  # Descomenta si tienes esta función
    
    
 
    
    return study, best_params




def semillerio_en_prediccion_con_pesos(train, test, version="v1", pesos="log"):
    """
    Entrena un modelo LightGBM con múltiples semillas y promedia las predicciones.
    Versión que incluye pesos consistentes con el entrenamiento original.
    """
    # Preparación de datos
    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64', 'category']).columns.tolist()
    X_train = train.drop(columns=[*datetime_cols, 'target']).replace([np.inf, -np.inf], np.nan)
    y_train = train['target'].replace([np.inf, -np.inf], np.nan)
    y_train = y_train.fillna(0)
    X_test = test.drop(columns=[*datetime_cols, 'target']).replace([np.inf, -np.inf], np.nan)
    

    
    # Calcular pesos consistentes con el entrenamiento
    weights = np.log1p(y_train)
    weights = weights.replace([np.inf, -np.inf], 0)
    weights = weights.fillna(0)
    weights = weights.clip(lower=1e-3)
    
        
        
    # Crear Dataset con pesos
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        weight=weights,  # ¡Incluir los pesos aquí!
        free_raw_data=False
    )
    
    # Número de repeticiones con semillas distintas
    best_params = levantar_hiperparametros(version)
    

    seeds = [42, 101, 202, 303, 404]
    predictions = []
    feature_importances = []
    feature_names = X_train.columns.tolist()

    for seed in seeds:
        params = best_params.copy()
        params['seed'] = seed
        
        model = lgb.train(
            params,
            train_data,  # Usar el dataset con pesos
            num_boost_round=1000,
            valid_sets=[train_data],
            callbacks=[
                lgb.early_stopping(50), 
                lgb.log_evaluation(0)
            ]
        )
        
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
        
        importance = model.feature_importance(importance_type='gain')
        feature_importances.append(importance)

    # Promediar predicciones
    final_prediction = np.mean(predictions, axis=0)   
    
    # Crear DataFrame con IDs y predicciones
    result_df = test[['periodo', 'product_id', 'target']].copy()
    result_df['pred'] = final_prediction
    
    # Procesar feature importance
    avg_importance = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    importance_dict = importance_df.set_index('feature')['importance'].to_dict()
    with open(f'./feature_importance/{version}.json', 'w') as f:
        json.dump(importance_dict, f, indent=4)
    
    return result_df