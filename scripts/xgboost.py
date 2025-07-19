import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from optuna.storages import RDBStorage
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
import os
import json






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



def optimizar_xgboost_con_semillerio(train, semillas=[42, 101, 202, 303, 404], version="v1"):
    """
    Optimiza hiperparámetros de XGBoost con Optuna usando semillerío en cada fold.
    """

    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    # train['tn_scaled'] = scaler.fit_transform(train[['target']])
    X = train.drop(columns=[*datetime_cols, 'target'])
    y = train['target']
    
    weights = y / y.max() ########### Probar tambien: np.log1p(y)
    
    
   

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'max_bin': 1024,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'verbosity': 0
        }

        rmse_folds = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_train, w_val = weights.iloc[train_idx], weights.iloc[val_idx]

            rmse_seeds = []

            for seed in semillas:
                params = base_params.copy()
                params['random_state'] = seed

                model = xgb.XGBRegressor(**params)

                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    sample_weight_eval_set=[w_val],
                    early_stopping_rounds=50,
                    verbose=False
                )

                y_pred = model.predict(X_val)
                rmse = mean_squared_error(y_val, y_pred, sample_weight=w_val)
                rmse_seeds.append(rmse)

            rmse_folds.append(np.mean(rmse_seeds))

        return np.mean(rmse_folds)

    def print_best_trial(study, trial):
        print(f"Mejor trial hasta ahora: RMSE={study.best_value:.4f}, Parámetros={study.best_trial.params}")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, callbacks=[print_best_trial], timeout=3600)

    print("\nMejores hiperparámetros:", study.best_params)

    best_params = study.best_params
    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'verbosity': 0,
        'max_bin': 1024
    })

    # Opcional: guardar los parámetros si tenés una función
    guardar_hiperparametros(best_params, version)

    return best_params, study
        






def semillerio_en_prediccion_xgboost(train, test, version="v1"):
    """
    Entrena modelos XGBoost con múltiples semillas y promedia las predicciones.
    """

    datetime_cols = train.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    X_train = train.drop(columns=[*datetime_cols, 'target'])
    y_train = train['target']
    X_test = test.drop(columns=[*datetime_cols, 'target'])

    sample_weight = y_train / y_train.max()  # <-- mismo cálculo que en Optuna


    # Cargar hiperparámetros previamente optimizados
    best_params = levantar_hiperparametros(version)
    
    # Asegurar parámetros mínimos requeridos por XGBRegressor
    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'verbosity': 0,
        'max_bin': 1024
    })

    seeds = [42, 101, 202, 303, 404]
    predictions = []
    feature_importances = []
    feature_names = X_train.columns.tolist()

    for seed in seeds:
        params = best_params.copy()
        params['random_state'] = seed

        model = xgb.XGBRegressor(**params)

        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred = model.predict(X_test)
        predictions.append(y_pred)

        # Feature importance tipo gain
        importance = model.get_booster().get_score(importance_type='gain')

        # Reconvertir a vector completo para que se puedan promediar luego
        importance_vector = [importance.get(f, 0.0) for f in feature_names]
        feature_importances.append(importance_vector)

    # Promedio de predicciones
    final_prediction = np.mean(predictions, axis=0)

    # Armar DataFrame de resultados
    result_df = test[['periodo', 'product_id', 'target']].copy()
    result_df['pred'] = final_prediction

    # Feature importance promedio
    avg_importance = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values(by='importance', ascending=False)

    # Guardar a JSON
    importance_dict = importance_df.set_index('feature')['importance'].to_dict()
    with open(f'./feature_importance/{version}.json', 'w') as f:
        json.dump(importance_dict, f, indent=4)

    return result_df

        
        



#################### Optimus Prime ####################

def evaluate_model(df, periodos_train, periodos_val, features, params, 
                   seeds=[42, 123, 456], use_cv=True, n_splits=5, verbose=False):
    """
    Evalúa un modelo con múltiples semillas. Usa CV temporal interna si use_cv=True.
    """
    df_train = df[df['periodo'].isin(periodos_train)].copy()
    df_val = df[df['periodo'].isin(periodos_val)].copy()
    
    # Validaciones básicas
    if df_train.empty or df_val.empty:
        raise ValueError("Train o Val está vacío.")

    

    X_train = df_train[features]
    y_train = df_train['target']
    ### Peso
    sw_train = df_train['tn_scaled']
    # df_train['tn'] = pd.to_numeric(df_train['tn'], errors='coerce').fillna(1e-5).clip(lower=1e-5)

    X_val = df_val[features]
    y_val = df_val['target']

    if use_cv:
        cv_maes = []
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            fold_maes = []

            for seed in seeds:
                model = xgb.XGBRegressor(**params, random_state=seed)
                model.fit(
                    X_train.iloc[train_idx], y_train.iloc[train_idx],
                    sample_weight=sw_train.iloc[train_idx],
                    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                    eval_metric="mae",
                    callbacks=[xgb.early_stopping(30), xgb.log_evaluation(0)]
                )
                pred = model.predict(X_train.iloc[val_idx])
                fold_maes.append(mean_absolute_error(y_train.iloc[val_idx], pred))

            cv_maes.append(np.mean(fold_maes))
        
        mae_cv = np.mean(cv_maes)
        if verbose:
            print(f"CV MAE: {mae_cv:.4f}")
    else:
        mae_cv = None

    # Entrenamiento final con todo el set de train
    final_preds = []
    models = []

    for seed in seeds:
        model = xgb.XGBRegressor(**params, random_state=seed)
        model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[xgb.early_stopping(30), xgb.log_evaluation(0)]
        )
        pred = model.predict(X_val)
        final_preds.append(pred)
        models.append(model)

    ensemble_pred = np.mean(final_preds, axis=0)
    mae_val = mean_absolute_error(y_val, ensemble_pred)

    if verbose:
        print(f"Ensemble MAE Val: {mae_val:.4f}")

    return mae_cv if use_cv else mae_val, models, ensemble_pred



def create_objective(df, periodos_train, periodos_val, features, use_cv=True):
    """ 
    Crea una función objetivo para Optuna que evalúa un modelo LightGBM.
    Esta función se utiliza para la optimización de hiperparámetros.
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'max_bin': 256,  # Valor máximo como sugerido
            'n_jobs': -1,
            'tree_method': 'hist'  # Para mejor manejo de max_bin
        }

        try:
            mae, _, _ = evaluate_model(df, periodos_train, periodos_val, features, params, use_cv=use_cv)
            return mae
        except Exception as e:
            print(f"Trial error: {e}")
            return float("inf")
    return objective




def run_bayesian_optimization(df, periodos_train, periodos_val, periodos_test,
                              n_trials=20, use_cv=True):
    """ 
    Ejecuta la optimización bayesiana de hiperparámetros para LightGBM.
    
    """
    features = [col for col in df.columns if col not in ['target']]
    print(f"Features: {features}")

    objective = create_objective(df, periodos_train, periodos_val, features, use_cv=use_cv)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        # 'min_child_samples': best_params.pop('min_child_weight') * 5,
        # 'bagging_fraction': best_params.pop('subsample'),
        # 'bagging_freq': 1,
        # 'feature_fraction': best_params.pop('colsample_bytree'),
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'deterministic': True,
        'force_col_wise': True
        # 'linear_tree': True ##### CUIDADO!!!
    })

    print("\n🏆 Mejores parámetros:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    # Entrenamiento final
    print("\nEntrenando modelo final...")
    mae_val, models, _ = evaluate_model(df, periodos_train, periodos_val, features, best_params, use_cv=False, verbose=True)

    # Evaluación en test
    df_test = df[df['periodo'].isin(periodos_test)].copy()
    if df_test.empty:
        print("No hay datos de test")
        return study, models, None

    X_test = df_test[features]
    y_test = df_test['target']

    preds = [model.predict(X_test) for model in models]
    ensemble_pred = np.mean(preds, axis=0)
    mae_test = mean_absolute_error(y_test, ensemble_pred)

    print(f"\n🎯 MAE Test Ensemble: {mae_test:.4f}")

    return study, models, {
        'mae_val': mae_val,
        'mae_test': mae_test,
        'ensemble_pred': ensemble_pred,
        'best_params': best_params
    }



def main_entrenar(df, VERSION="v12"):
    """ 
    Entrena un modelo LightGBM con optimización bayesiana y semillerío.
    """
    scaler = MinMaxScaler(feature_range=(1, 100))
    df['tn_scaled'] = scaler.fit_transform(df[['tn']])
    
    # Configurar períodos
    periodos_train  = [ 201701, 201702, 201703, 201704, 201705, 201706, 201707, 201708, 201709, 201710, 201711, 201712,
                        201801, 201802, 201803, 201804, 201805, 201806, 201807, 201808, 201809, 201810, 201811, 201812,
                        201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201910  ]
    periodos_val    = [ 201907, 201908  ]
    periodos_test   = [ 201909, 201910  ]

    # Ejecutar optimización
    study, models, results = run_bayesian_optimization(
        df=df,
        periodos_train=periodos_train,
        periodos_val=periodos_val,
        periodos_test=periodos_test,
        n_trials=50,
        use_cv=True  # True = validación cruzada temporal dentro de train
    )

    # Análisis adicional
    print("\nTop 5 mejores trials:")
    top_trials = study.trials_dataframe().nsmallest(5, 'value')
    for i, (_, row) in enumerate(top_trials.iterrows(), 1):
        print(f"{i}. MAE: {row['value']:.4f}")

    guardar_hiperparametros(study.best_params, VERSION)
    
    return study, models, results





def predict_with_ensemble(models, df_future, features):
    """
    Realiza predicción con ensemble (promedio) sobre un nuevo dataset.
    """
    if df_future.empty:
        raise ValueError("El dataframe de predicción está vacío.")

    X_future = df_future[features]

    # Promedio de predicciones de todas las semillas
    preds = [model.predict(X_future) for model in models]
    ensemble_pred = np.mean(preds, axis=0)

    return ensemble_pred