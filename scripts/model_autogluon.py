import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

import pandas as pd




def entrenar_y_predecir(df, periodo_limite=201910):
    """
    Entrena un modelo de predicción de series temporales con variables exógenas y realiza predicciones.
    Divide los datos en train/test según el periodo límite.
    
    Args:
        df: DataFrame con los datos históricos
        periodo_limite: Mes a partir del cual se consideran datos de test (formato YYYYMM)
    """
    # Procesamiento inicial
    dfg = df.groupby(['periodo', 'product_id']).agg({'tn': 'sum'}).reset_index()

    # Convertir periodo a datetime
    dfg['periodo_dt'] = pd.to_datetime(dfg['periodo'].astype(str), format='%Y%m')
    dfg.rename(columns={'tn': 'target', 'product_id':'item_id', 'periodo_dt': 'timestamp'}, inplace=True)

    # Filtrar productos a predecir
    productos_ok = pd.read_csv('../../data/raw/product_id_apredecir201912.csv', sep=',')
    dfg = dfg[dfg['item_id'].isin(productos_ok['product_id'].unique())]

    # Cargar y procesar variables exógenas
    dolar = pd.read_csv("./datasets/dolar_oficial_bna.csv", sep=';', encoding='utf-8')
    ipc = pd.read_csv("./datasets/ipc.csv", sep=';', encoding='utf-8')

    # Convertir y unir variables exógenas
    dolar['periodo_dt'] = pd.to_datetime(dolar['periodo'].astype(str), format='%Y%m')
    ipc['periodo_dt'] = pd.to_datetime(ipc['periodo'].astype(str), format='%Y%m')

    # Unir variables exógenas
    dfg = dfg.merge(
        dolar,
        left_on='timestamp', right_on='periodo_dt', how='left'
    )

    dfg = dfg.merge(
        ipc,
        left_on='timestamp', right_on='periodo_dt', how='left'
    )

    # Dividir en train y test
    train = dfg[dfg['periodo'] < periodo_limite].copy()
    test = dfg[dfg['periodo'] >= periodo_limite].copy()
    
    # Columnas a usar
    columns_to_use = ['item_id', 'timestamp', 'target', 'dolar', 'ipc']
    train = train[columns_to_use]
    test = test[columns_to_use]

    # Crear TimeSeriesDataFrame para entrenamiento
    train_data = TimeSeriesDataFrame.from_data_frame(
        train,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Configurar y entrenar predictor
    predictor = TimeSeriesPredictor(
        target='target',
        prediction_length=2,  # Predecir 2 meses hacia adelante
        freq="M",        
        known_covariates_names=["dolar", "ipc"]
    ).fit(
        train_data,
        num_val_windows=2,
        presets="medium_quality"
    )
    
    # Preparar datos futuros de variables exógenas
    # Usamos los valores REALES del periodo a predecir (de test)
    future_covariates = test[['timestamp', 'dolar', 'ipc', 'item_id']].copy()
    
    # Convertir a TimeSeriesDataFrame
    future_covariates = TimeSeriesDataFrame.from_data_frame(
        future_covariates,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Realizar predicciones
    predictions = predictor.predict(train_data, known_covariates=future_covariates)

    # Procesamiento de resultados
    predictions_v1 = predictions.reset_index()
    predictions_v1 = predictions_v1[["item_id", "timestamp", "mean"]]
    
    # Filtrar la última predicción (mes +2)
    predictions_v1 = predictions_v1[predictions_v1.timestamp == predictions_v1.timestamp.max()]
    
    # Unir con los valores reales para evaluación
    test_eval = test[test['periodo'] == periodo_limite + 2][['item_id', 'target']]  # Datos reales de dic-2019
    predictions_v1 = predictions_v1.merge(
        test_eval,
        on='item_id',
        how='left'
    )
    
    predictions_v1 = predictions_v1.rename(columns={
        "item_id": "product_id",
        "mean": "tn_pred",
        "target": "tn_real"
    })

    # Guardar resultados
    predictions_v1.to_csv("./outputs/prediccion_autogluon_2ventanas.csv", sep=",", index=False)

    return predictions_v1





    



def ensemble_de_ventanasValidacion():
    # Carga y preparación de datos
    df = pd.read_csv("./datasets/periodo_x_producto_con_target.csv", sep=',', encoding='utf-8')
    
    # Verificación y limpieza de datos
    print("Verificando datos...")
    print(f"Filas originales: {len(df)}")
    df = df.dropna(subset=['periodo', 'product_id', 'tn'])
    print(f"Filas después de limpieza: {len(df)}")
    
    # Agregación y transformación
    dfg = df.groupby(['periodo', 'product_id']).agg({'tn': 'sum'}).reset_index()
    dfg['periodo_dt'] = pd.to_datetime(dfg['periodo'].astype(str), format='%Y%m')
    dfg.rename(columns={'tn': 'target', 'product_id':'item_id', 'periodo_dt': 'timestamp'}, inplace=True)
    dfg.drop(columns=['periodo'], inplace=True)

    # Filtrar productos
    productos_ok = pd.read_csv('../../data/raw/product_id_apredecir201912.csv', sep=',')
    dfg = dfg[dfg['item_id'].isin(productos_ok['product_id'].unique())]
    print(f"Productos únicos a predecir: {len(dfg['item_id'].unique())}")
    
    # Conversión a TimeSeriesDataFrame con verificación
    if len(dfg) == 0:
        raise ValueError("El DataFrame está vacío después del filtrado")
    
    try:
        data = TimeSeriesDataFrame.from_data_frame(
            dfg,
            id_column="item_id",
            timestamp_column="timestamp"
        )
        print("TimeSeriesDataFrame creado exitosamente")
        print(f"Número de series temporales: {len(data.item_ids)}")
    except Exception as e:
        raise ValueError(f"Error al crear TimeSeriesDataFrame: {str(e)}")
    
    all_predictions = []
    
    for n_windows in range(1, 3):  # Probando con 2 ventanas
        print(f"\n--- Entrenamiento con {n_windows} ventana(s) ---")
        
        try:
            predictor = TimeSeriesPredictor(
                target='target',
                prediction_length=2,
                freq="M",
                eval_metric="MAPE"
            ).fit(
                data,
                num_val_windows=n_windows,
                verbosity=2  # Más detalle en logs
            )
            
            preds = predictor.predict(data)
            print("Predicciones obtenidas exitosamente")
            
            # Procesamiento robusto de predicciones
            preds_202002 = preds.reset_index()
            preds_202002 = preds_202002[preds_202002['timestamp'] == '2020-02-29']
            
            if len(preds_202002) == 0:
                print(f"Advertencia: No hay predicciones para febrero 2020 con {n_windows} ventanas")
                continue
                
            preds_202002 = preds_202002[["item_id", "mean"]].rename(columns={
                "item_id": "product_id", 
                "mean": f"pred_windows_{n_windows}"
            })
            
            all_predictions.append(preds_202002.set_index("product_id"))
            print(f"Predicciones para {n_windows} ventanas procesadas")
            
        except Exception as e:
            print(f"Error durante el entrenamiento/predicción: {str(e)}")
            continue
    
    # Verificación final antes de consolidar
    if not all_predictions:
        raise ValueError("No se generaron predicciones válidas en ninguna iteración")
    
    print("\nConsolidando resultados...")
    final_df = pd.concat(all_predictions, axis=1)
    print(f"DataFrame consolidado: {final_df.shape}")
    
    # Cálculo seguro del promedio ponderado
    try:
        weights = pd.Series(
            range(1, len(all_predictions)+1), 
            index=[f"pred_windows_{i+1}" for i in range(len(all_predictions))]
        )
        print(f"Pesos aplicados: {weights.to_dict()}")
        
        # Verificar que las columnas existan
        missing_cols = [col for col in weights.index if col not in final_df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        final_df['pred_promedio_ponderado'] = (
            final_df[weights.index].multiply(weights).sum(axis=1) / weights.sum()
        )
    except Exception as e:
        raise ValueError(f"Error al calcular promedio ponderado: {str(e)}")
    
    # Resultado final
    final_df = final_df.reset_index().sort_values('product_id')
    
    # Guardado seguro
    output_path = "./outputs/predicciones_exp_06_autogluon_v1.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nProceso completado exitosamente. Resultados guardados en: {output_path}")
    print(f"Resumen de predicciones:\n{final_df.describe()}")
    
    return final_df