import pandas as pd
import numpy as np  
import gc
from sklearn.preprocessing import OneHotEncoder
import os


def aplicarOHE(df):
    """
    Aplica OneHotEncoder a las columnas categóricas.
    """
    
    # Seleccionar columnas categóricas
    categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Crear codificador
    encoder = OneHotEncoder(sparse_output=False, drop=None)

    # Ajustar y transformar
    encoded = encoder.fit_transform(df[categoricas])

    # Convertir a DataFrame
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categoricas), index=df.index)

    # Unir con el resto del DataFrame
    df_ohe = df.drop(columns=categoricas).join(encoded_df)
    
    # Liberar memoria
    del categoricas, encoded, encoded_df, encoder
    gc.collect()
    
    return df_ohe


def media_y_desvio(df, col, fecha):
    """
    Calcula la media y desviación estándar de una columna específica del DataFrame, agrupando por 'product_id'.
    Se utiliza para normalizar los datos y evitar data leakage.
    """
           
    # Calcular media y desviación estándar por product_id
    
    # Si necesitas desviación estándar poblacional (ddof=0):
    stats = (
        df[df['periodo'].astype(int) <= fecha]
        .groupby('product_id')[col]
        .agg([('mean', 'mean'), ('std', lambda x: x.std(ddof=0))])  # ddof=0 aquí
        .rename(columns={'mean': f'{col}_mean', 'std': f'{col}_std'})
        .reset_index()
    )
    
    # Guardar en CSV para posterior uso
    stats.to_csv(f'./datasets/{col}_stats_{fecha}.csv', index=False, sep=',')
    
    return stats


def normalizar_con_zscore(df, col, fecha=201909):
    """
    Aplica la normalización Z-score a una columna específica del DataFrame, agrupando por 'product_id'.
    Esto se debe aplicar luego de hacer el train-test split para evitar data leakage.
    Guardamos la media y desviación estándar de cada producto para desnormalizar posteriormente (de train, no de test)
    ¿hay ceros?
    """
    
    # Seleccionar columnas numéricas
    
    ## calculo la media, desvio por product_id y lo guardo en csv para posterior
    
    ### Fecha: 2019-09 para testeo local
    ### Fecha: 2019-12 para testeo en Kaggle
    
    ruta_archivo = f'./datasets/{col}_stats_{fecha}.csv'
    
    df_stats = pd.DataFrame()
    
    if os.path.exists(ruta_archivo) and ruta_archivo.endswith('.csv'):
        df_stats = pd.read_csv(ruta_archivo, sep=',')
    else:
        df_stats = media_y_desvio(df, col, fecha)
    
        
    ### Hago el merge
    df = df.merge(df_stats, on='product_id', how='left')
    
    ### Calculo el zscore       
    df[col+'_zscore'] = (df[col] - df[col+'_mean']) / df[col+'_std']
    # df[col+'_zscore'] = df.groupby('product_id')[col].transform(lambda x: (x - x[col+'_mean']) / x[col+'_std'])
    
    # Liberar memoria          
    del ruta_archivo, df_stats
    gc.collect()

    return df



def desnormalizar_con_zscore(df, col):
    """
    Desnormaliza las columnas numéricas del DataFrame utilizando la media y desviación estándar de cada producto.
    """
    
    # Desestandarizar
    ### valor de y_pred * desvio estandar del producto (guardado en un excel) + media del producto (guardado en un excel)
    df[col+'_real'] = df[col+'_zscore'] * df[col+'_std'] + df[col+'_mean']
    
    # Liberar memoria
    gc.collect()
    
    return df


def normalizar_con_periodo201909(df, col):
    """
    Normaliza una columna específica del DataFrame utilizando el valor del periodo 2019-09.
    Esto se debe aplicar luego de hacer el train-test split para evitar data leakage.
    """
    
    fecha = '201909'
    ruta_archivo = f'./datasets/{col}_stats_{fecha}.csv'
    df_stats = pd.read_csv(ruta_archivo, sep=',')
    
    ### Hago el merge
    df = df.merge(df_stats, on='product_id', how='left')
    
    ### Calculo el zscore       
    df[col+'_zscore'] = (df[col] - df[col+'_mean']) / df[col+'_std']
    # df[col+'_zscore'] = df.groupby('product_id')[col].transform(lambda x: (x - x[col+'_mean']) / x[col+'_std'])
    
    # Liberar memoria          
    del ruta_archivo, df_stats
    gc.collect()

    return df


def transformar_logaritmca(df, col):
    """
    Aplica la transformación logarítmica a una columna específica del DataFrame.
    Utiliza log1p para evitar problemas con valores cero.
    """
    df[col+'_log'] = np.log1p(df[col])
    return df


def destransformar_logaritmicamente(df, col):
    """
    Destransforma la columna logarítmica a su valor original.
    Utiliza expm1 para revertir la transformación logarítmica.
    """
    df[col+'_real'] = np.expm1(df[col+'_log'])
    
    # Liberar memoria
    gc.collect()
    
    return df

def eliminar_columnas_calculadas(df):
    """
    Elimina las columnas que fueron calculadas y no son necesarias para el modelo.
    """
    df = df.loc[:, ~df.columns.str.contains('_mean$|_std$', regex=True)]
    return df
