import pandas as pd
import gc   
import os

def target_tn_mas_dos(df):
    """
    Crea una nueva columna target que contiene las ventas en el mes+2.
    Si estoy en enero de 2017 entonces target tendrá las toneladas de marzo de 2017
    """
    # Asegurarte de tener 'periodo_dt' (datetime) en completo
    df['periodo_dt'] = pd.to_datetime(df['periodo'], format='%Y%m')

    # Crear DataFrame auxiliar con tn como target y fecha adelantada
    ventas_futuras = df[['periodo_dt', 'product_id', 'tn']].copy()
    ventas_futuras['periodo_target_dt'] = ventas_futuras['periodo_dt'] - pd.DateOffset(months=2)
    ventas_futuras = ventas_futuras.rename(columns={'tn': 'target'})

    # Merge con completo usando periodo adelantado
    df = df.merge(
        ventas_futuras[['periodo_target_dt', 'product_id', 'target']],
        how='left',
        left_on=['periodo_dt', 'product_id'],
        right_on=['periodo_target_dt', 'product_id']
    )

    # Eliminar columna auxiliar
    df = df.drop(columns=['periodo_target_dt', 'periodo_dt'])
    del ventas_futuras
    gc.collect()
    
    return df


def target_tn_mas_dos_transformado(df, col, fecha=201909):
    # Asegurarte de tener 'periodo_dt' (datetime) en completo
    df['periodo_dt'] = pd.to_datetime(df['periodo'], format='%Y%m')

    # Crear DataFrame auxiliar con tn como target y fecha adelantada
    ventas_futuras = df[['periodo_dt', 'product_id', 'tn']].copy()
    
    ruta_archivo = f'./datasets/{col}_stats_{fecha}.csv'
    
    df_stats = pd.DataFrame()
    
    if os.path.exists(ruta_archivo) and ruta_archivo.endswith('.csv'):
        df_stats = pd.read_csv(ruta_archivo, sep=',')
   
    ### Hago el merge
    ventas_futuras = ventas_futuras.merge(df_stats, on='product_id', how='left')   
    
    ### Calculo el zscore       
    ventas_futuras[col+'_zscore'] = (ventas_futuras[col] - ventas_futuras[col+'_mean']) / ventas_futuras[col+'_std']  
    
    ventas_futuras['periodo_target_dt'] = ventas_futuras['periodo_dt'] - pd.DateOffset(months=2)
    ventas_futuras = ventas_futuras.rename(columns={'tn_zscore': 'target'})

    # Merge con completo usando periodo adelantado
    df = df.merge(
        ventas_futuras[['periodo_target_dt', 'product_id', 'target']],
        how='left',
        left_on=['periodo_dt', 'product_id'],
        right_on=['periodo_target_dt', 'product_id']
    )

    # Eliminar columna auxiliar
    df = df.drop(columns=['periodo_target_dt', 'periodo_dt'])
    del ventas_futuras
    gc.collect()
    
    return df