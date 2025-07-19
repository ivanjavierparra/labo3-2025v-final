import pandas as pd
import numpy as np

def agregar_ceros_consecutivos_atras(df, col_tn='tn', grupo='product_id'):
    """
    Calcula la cantidad de meses consecutivos con tn == 0 hacia atrás (sin incluir el mes actual),
    para cada producto.
    
    Devuelve un DataFrame con la columna 'ceros_consecutivos_atras'.
    """
    df = df.sort_values([grupo, 'periodo']).copy()
    
    # Crear indicador binario desplazado (solo mira hacia atrás)
    df['venta_cero_shifted'] = (df[col_tn].shift(1) == 0).astype(int)
    
    # Crear agrupación para corte de rachas (0 si la racha se corta)
    df['corte'] = (df[grupo] != df[grupo].shift(1)) | (df['venta_cero_shifted'] == 0)
    df['grupo_racha'] = df['corte'].cumsum()
    
    # Contador dentro de cada racha de ceros
    df['ceros_consecutivos_atras'] = df.groupby(['grupo_racha']).cumcount()
    
    # Asegurar que sólo cuente ceros (los que no vienen de ceros deben ir en 0)
    df.loc[df['venta_cero_shifted'] == 0, 'ceros_consecutivos_atras'] = 0
    
    # Limpieza
    df.drop(columns=['venta_cero_shifted', 'corte', 'grupo_racha'], inplace=True)
    
    return df




def agregar_no_ceros_consecutivos_atras(df, col_tn='tn', grupo='product_id'):
    """
    Calcula la cantidad de meses consecutivos hacia atrás con tn ≠ 0 (ventas distintas de cero),
    sin incluir el mes actual, para cada producto.

    Devuelve un DataFrame con la columna 'no_ceros_consecutivos_atras'.
    """
    df = df.sort_values([grupo, 'periodo']).copy()
    
    # Indicador desplazado (sólo mira hacia atrás)
    df['venta_no_cero_shifted'] = (df[col_tn].shift(1) != 0).astype(int)
    
    # Detectar cortes de racha: nuevo producto o cambio a 0
    df['corte'] = (df[grupo] != df[grupo].shift(1)) | (df['venta_no_cero_shifted'] == 0)
    df['grupo_racha'] = df['corte'].cumsum()
    
    # Contador dentro de cada racha
    df['no_ceros_consecutivos_atras'] = df.groupby('grupo_racha').cumcount()
    
    # Resetear a 0 si el valor previo fue cero
    df.loc[df['venta_no_cero_shifted'] == 0, 'no_ceros_consecutivos_atras'] = 0
    
    # Limpieza
    df.drop(columns=['venta_no_cero_shifted', 'corte', 'grupo_racha'], inplace=True)
    
    return df




def agregar_ceros_ultimos_n_meses(df, col_tn='tn', grupo='product_id', ventanas=[3, 6, 12], incluir_actual=True):
    """
    Agrega columnas con la cantidad de meses con ventas == 0 en las últimas N observaciones
    para cada producto.

    Parámetros:
        df: DataFrame con columnas ['product_id', 'periodo', 'tn']
        col_tn: nombre de la columna de ventas
        grupo: columna de agrupación (ej: product_id)
        ventanas: lista de ventanas (en meses) a considerar
        incluir_actual: si True, incluye el mes actual en la cuenta

    Devuelve:
        DataFrame con columnas nuevas como: 'ventas_cero_ult_3', 'ventas_cero_ult_6', etc.
    """
    df = df.sort_values([grupo, 'periodo']).copy()
    df['es_cero'] = (df[col_tn] == 0).astype(int)

    for n in ventanas:
        nombre_col = f'ventas_cero_ult_{n}'
        if incluir_actual:
            df[nombre_col] = df.groupby(grupo)['es_cero'].transform(lambda x: x.rolling(window=n, min_periods=1).sum())
        else:
            df[nombre_col] = df.groupby(grupo)['es_cero'].transform(lambda x: x.shift(1).rolling(window=n, min_periods=1).sum())

    df.drop(columns='es_cero', inplace=True)
    return df


df = agregar_ceros_ultimos_n_meses(df, ventanas=[3, 6, 12], incluir_actual=False)
"""
Esto te agrega:

    ventas_cero_ult_3

    ventas_cero_ult_6

    ventas_cero_ult_12

Cada uno indicando cuántos meses de los últimos n (sin contar el actual) el producto tuvo ventas = 0.
"""


def agregar_min_max_ult_n(df, n_list=(3, 6, 12), col_tn='tn', grupo='product_id'):
    """
    Agrega columnas con tn mínima y máxima en las últimas n observaciones para cada product_id.
    
    Parámetros:
        df: DataFrame con columnas ['product_id', 'periodo', col_tn]
        n_list: lista o tupla de valores de ventana (ej. [3,6,12])
        col_tn: nombre de la columna de toneladas (default='tn')
        grupo: columna de agrupamiento (default='product_id')
    
    Devuelve:
        DataFrame con columnas tn_min_ult_{n} y tn_max_ult_{n} para cada n.
    """
    df = df.sort_values([grupo, 'periodo']).copy()
    
    for n in n_list:
        df[f'tn_min_ult_{n}'] = (
            df.groupby(grupo)[col_tn]
            .transform(lambda x: x.rolling(window=n, min_periods=1).min())
        )
        df[f'tn_max_ult_{n}'] = (
            df.groupby(grupo)[col_tn]
            .transform(lambda x: x.rolling(window=n, min_periods=1).max())
        )
    
    return df


def agregar_edad_producto(df, fechas_nacimiento, col_periodo='periodo', col_id='product_id'):
    """
    Agrega una columna 'edad_meses' con la edad del producto a cada fila.
    El mes de nacimiento tiene edad = 1.
    
    fechas_nacimiento: dict {product_id: yyyymm}
    """
    def calcular_edad(row):
        nacimiento = fechas_nacimiento.get(row[col_id], None)
        if nacimiento is None:
            return np.nan
        return meses_entre(nacimiento, row[col_periodo]) + 1  # inicio desde 1

    df['edad_meses'] = df.apply(calcular_edad, axis=1)
    return df




##################################################################

def calcular_métricas_producto(df, n_periodos=[3, 6, 12], fechas_muerte=None):
    """
    Calcula métricas de ventas por producto, como rachas de ceros/no-ceros, tn mín/máx, edad, etc.
    
    Parámetros:
        df: DataFrame con columnas ['product_id', 'periodo', 'tn']
        n_periodos: lista con las ventanas de tiempo a analizar
        fechas_muerte: diccionario {product_id: periodo_muerte} (opcional)

    Devuelve:
        DataFrame con métricas por producto y periodo
    """
    df = df.sort_values(['product_id', 'periodo']).copy()

    resultados = []

    for prod_id, grupo in df.groupby('product_id'):
        grupo = grupo.sort_values('periodo').reset_index(drop=True)
        grupo['ventas_cero'] = (grupo['tn'] == 0).astype(int)
        grupo['ventas_no_cero'] = (grupo['tn'] != 0).astype(int)
        
        # 1 y 2: rachas consecutivas de 0 y no-0
        grupo['racha_0'] = grupo['ventas_cero'] * (grupo['ventas_cero'].groupby((grupo['ventas_cero'] != grupo['ventas_cero'].shift()).cumsum()).cumcount() + 1)
        grupo['racha_no_0'] = grupo['ventas_no_cero'] * (grupo['ventas_no_cero'].groupby((grupo['ventas_no_cero'] != grupo['ventas_no_cero'].shift()).cumsum()).cumcount() + 1)
        
        grupo['meses_0_consecutivos'] = grupo['racha_0']
        grupo['meses_no_0_consecutivos'] = grupo['racha_no_0']

        # 3 y 4 y 5: en las últimas n ventanas
        for n in n_periodos:
            col_base = f'ult_{n}'
            grupo[f'meses_0_ult_{n}'] = grupo['ventas_cero'].rolling(window=n, min_periods=1).sum()
            grupo[f'tn_min_ult_{n}'] = grupo['tn'].rolling(window=n, min_periods=1).min()
            grupo[f'tn_max_ult_{n}'] = grupo['tn'].rolling(window=n, min_periods=1).max()
        
        # 6: edad del producto
        grupo['edad_meses'] = np.arange(1, len(grupo) + 1)

        # 7: meses hasta la muerte (si se conoce)
        if fechas_muerte and prod_id in fechas_muerte:
            periodo_muerte = fechas_muerte[prod_id]
            grupo['meses_hasta_muerte'] = (periodo_muerte - grupo['periodo']) // 100 * 12 + (periodo_muerte % 100 - grupo['periodo'] % 100)
            grupo['meses_hasta_muerte'] = grupo['meses_hasta_muerte'].clip(lower=0)
        else:
            grupo['meses_hasta_muerte'] = 999  # valor alto si no se conoce

        resultados.append(grupo)

    df_final = pd.concat(resultados, ignore_index=True)
    return df_final


fechas_muerte = {
    'P001': 202412,
    'P002': 202310,
    # ...
}

df_metrico = calcular_métricas_producto(df, n_periodos=[3, 6, 12], fechas_muerte=fechas_muerte)
