import pandas as pd
import numpy as np
import gc



def combinatoria_periodo_cliente_producto():
    """
    Devuelve df con combinatoria de todos los productos con todos los periodos.
    """
    
    df = pd.read_csv("../../data/preprocessed/base.csv", sep=',')
    
    df["periodo_dt"] = pd.to_datetime(df["periodo"].astype(str), format="%Y%m")

    periodos = pd.date_range(start=df['periodo_dt'].min(), end=df['periodo_dt'].max(), freq="MS")
    productos = df['product_id'].unique()    
    clientes = df['customer_id'].unique()
    
    idx = pd.MultiIndex.from_product([productos, clientes, periodos], names=['product_id', 'customer_id', 'periodo'])
    completo = idx.to_frame(index=False)
        
    del periodos, productos, idx, clientes
    gc.collect()
    
    return completo


def getProductos_sinHistoria(meses = 3):
    """
    Devuelve df con productos cuya historia es <= 3 meses.
    """
    df = pd.read_csv("../../data/preprocessed/base.csv", sep=',')
    historia = df.groupby('product_id')['periodo'].agg(['min', 'max']).reset_index()
    historia['fecha_min'] = pd.to_datetime(historia['min'], format='%Y%m')
    historia['fecha_max'] = pd.to_datetime(historia['max'], format='%Y%m')
    historia['meses_diff'] = (
        (historia['fecha_max'].dt.year - historia['fecha_min'].dt.year) * 12 + 
        (historia['fecha_max'].dt.month - historia['fecha_min'].dt.month) + 1  # +1 para incluir ambos extremos
    )
    prod_sin_historia = historia.loc[historia['meses_diff'] <= meses, 'product_id'].tolist()
    del historia, df
    gc.collect()
    return prod_sin_historia


def eliminarProductos_sinNacer(df, data):
    """
    Elimina productos que no tienen periodo de nacimiento: primera venta.
    """
    df["periodo_dt"] = pd.to_datetime(df["periodo"].astype(str), format="%Y%m")
    data["periodo_dt"] = pd.to_datetime(data["periodo"].astype(str), format="%Y%m")
    
    nacimiento_producto = df.groupby("product_id")["periodo_dt"].agg(["min"]).reset_index()
    # Renombrar columna max a muerte_cliente_dt
    nacimiento_producto = nacimiento_producto.rename(columns={'min': 'nacimiento_producto'})

    # Unir con df_final para traer fecha de muerte del cliente
    data = data.merge(nacimiento_producto, on='product_id', how='left')

    # Filtrar filas donde periodo_dt > muerte_cliente_dt
    data = data[data['periodo_dt'] >= data['nacimiento_producto']]

    data.drop(columns=['periodo_dt'], inplace=True)
    del nacimiento_producto
    gc.collect()
    
    return data


def eliminarProductosMuertos(df, data):
    """
    Elimina productos que murieron: última venta.
    """
    df["periodo_dt"] = pd.to_datetime(df["periodo"].astype(str), format="%Y%m")
    data["periodo_dt"] = pd.to_datetime(data["periodo"].astype(str), format="%Y%m")
    
    muerte_producto = df.groupby("product_id")["periodo_dt"].agg(["max"]).reset_index()
    # Renombrar columna max a muerte_producto
    muerte_producto = muerte_producto.rename(columns={'max': 'muerte_producto'})
    

    # Unir con df_final para traer fecha de muerte del cliente
    data = data.merge(muerte_producto, on='product_id', how='left')

    # Filtrar filas donde periodo_dt > muerte_cliente_dt
    data = data[~((data['periodo_dt'] > data['muerte_producto']) & (data['muerte_producto'] < '2019-12-01'))]

    data.drop(columns=['periodo_dt'], inplace=True)
    del muerte_producto
    gc.collect()
    
    return data


def eliminarClientes_sinNacer(df, data):
    """
    Elimina clientes que no tienen periodo de nacimiento: primera venta.
    """
    df["periodo_dt"] = pd.to_datetime(df["periodo"].astype(str), format="%Y%m")
    data["periodo_dt"] = pd.to_datetime(data["periodo"].astype(str), format="%Y%m")
    
    nacimiento_cliente = df.groupby("customer_id")["periodo_dt"].agg(["min"]).reset_index()
    # Renombrar columna max a muerte_cliente_dt
    nacimiento_cliente = nacimiento_cliente.rename(columns={'min': 'nacimiento_cliente'})

    # Unir con df_final para traer fecha de muerte del cliente
    data = data.merge(nacimiento_cliente, on='customer_id', how='left')

    # Filtrar filas donde periodo_dt > muerte_cliente_dt
    data = data[data['periodo_dt'] >= data['nacimiento_cliente']]

    data.drop(columns=['periodo_dt'], inplace=True)
    del nacimiento_cliente
    gc.collect()
    
    return data


def eliminarClientesMuertos(df, data):
    """
    Elimina clientes que murieron: última venta.
    """
    df["periodo_dt"] = pd.to_datetime(df["periodo"].astype(str), format="%Y%m")
    data["periodo_dt"] = pd.to_datetime(data["periodo"].astype(str), format="%Y%m")
    
    muerte_cliente = df.groupby("customer_id")["periodo_dt"].agg(["max"]).reset_index()
    # Renombrar columna max a muerte_cliente
    muerte_cliente = muerte_cliente.rename(columns={'max': 'muerte_cliente'})
    

    # Unir con df_final para traer fecha de muerte del cliente
    data = data.merge(muerte_cliente, on='customer_id', how='left')

    # Filtrar filas donde periodo_dt > muerte_cliente_dt
    data = data[~((data['periodo_dt'] > data['muerte_cliente']) & (data['muerte_cliente'] < '2019-12-01'))]

    data.drop(columns=['periodo_dt'], inplace=True)
    del muerte_cliente
    gc.collect()
    
    return data

def marcarProductosNuevos_3M(data):
    """
    Productos Nuevos = aquellos que tienen menos de 12 meses de historia.
    Sus primeros 3 meses de historia no se tienen en cuenta para predecir.
    En este metodo se tomó la decisión de no eliminarlos, sino marcarlos.
    """
    data["periodo_dt"] = pd.to_datetime(data["periodo"].astype(str), format="%Y%m")
    
    data = data.sort_values(by=['product_id', 'periodo_dt'])
    
    data['mes_n'] = data.groupby('product_id').cumcount() + 1
    
    meses_totales = data.groupby("product_id")['periodo_dt'].count().rename('total_meses').reset_index()
    
    data = data.merge(meses_totales, on='product_id')
    
    data['producto_nuevo'] = (data['total_meses'] <= 12).astype(int)
    
    
    data['ciclo_de_vida_inicial'] = ((data['mes_n'] <= 3) & (data['producto_nuevo'] == 1)).astype(int)
    
    data.drop(columns=['periodo_dt'], inplace=True)
    
    return data


