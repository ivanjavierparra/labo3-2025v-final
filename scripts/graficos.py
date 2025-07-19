import pandas as pd
import numpy as np
import gc
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt




def graficar_780(df_pred_201912_real):
    
    productos_ok = pd.read_csv('../../data/raw/product_id_apredecir201912.csv', sep=',', encoding='utf-8')
    df = pd.read_csv('./datasets/periodo_x_producto.csv', sep=',', encoding='utf-8')    
    df = df[df['product_id'].isin(productos_ok['product_id'].unique())]

    

 

    # Asegurar formato de fechas
    df['periodo'] = pd.to_datetime(df['periodo'], format='%Y%m')
    df_pred_201912_real['periodo'] = 201912
    df_pred_201912_real['periodo'] = pd.to_datetime(df_pred_201912_real['periodo'], format='%Y%m')

    # Merge real + predicción
    df_merged = df.merge(df_pred_201912_real, on=['periodo', 'product_id'], how='left')

    # Lista de productos
    product_ids = sorted(df['product_id'].unique())

    # Graficar 2 productos por figura
    for i in range(0, len(product_ids), 2):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

        for j in range(2):
            if i + j < len(product_ids):
                product_id = product_ids[i + j]
                ax = axes[j]

                df_prod = df_merged[df_merged['product_id'] == product_id].sort_values('periodo')

                # Línea de valores reales
                ax.plot(df_prod['periodo'], df_prod['tn'], label='Real', color='black')

                # Punto de predicción (si existe)
                pred_row = df_prod[df_prod['pred'].notnull()]
                if not pred_row.empty:
                    ax.scatter(pred_row['periodo'], pred_row['pred'], color='red', label='Predicción', zorder=5)

                ax.set_title(f'Producto {product_id}')
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
            else:
                fig.delaxes(axes[j])  # Borrar subplot vacío

        fig.tight_layout()
        plt.show()
        # plt.savefig(f'grafico_productos_{i}_{i+1}.png')  # O plt.show() si querés verlos
        # plt.close(fig)





    
    

def grafico_interactivo_780():
    
    productos_ok = pd.read_csv('../../data/raw/product_id_apredecir201912.csv', sep=',', encoding='utf-8')
    dfa = pd.read_csv('./datasets/periodo_x_producto.csv', sep=',', encoding='utf-8')
    df = dfa.copy()
    df = df[df['product_id'].isin(productos_ok['product_id'].unique())]
    
    df['periodo_dt'] = pd.to_datetime(df['periodo'], format='%Y%m')
    
    fig = px.line(df, x='periodo_dt', y='tn', color='product_id',
                  title='Ventas por Producto (Interactivo)',
                  labels={'toneladas': 'Toneladas Vendidas', 'periodo_dt': 'Período'},
                  hover_data={'periodo': True, 'product_id': True},
                  template='plotly_white')
    
    fig.update_traces(mode='lines+markers', marker_size=8)
    fig.update_layout(hovermode='x unified')
    fig.show()