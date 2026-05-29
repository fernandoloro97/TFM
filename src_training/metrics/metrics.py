
import boto3
import pickle
import numpy as np
import pandas as pd


# Configuracion de AWS 
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
s3 = boto3.client('s3')

# Transformo la tabla de dynmodb a dataframe
def get_table_df(table_name):
    table = dynamodb.Table(table_name)
    response = table.scan()
    data = response['Items']
    
    # Manejo la contiuidad para tablas grandes
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    
    return pd.DataFrame(data)

sp500_precio_sesion = (
        get_table_df('period_sp500_sesion_close_prices')
        .assign(Date=lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        .sort_index()
    )
sp500_precio_sesion['SP500'] = sp500_precio_sesion['SP500'].astype(float)


# Fijo los nombres del bucket y pickle donde estan las inferencias
NOMBRE_BUCKET = 'top20-trading-results'
NOMBRE_ARCHIVO = 'resutados_backtest_completo.pkl'

response = s3.get_object(Bucket=NOMBRE_BUCKET, Key=NOMBRE_ARCHIVO)
archivo_bytes = response['Body'].read()

# Reconstruye tu diccionario manteniendo el 100% de los tipos de datos originales
resultados_mercado = pickle.loads(archivo_bytes)
print(f"Cargue el pickle exitosamentes, donde tengo cargadas en el diccionario: {len(resultados_mercado)} elementos", flush=True)


# Listas auxiliares para recopilar los datos de cada ejecución
todos_los_resultados = []

# Obtener el calendario de días hábiles globales (normalizado)
dias_disponibles = sorted(
    pd.to_datetime(sp500_precio_sesion.index).normalize().unique()
)
dias_series = pd.Series(dias_disponibles)

print("--- INICIANDO PROCESAMIENTO TEMPORAL RESTRUCTURADO ---")

# -------------------------------------------------------------------------
# BUCLE PRINCIPAL: Recorrer directamente los modelos en resultados_mercado (0 al 19)
# -------------------------------------------------------------------------
for ranking_id, variantes_backtest in resultados_mercado.items():

    for variante_id, datos_variante in variantes_backtest.items():

        # 1. Validar existencia y preparación de df_resultado de esta variante
        if "df_resultado" not in datos_variante or datos_variante["df_resultado"] is None:
            print(f"⚠️ Modelo {ranking_id} -> Variante {variante_id}: Saltada por falta de df_resultado.")
            continue
            
        df_res = datos_variante["df_resultado"].copy()
        if len(df_res) == 0:
            print(f"⚠️ Modelo {ranking_id} -> Variante {variante_id}: Saltada por df_resultado vacío.")
            continue

        # Convertir columna Date y normalizar para eliminar horas, minutos y segundos
        df_res["Date"] = pd.to_datetime(df_res["Date"])
        fecha_entrada_cruda = df_res["Date"].iloc[0].normalize()
        fecha_final_cruda = df_res["Date"].iloc[-1].normalize()

        # 2. CALCULAR FECHA ENTRADA (-1 día hábil en dias_series)
        idx_entrada_base = dias_series[dias_series == fecha_entrada_cruda].index
        if len(idx_entrada_base) == 0:
            # Respaldo si cayó en fin de semana: buscar el primer día hábil posterior
            dias_posteriores = dias_series[dias_series >= fecha_entrada_cruda]
            if len(dias_posteriores) == 0:
                continue
            idx_base_in = dias_posteriores.index[0]
        else:
            idx_base_in = idx_entrada_base[0]
            
        idx_entrada_final = max(0, idx_base_in - 1)
        fecha_entrada = dias_series.iloc[idx_entrada_final].normalize()

        # 3. CALCULAR FECHA SALIDA (+2 días hábiles en dias_series)
        idx_final_base = dias_series[dias_series == fecha_final_cruda].index
        if len(idx_final_base) == 0:
            # Respaldo si cayó en fin de semana: buscar el primer día hábil posterior
            dias_posteriores_out = dias_series[dias_series >= fecha_final_cruda]
            if len(dias_posteriores_out) == 0:
                continue
            idx_base_out = dias_posteriores_out.index[0]
        else:
            idx_base_out = idx_final_base[0]
            
        idx_salida_final = min(idx_base_out + 2, len(dias_series) - 1)
        fecha_salida = dias_series.iloc[idx_salida_final].normalize()

        # 4. Obtener y preparar el df_balance de la variante actual
        df_balance = datos_variante["df_balance"].copy()
        df_balance["fecha"] = pd.to_datetime(df_balance["fecha"])

        if len(df_balance) == 0:
            print(f"⚠️ Modelo {ranking_id} -> Variante {variante_id}: Saltada por df_balance vacío.")
            continue

        # 5. Filtrar el periodo de evaluación expandido sobre df_balance
        periodo_filtrado = df_balance[
            (df_balance["fecha"] >= fecha_entrada)
            & (df_balance["fecha"] <= fecha_salida)
        ].reset_index(drop=True)

        # Si el filtro no tiene suficientes registros para calcular retornos, saltar
        if len(periodo_filtrado) <= 1:
            print(f"⚠️ Modelo {ranking_id} -> Variante {variante_id}: Datos insuficientes tras filtrar.")
            continue

        # 6. Calcular rendimientos diarios logarítmicos
        periodo_filtrado["ret_diario_log"] = np.log(
            periodo_filtrado["equity_total"]
            / periodo_filtrado["equity_total"].shift(1)
        )

        num_dias_mercado = len(periodo_filtrado) - 1
        fraccion_ano_bursatil = num_dias_mercado / 252

        # 7. Cálculo de métricas
        rent_total_log = periodo_filtrado["ret_diario_log"].sum()
        rent_anualizada = rent_total_log / fraccion_ano_bursatil

        vol_diaria = periodo_filtrado["ret_diario_log"].std()
        vol_anualizada = vol_diaria * np.sqrt(252)

        sharpe_anualizado = (
            (rent_anualizada / vol_anualizada) if vol_anualizada != 0 else 0
        )

        picos = periodo_filtrado["equity_total"].cummax()
        drawdowns = (periodo_filtrado["equity_total"] - picos) / picos
        max_drawdown = drawdowns.min()

        # Sortino
        ret_negativos = periodo_filtrado["ret_diario_log"][
            periodo_filtrado["ret_diario_log"] < 0
        ]
        vol_downside_anualizada = ret_negativos.std() * np.sqrt(252)
        sortino_anualizado = (
            (rent_anualizada / vol_downside_anualizada)
            if vol_downside_anualizada != 0
            else 0
        )

        # Calmar
        calmar_ratio = (
            (rent_anualizada / abs(max_drawdown)) if max_drawdown != 0 else 0
        )

        # Win Rate y P/L
        dias_ganadores = periodo_filtrado["ret_diario_log"] > 0
        dias_perdedores = periodo_filtrado["ret_diario_log"] < 0
        win_rate = dias_ganadores.sum() / num_dias_mercado

        avg_ganancia = periodo_filtrado.loc[
            dias_ganadores, "ret_diario_log"
        ].mean()
        avg_perdida = periodo_filtrado.loc[
            dias_perdedores, "ret_diario_log"
        ].mean()
        pl_ratio = (
            abs(avg_ganancia / avg_perdida) if avg_perdida != 0 else 0
        )

        # Duración máxima Drawdown
        en_drawdown = periodo_filtrado["equity_total"] < picos
        racha_drawdown = en_drawdown.groupby((~en_drawdown).cumsum()).cumsum()
        max_duracion_drawdown = racha_drawdown.max()

        # 8. Guardar metadatos y métricas calculadas
        registro = {
            "Ranking Modelo": ranking_id,
            "Variante Id": variante_id,
            "Modelo Id": datos_variante["model_id"],
            "Modelo": datos_variante["modelo"],
            "Ventana Original": datos_variante["ventana_original"],
            "Ventana Minutos": datos_variante["ventana_minutos"],
            "Capital Inicial": datos_variante["capital_inicial"],
            "Trailing": datos_variante["trailing"],
            "Limite Exposicion": datos_variante["usar_limite_exposicion"],
            "Asig. Riesgo": datos_variante["perc_riesgo"],
            "Fecha Inicio": fecha_entrada.strftime("%Y-%m-%d"),
            "Fecha Salida": fecha_salida.strftime("%Y-%m-%d"),
            "Días Activos": num_dias_mercado,
            "Frac. Año Evaluada": round(fraccion_ano_bursatil, 4),
            "Rentabilidad": f"{rent_anualizada:.4%}",
            "Volatilidad": f"{vol_anualizada:.4%}",
            "Ratio Sharpe": f"{sharpe_anualizado:.4f}",
            "Ratio Sortino": f"{sortino_anualizado:.4f}",
            "Ratio Calmar": f"{calmar_ratio:.4f}",
            "Máx. Drawdown": f"{max_drawdown:.4%}",
            "Máx. Duración Drawdown": int(max_duracion_drawdown),
            "% Dias Ganadores": f"{win_rate:.2%}",
            "Ratio G/P Diario": f"{pl_ratio:.2f}",
        }

        todos_los_resultados.append(registro)

print("--- FIN DEL PROCESAMIENTO ---")

# 9. Crear el DataFrame unificado final
df_consolidado_modelos = pd.DataFrame(todos_los_resultados)
print(f"\nFilas totales consolidadas con éxito: {df_consolidado_modelos.shape}")


# 1. Crear una copia limpia para trabajar los números
df_score = df_consolidado_modelos.copy()

# 2. Limpiar y convertir las columnas de texto a formato numérico
df_score["Rentabilidad_n"] = (
    df_score["Rentabilidad"].str.rstrip("%").astype(float) / 100
)
df_score["Volatilidad_n"] = (
    df_score["Volatilidad"].str.rstrip("%").astype(float) / 100
)
df_score["Max_DD_n"] = (
    df_score["Máx. Drawdown"].str.rstrip("%").astype(float) / 100
)
df_score["Sharpe_n"] = df_score["Ratio Sharpe"].astype(float)
df_score["Sortino_n"] = df_score["Ratio Sortino"].astype(float)
df_score["Calmar_n"] = df_score["Ratio Calmar"].astype(float)
df_score["Duracion_DD_n"] = df_score["Máx. Duración Drawdown"].astype(float)

# 3. Calcular el Z-Score para cada métrica clave
df_Z = pd.DataFrame(index=df_score.index)

# En estas métricas, un valor MÁS ALTO es mejor:
for col in ["Sortino_n", "Calmar_n", "Rentabilidad_n"]:
    std = df_score[col].std()
    df_Z[col] = (df_score[col] - df_score[col].mean()) / std if std != 0 else 0

# En estas métricas, un valor MÁS BAJO es mejor (invertimos el signo):
for col in ["Max_DD_n", "Duracion_DD_n"]:
    std = df_score[col].std()
    df_Z[col] = (
        -1 * ((df_score[col] - df_score[col].mean()) / std) if std != 0 else 0
    )

# 4. Definir los pesos de importancia (Suman 1.0)
pesos = {
    "Sortino_n": 0.35,  # 35% - Premia ganar penalizando solo las caídas malas
    "Calmar_n": 0.25,  # 25% - Premia la relación Retorno / Peor racha
    "Max_DD_n": 0.15,  # 15% - Castiga la profundidad de la pérdida
    "Duracion_DD_n": 0.15,  # 15% - Castiga pasar demasiado tiempo estancado
    "Rentabilidad_n": 0.10,  # 10% - Aporta un extra por beneficio puro
}

# 5. Calcular el Score de Robustez final (Se añade automáticamente al final del DF)
df_consolidado_modelos["Score Robustez"] = sum(
    df_Z[col] * peso for col, peso in pesos.items()
).round(4)

# 6. Ordenar el DataFrame original de forma descendente por el Score
df_consolidado_modelos = df_consolidado_modelos.sort_values(
    by="Score Robustez", ascending=False
).reset_index(drop=True)

# 7. Mostrar TODO el DataFrame completo en consola/Jupyter
print(
    f"--- DATAFRAME COMPLETO CLASIFICADO POR ROBUSTEZ ({len(df_consolidado_modelos)} filas) ---"
)
# Si usas Jupyter Notebook, simplemente escribe: df_consolidado_modelos
# Si usas un script normal de Python (.py), usa print()
df_consolidado_modelos = df_consolidado_modelos.round(4)

print(f"\nMetricas del sistema trading de los 20 mejores modelos de prediccion: {df_consolidado_modelos.shape}")
