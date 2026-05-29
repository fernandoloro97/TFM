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

# Descargo la tabla del SP500 solo para tener sus fechas
sp500_precio_sesion = (
        get_table_df('period_sp500_sesion_close_prices')
        .assign(Date=lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        .sort_index()
    )
# Cambio a float sus valores por si acaso
sp500_precio_sesion['SP500'] = sp500_precio_sesion['SP500'].astype(float)


# Fijo los nombres del bucket y pickle donde estan las inferencias
NOMBRE_BUCKET = 'top20-trading-results'
NOMBRE_ARCHIVO = 'resultados_backtest_completo.pkl'

response = s3.get_object(Bucket=NOMBRE_BUCKET, Key=NOMBRE_ARCHIVO)
archivo_bytes = response['Body'].read()

# Reconstruyo el diccionario de pickle
resultados_mercado = pickle.loads(archivo_bytes)

# Fijo los nombres del bucket y pickle donde estan las inferencias
NOMBRE_BUCKET_CAPITALES = 'capitals-trading-results'
NOMBRE_ARCHIVO_CAPITALES = 'resultados_backtest_capitales.pkl'

response_capitales = s3.get_object(Bucket=NOMBRE_BUCKET, Key=NOMBRE_ARCHIVO)
archivo_bytes_capitales= response_capitales['Body'].read()

# Reconstruyo el diccionario de pickle
resultados_mercado = pickle.loads(archivo_bytes)
resultados_capitales = pickle.loads(archivo_bytes_capitales)
print(f"Cargue el pickle  de 20 modelos exitosamentes, donde tengo cargadas en el diccionario: {len(resultados_mercado)} elementos", flush=True)
print(f"Cargue el pickle  de capitales exitosamentes, donde tengo cargadas en el diccionario: {len(resultados_capitales)} elementos", flush=True)



# Listado para recopilar las metricas calculadas y mas data
todos_los_resultados = []

# Obtener el calendario de dias con sesion viendo al indice
dias_disponibles = sorted(
    pd.to_datetime(sp500_precio_sesion.index).normalize().unique()
)
dias_series = pd.Series(dias_disponibles)

print("Inicio el proceso de calculo de las metricas para los 20 mejores modelos")

# Calculo las metricas de rendimiento en el mercado para los 20 modelos con mayor valor predictivo
for ranking_id, variantes_backtest in resultados_mercado.items():

    for variante_id, datos_variante in variantes_backtest.items():

        # Valido la existencia y preparacion de df_resultado de esta variante
        if "df_resultado" not in datos_variante or datos_variante["df_resultado"] is None:
            continue
            
        df_res = datos_variante["df_resultado"].copy()
        if len(df_res) == 0:
            continue

        # Solo me quedo con la fecha y quito la hora
        df_res["Date"] = pd.to_datetime(df_res["Date"])
        fecha_entrada_cruda = df_res["Date"].iloc[0].normalize()
        fecha_final_cruda = df_res["Date"].iloc[-1].normalize()

        # Calculo la fecha de entrada
        idx_entrada_base = dias_series[dias_series == fecha_entrada_cruda].index
        if len(idx_entrada_base) == 0:
            dias_posteriores = dias_series[dias_series >= fecha_entrada_cruda]
            if len(dias_posteriores) == 0:
                continue
            idx_base_in = dias_posteriores.index[0]
            
        else:
            idx_base_in = idx_entrada_base[0]
            
        idx_entrada_final = max(0, idx_base_in - 1)
        fecha_entrada = dias_series.iloc[idx_entrada_final].normalize()

        # Calculo la fecha de salida
        idx_final_base = dias_series[dias_series == fecha_final_cruda].index
        if len(idx_final_base) == 0:
            dias_posteriores_out = dias_series[dias_series >= fecha_final_cruda]
            if len(dias_posteriores_out) == 0:
                continue
            idx_base_out = dias_posteriores_out.index[0]
            
        else:
            idx_base_out = idx_final_base[0]
            
        idx_salida_final = min(idx_base_out + 2, len(dias_series) - 1)
        fecha_salida = dias_series.iloc[idx_salida_final].normalize()

        # Obtengo y preparo el df_balance de la variante acutal
        df_balance = datos_variante["df_balance"].copy()
        df_balance["fecha"] = pd.to_datetime(df_balance["fecha"])

        if len(df_balance) == 0:
            continue

        # Filtro el periodo de evaluacoin expandido sobre el df_balance
        periodo_filtrado = df_balance[
            (df_balance["fecha"] >= fecha_entrada)
            & (df_balance["fecha"] <= fecha_salida)
        ].reset_index(drop=True)

        # Si no hay suficientes registros para calcular retornos, me salto
        if len(periodo_filtrado) <= 1:
            continue

        # Calculo los rendimientos diarios logaritmicos
        periodo_filtrado["ret_diario_log"] = np.log(
            periodo_filtrado["equity_total"]
            / periodo_filtrado["equity_total"].shift(1)
        )

        num_dias_mercado = len(periodo_filtrado) - 1
        fraccion_ano_bursatil = num_dias_mercado / 252

        # Calculo el resto de metricas de rendimiento, riesgo y ratio de estos
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

        ret_negativos = periodo_filtrado["ret_diario_log"][
            periodo_filtrado["ret_diario_log"] < 0
        ]
        vol_downside_anualizada = ret_negativos.std() * np.sqrt(252)
        sortino_anualizado = (
            (rent_anualizada / vol_downside_anualizada)
            if vol_downside_anualizada != 0
            else 0
        )

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

        en_drawdown = periodo_filtrado["equity_total"] < picos
        racha_drawdown = en_drawdown.groupby((~en_drawdown).cumsum()).cumsum()
        max_duracion_drawdown = racha_drawdown.max()

        # Guardo los metadatos y metricas
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


# Creo el df final con todas las variantes
df_consolidado_modelos = pd.DataFrame(todos_los_resultados)
print(f"\nFilas totales consolidadas con exito: {df_consolidado_modelos.shape}")


# Creo una copia limpia para modificar
df_score = df_consolidado_modelos.copy()

# Limpio y convierto las columnas de texto a formato numerico
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

# Calculo el z-score de cada metrica elegida para que sean comparables
df_Z = pd.DataFrame(index=df_score.index)

# Por tanto, si tiene una valor mas alto dentro del universo,es mejor para estas metricas de rendimiento y ratios
for col in ["Sortino_n", "Calmar_n", "Rentabilidad_n"]:
    std = df_score[col].std()
    df_Z[col] = (df_score[col] - df_score[col].mean()) / std if std != 0 else 0

# Si tiene un valor mas bajo dentro del universo, es mejor para estas metricas de riesgo
for col in ["Max_DD_n", "Duracion_DD_n"]:
    std = df_score[col].std()
    df_Z[col] = (
        -1 * ((df_score[col] - df_score[col].mean()) / std) if std != 0 else 0
    )

# Defino los pesos de las metricas de interes para crear el score de robustez
pesos = {
    "Sortino_n": 0.35,  
    "Calmar_n": 0.25,  
    "Max_DD_n": 0.15,  
    "Duracion_DD_n": 0.15,  
    "Rentabilidad_n": 0.10, 
}

# Calculo el score de robuste con los calculos previos y lo añado al df principal
df_consolidado_modelos["Score Robustez"] = sum(
    df_Z[col] * peso for col, peso in pesos.items()
).round(4)

# Ordeno el df principal de forma ascendentemente por el score de robustez
df_consolidado_modelos = df_consolidado_modelos.sort_values(
    by="Score Robustez", ascending=False
).reset_index(drop=True)

# Rendonde los metricas a 4 decimales
df_consolidado_modelos = df_consolidado_modelos.round(4)

print(f"\nMetricas del sistema trading de los 20 mejores modelos de prediccion: {df_consolidado_modelos.shape}")



print("\nIncio el proceso de analisis de que configuracion de las 3 posibles generar mejores resultados en el mercado")
# Creo una copia de df de resultados  para poder promediar
df_analisis = df_consolidado_modelos.copy()

# Convierto las metricas a float
df_analisis["Rentabilidad_Num"] = (
    df_analisis["Rentabilidad"].str.rstrip("%").astype(float) / 100
)
df_analisis["Volatilidad_Num"] = (
    df_analisis["Volatilidad"].str.rstrip("%").astype(float) / 100
)
df_analisis["Max_DD_Num"] = (
    df_analisis["Máx. Drawdown"].str.rstrip("%").astype(float) / 100
)
df_analisis["Sharpe_Num"] = df_analisis["Ratio Sharpe"].astype(float)
df_analisis["Sortino_Num"] = df_analisis["Ratio Sortino"].astype(float)
df_analisis["Score_Robustez"] = df_analisis["Score Robustez"].astype(float)


# Modifico la funcion para que retorne un bloque de filas estructurado
def calcular_bloque_parametro(columna_parametro):
    # Agrupo por el parametro y calculo la media aritmetica
    resumen = df_analisis.groupby(columna_parametro)[
        [
            "Score_Robustez",
            "Rentabilidad_Num",
            "Volatilidad_Num",
            "Max_DD_Num",
            "Sharpe_Num",
            "Sortino_Num",
        ]
    ].mean()

    # Formateo las columnas visuales
    resumen_visual = pd.DataFrame(index=resumen.index)
    resumen_visual["Score Robustez"] = resumen["Score_Robustez"].round(4)
    resumen_visual["Rentabilidad"] = (
        resumen["Rentabilidad_Num"] * 100
    ).round(2).astype(str) + "%"
    resumen_visual["Volatilidad"] = (
        resumen["Volatilidad_Num"] * 100
    ).round(2).astype(str) + "%"
    resumen_visual["Máx. Drawdown"] = (
        resumen["Max_DD_Num"] * 100
    ).round(2).astype(str) + "%"
    resumen_visual["Ratio Sharpe"] = resumen["Sharpe_Num"].round(2)
    resumen_visual["Ratio Sortino"] = resumen["Sortino_Num"].round(2)

    # Convierto a true o false para algunos tipos de configuracion
    resumen_visual = resumen_visual.reset_index()
    resumen_visual = resumen_visual.rename(
        columns={columna_parametro: "Valor Configuración"}
    )

    # Añado la columna identificadora del parametro al principio
    resumen_visual.insert(0, "Parámetro", columna_parametro)

    return resumen_visual

# Parametros que configuran si mi sistema trading es robusto o basico
parametros_interes = ["Trailing", "Limite Exposicion", "Asig. Riesgo"]

lista_bloques = []

for param in parametros_interes:
    bloque = calcular_bloque_parametro(param)

    # Fuerzo esta columna a string
    bloque["Valor Configuración"] = bloque["Valor Configuración"].astype(str)

    # Corrijo la presentacion de opciones para valores booleanos
    if param in ["Trailing", "Limite Exposicion"]:
        mapeo_booleano = {
            "True": "True",
            "False": "False",
            "1.0": "True",
            "0.0": "False",
            "1": "True",
            "0": "False",
        }
        bloque["Valor Configuración"] = bloque["Valor Configuración"].map(
            mapeo_booleano
        )

    lista_bloques.append(bloque)

# Concateno todas las filas
df_resumen_global = pd.concat(lista_bloques, ignore_index=True)

print(f"Comparacion por configuracion de sistema tradign conseguido, reviso el shape: {df_resumen_global.shape}")




print("\nIncio el analisis del algoritmo en distintos capitales") 
# Listas auxiliares para recopilar los datos de cada ejecución
resultados_distintos_capitales = []

# Obtener el calendario de días hábiles globales (normalizado)
dias_disponibles = sorted(
    pd.to_datetime(sp500_precio_sesion.index).normalize().unique()
)
dias_series = pd.Series(dias_disponibles)

# ID del modelo fijo
ranking_id = 15

print(f"PROCESANDO VARIANTES DE CAPITAL PARA MODELO {ranking_id}")

# Obtener las variantes de backtest para el modelo 15 desde resultados_capitales
variantes_backtest = resultados_capitales[ranking_id]

# Calculo las metricas de rendimiento en el mercado para el modelo optimo con distintso capitales
for variante_id, datos_variante in variantes_backtest.items():

    # Valido la existencia y preparacion de df_resultado de esta variante
    if "df_resultado" not in datos_variante or datos_variante["df_resultado"] is None:
        continue
        
    df_res = datos_variante["df_resultado"].copy()
    if len(df_res) == 0:
        print(f"⚠️ Variante {variante_id}: Saltada por df_resultado vacío.")
        continue

    # Solo me quedo con la fecha y quito la hora
    df_res["Date"] = pd.to_datetime(df_res["Date"])
    fecha_entrada_cruda = df_res["Date"].iloc[0].normalize()
    fecha_final_cruda = df_res["Date"].iloc[-1].normalize()

    # Calculo la fecha de entrada
    idx_entrada_base = dias_series[dias_series == fecha_entrada_cruda].index
    if len(idx_entrada_base) == 0:
        dias_posteriores = dias_series[dias_series >= fecha_entrada_cruda]
        if len(dias_posteriores) == 0:
            continue
        idx_base_in = dias_posteriores.index[0]
    else:
        idx_base_in = idx_entrada_base[0]
        
    idx_entrada_final = max(0, idx_base_in - 1)
    fecha_entrada = dias_series.iloc[idx_entrada_final].normalize()

    # Calculo la fecha de salida
    idx_final_base = dias_series[dias_series == fecha_final_cruda].index
    if len(idx_final_base) == 0:
        dias_posteriores_out = dias_series[dias_series >= fecha_final_cruda]
        if len(dias_posteriores_out) == 0:
            continue
        idx_base_out = dias_posteriores_out.index[0]
    else:
        idx_base_out = idx_final_base[0]
        
    idx_salida_final = min(idx_base_out + 2, len(dias_series) - 1)
    fecha_salida = dias_series.iloc[idx_salida_final].normalize()

    # Obtengo y preparo el df_balance de cada capital
    df_balance = datos_variante["df_balance"].copy()
    df_balance["fecha"] = pd.to_datetime(df_balance["fecha"])

    if len(df_balance) == 0:
        continue

    # Filtro el periodo de evaluacion expandido sobre df_balance
    periodo_filtrado = df_balance[
        (df_balance["fecha"] >= fecha_entrada)
        & (df_balance["fecha"] <= fecha_salida)
    ].reset_index(drop=True)

    # Si el filtro devuelve datos vacios por desfase, saltar a la siguiente variante
    if len(periodo_filtrado) <= 1:
        continue

    # Calculo los rendimientos diarios logaritmicos
    periodo_filtrado["ret_diario_log"] = np.log(
        periodo_filtrado["equity_total"]
        / periodo_filtrado["equity_total"].shift(1)
    )

    num_dias_mercado = len(periodo_filtrado) - 1
    fraccion_ano_bursatil = num_dias_mercado / 252

    # Calculos de metricas de rendimiento, riesgo y ratios
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

    ret_negativos = periodo_filtrado["ret_diario_log"][
        periodo_filtrado["ret_diario_log"] < 0
    ]
    vol_downside_anualizada = ret_negativos.std() * np.sqrt(252)
    sortino_anualizado = (
        (rent_anualizada / vol_downside_anualizada)
        if vol_downside_anualizada != 0
        else 0
    )

    calmar_ratio = (
        (rent_anualizada / abs(max_drawdown)) if max_drawdown != 0 else 0
    )

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

    en_drawdown = periodo_filtrado["equity_total"] < picos
    racha_drawdown = en_drawdown.groupby((~en_drawdown).cumsum()).cumsum()
    max_duracion_drawdown = racha_drawdown.max()

    # Guardo los metadatos y metricas calculadas en el diccionario
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

    resultados_distintos_capitales.append(registro)


# Creo el df con los resultados por cada capital
metricas_distintos_capitales = pd.DataFrame(resultados_distintos_capitales)
print(f"\nFilas totales consolidadas con exito: {metricas_distintos_capitales.shape}")


# Creo una copia limpia para modificar
df_score = metricas_distintos_capitales.copy()

# Limpio y convierto las columnas de texto a formato numerico
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

# Calculo el Z-Score para cada metrica elegida para que sean comparables
df_Z = pd.DataFrame(index=df_score.index)

# Por tanto, si tiene una valor mas alto dentro del universo,es mejor para estas metricas de rendimiento y ratios
for col in ["Sortino_n", "Calmar_n", "Rentabilidad_n"]:
    std = df_score[col].std()
    df_Z[col] = (df_score[col] - df_score[col].mean()) / std if std != 0 else 0

# Si tiene un valor mas bajo dentro del universo, es mejor para estas metricas de riesgo
for col in ["Max_DD_n", "Duracion_DD_n"]:
    std = df_score[col].std()
    df_Z[col] = (
        -1 * ((df_score[col] - df_score[col].mean()) / std) if std != 0 else 0
    )

# Defino los pesos otorgargos a la metricas elegidas
pesos = {
    "Sortino_n": 0.35, 
    "Calmar_n": 0.25,  
    "Max_DD_n": 0.15,  
    "Duracion_DD_n": 0.15,  
    "Rentabilidad_n": 0.10,  
}

# Calculo el score de robuste con los calculos previos y lo añado al df principal
metricas_distintos_capitales["Score Robustez"] = sum(
    df_Z[col] * peso for col, peso in pesos.items()
).round(4)

# Ordeno el df princiipal de forma descendente por el Score
metricas_distintos_capitales = metricas_distintos_capitales.sort_values(
    by="Score Robustez", ascending=False
).reset_index(drop=True)

print(f"\nMetricas del sistema trading de distints capitales del mejor modelo: {metricas_distintos_capitales.shape}")
