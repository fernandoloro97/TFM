import boto3
import pandas as pd
import requests
import time
import pytz
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from decimal import Decimal

# --- 1. CONFIGURACIÓN INICIAL Y CONEXIÓN ---
API_KEY = "PKAPICHWO6YCDQQP7J54WS5VSB"
API_SECRET = "7pfkxGpxU16pgz836kSt2QHvZKjVKkmzGEUxmgHagE4u"

dynamodb = boto3.resource('dynamodb')

def descargar_tabla_completa(nombre_tabla):
    tabla = dynamodb.Table(nombre_tabla)
    response = tabla.scan()
    data = response['Items']
    while 'LastEvaluatedKey' in response:
        response = tabla.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    return pd.DataFrame(data)

# --- 2. CARGA DE TABLAS DESDE DYNAMODB ---
print("Cargando tablas desde DynamoDB...")
sp500_actualizado = descargar_tabla_completa('historic_composition_sp500')
clean_changes_sp500 = descargar_tabla_completa('clean_changes_sp500')
precios_cierre_sesion_historico = descargar_tabla_completa('sesion_close_prices')
volumenes_sesion_historico = descargar_tabla_completa('sesion_volumes')


# Asegurar formatos de fecha
sp500_actualizado['Date'] = pd.to_datetime(sp500_actualizado['Date'])
clean_changes_sp500['Effective Date'] = pd.to_datetime(clean_changes_sp500['Effective Date'])
precios_cierre_sesion_historico['Date'] = pd.to_datetime(precios_cierre_sesion_historico['Date'])
volumenes_sesion_historico['Date'] = pd.to_datetime(volumenes_sesion_historico['Date'])


# --- 3. OBTENER LISTA DE TICKERS PARA HOY ---
# Fecha más reciente de composición
fecha_comp_max = sp500_actualizado['Date'].max()
tickers_string = sp500_actualizado.loc[sp500_actualizado['Date'] == fecha_comp_max, 'Ticker'].values[0]
lista_tickers = [t.strip() for t in tickers_string.split(',')]

# Mapeo de nombres especiales
mapeo_alpaca = {'BF-B': 'BF.B', 'BRK-B': 'BRK.B'}
lista_tickers = [mapeo_alpaca.get(item, item) for item in lista_tickers]

# Lógica de Deletions (margen +1 día)
hoy_simulado = datetime(2026, 5, 20) # Según tu ejemplo
todos_los_deletions = clean_changes_sp500[clean_changes_sp500['Action'] == 'Deletion']

for _, fila in todos_los_deletions.iterrows():
    ticker = fila['Ticker']
    fecha_limite = fila['Effective Date'] + timedelta(days=1)
    if fila['Effective Date'] <= hoy_simulado <= fecha_limite:
        if ticker not in lista_tickers:
            lista_tickers.append(ticker)

# --- 4. CONFIGURACIÓN DE DESCARGA ALPACA ---
fecha_str = "2026-05-20"
tz_ny = pytz.timezone("America/New_York")
apertura_ny = tz_ny.localize(datetime.strptime(f"{fecha_str} 09:30:00", "%Y-%m-%d %H:%M:%S"))
cierre_ny = tz_ny.localize(datetime.strptime(f"{fecha_str} 16:00:00", "%Y-%m-%d %H:%M:%S"))
START = apertura_ny.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
END = cierre_ny.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def descargar_ticker(ticker, start_date=None, end_date=None):
    s = start_date if start_date else START
    e = end_date if end_date else END
    
    # URL CORREGIDA - Fíjate en el "data." y en la ruta final
    url = "https://data.alpaca.markets/v2/stocks/bars"
    
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
    params = {"symbols": ticker, "timeframe": "1Min", "start": s, "end": e, "feed": "sip", "adjustment": "all"}
    
    all_bars = []
    page_token = None
    while True:
        if page_token: params["page_token"] = page_token
        r = requests.get(url, headers=headers, params=params, timeout=30)
        
        if r.status_code != 200: 
            print(f"Error en {ticker}: {r.status_code} - {r.text[:50]}")
            break
            
        try:
            data = r.json()
            bars = data.get("bars", {}).get(ticker, [])
            all_bars.extend(bars)
            page_token = data.get("next_page_token")
            if not page_token: break
        except Exception as e:
            print(f"Fallo JSON en {ticker}: {e}")
            break
            
    return all_bars


# --- 5. DESCARGA DEL DÍA ACTUAL ---
print(f"Descargando datos para {fecha_str}...")
diccionario_datos = {} # Cambiamos nombre a algo genérico ya que trae Close y Volume
for i, ticker in enumerate(lista_tickers, 1):
    print(f"[{i}/{len(lista_tickers)}] {ticker}", end="\r")
    bars = descargar_ticker(ticker)
    if bars:
        # CORRECCIÓN: Renombramos 'c' a 'Close' Y 'v' a 'Volume'
        df_t = pd.DataFrame(bars).rename(columns={'t': 'Timestamp', 'c': 'Close', 'v': 'Volume'})
        df_t['Timestamp'] = pd.to_datetime(df_t['Timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
        diccionario_datos[ticker] = df_t
    time.sleep(0.5)

# --- 6. CREAR DATAFRAME DE PRECIOS ---
lista_cols_precios = []
for ticker, df_t in diccionario_datos.items():
    temp = df_t[['Timestamp', 'Close']].set_index('Timestamp')
    temp.columns = [ticker]
    lista_cols_precios.append(temp)

precios_cierre_hoy = pd.concat(lista_cols_precios, axis=1).rename(columns={'BF.B': 'BF-B', 'BRK.B': 'BRK-B'})
precios_cierre_hoy.index.name = 'Date'
precios_cierre_hoy = precios_cierre_hoy.between_time('09:30', '16:00').reset_index()

# --- 7. CREAR DATAFRAME DE VOLÚMENES ---
lista_cols_volumes = []
for ticker, df_t in diccionario_datos.items():
    # Ahora 'Volume' sí existe en df_t
    temp = df_t[['Timestamp', 'Volume']].set_index('Timestamp')
    temp.columns = [ticker]
    lista_cols_volumes.append(temp)

volumenes_hoy = pd.concat(lista_cols_volumes, axis=1).rename(columns={'BF.B': 'BF-B', 'BRK.B': 'BRK-B'})
volumenes_hoy.index.name = 'Date'
volumenes_hoy = volumenes_hoy.between_time('09:30', '16:00').reset_index()



# --- 6. UNIFICACIÓN Y TICKERS NUEVOS ---
precios_cierre_sesion_actualizado = pd.concat([precios_cierre_sesion_historico, precios_cierre_hoy], axis=0, sort=False).reset_index(drop=True)
volumenes_sesion_actualizado = pd.concat([volumenes_sesion_historico, volumenes_hoy], axis=0, sort=False).reset_index(drop=True)


columnas_nuevas = set(precios_cierre_sesion_actualizado.columns) - set(precios_cierre_sesion_historico.columns)
if 'Date' in columnas_nuevas: columnas_nuevas.remove('Date')


diccionario_hist = {}
if columnas_nuevas:
    print(f"\n✨ Nuevos tickers detectados: {columnas_nuevas}. Bajando 35 días...")
    ref = pd.to_datetime("2026-05-20")
    S_HIST = (ref - timedelta(days=35)).strftime('%Y-%m-%dT00:00:00Z')
    E_HIST = (ref - timedelta(days=1)).strftime('%Y-%m-%dT23:59:59Z')
    
    for ticker in columnas_nuevas:
        bars = descargar_ticker(ticker, start_date=S_HIST, end_date=E_HIST)
        if bars:
            # Asegúrate de incluir 'v' (Volume) en el rename si tu API lo devuelve así
            df_h = pd.DataFrame(bars).rename(columns={'t': 'Timestamp', 'c': 'Close', 'v': 'Volume'})
            df_h['Timestamp'] = pd.to_datetime(df_h['Timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
            diccionario_hist[ticker] = df_h

    # --- Actualización de Precios ---
    precios_cierre_sesion_actualizado.set_index('Date', inplace=True)
    
    # --- ACTUALIZACIÓN DE VOLUMEN (Nueva sección) ---
    volumenes_sesion_actualizado.set_index('Date', inplace=True)

    for ticker, df_h in diccionario_hist.items():
        # Procesar Precios
        df_p_ready = df_h.set_index('Timestamp')[['Close']].between_time('09:30', '16:00')
        df_p_ready.columns = [ticker]
        precios_cierre_sesion_actualizado.update(df_p_ready)
        
        # Procesar Volumen
        df_v_ready = df_h.set_index('Timestamp')[['Volume']].between_time('09:30', '16:00')
        df_v_ready.columns = [ticker]
        volumenes_sesion_actualizado.update(df_v_ready)

    # Devolver ambos al estado original con la columna Date
    precios_cierre_sesion_actualizado.reset_index(inplace=True)
    volumenes_sesion_actualizado.reset_index(inplace=True)


# --- 7.1 ACTUALIZACIÓN DE PRECIOS CIERRE ---
tabla_dest = dynamodb.Table('sesion_close_prices')

# A. Borrar día antiguo
fecha_min = precios_cierre_sesion_actualizado['Date'].min()
filas_borrar = precios_cierre_sesion_actualizado[precios_cierre_sesion_actualizado['Date'].dt.date == fecha_min.date()]
print(f"🗑️ Borrando {fecha_min.date()} de DynamoDB...")
with tabla_dest.batch_writer() as batch:
    for _, fila in filas_borrar.iterrows():
        batch.delete_item(Key={'Date': str(fila['Date'])})

# B. Subir día nuevo (Filtrando NaNs y convirtiendo a Decimal)
print(f"📤 Subiendo {fecha_str}...")
datos_nuevos = precios_cierre_hoy.to_dict(orient='records')

with tabla_dest.batch_writer() as batch:
    for f in datos_nuevos:
        # Creamos el item final omitiendo NaNs y convirtiendo floats a Decimal
        item = {}
        for k, v in f.items():
            if pd.notna(v):  # Filtro para omitir NaNs
                if k == 'Date':
                    item[k] = str(v)
                elif isinstance(v, (float, int)):
                    item[k] = Decimal(str(v))
                else:
                    item[k] = v
        
        # Solo subimos si tiene la fecha y al menos un precio
        if len(item) > 1:
            batch.put_item(Item=item)

# C. Inyectar columnas nuevas en historial nube (Filtrando NaNs y usando Decimal)
if diccionario_hist:
    print("💉 Inyectando historial de nuevos tickers en DynamoDB...")
    for ticker, df_h in diccionario_hist.items():
        # Procesamos el historial descargado
        df_h_s = df_h.set_index('Timestamp').between_time('09:30', '16:00').reset_index()
        for _, fila in df_h_s.iterrows():
            valor = fila['Close']
            
            # Solo actualizamos en DynamoDB si el valor NO es NaN
            if pd.notna(valor):
                try:
                    tabla_dest.update_item(
                        Key={'Date': str(fila['Timestamp'])},
                        UpdateExpression="SET #tk = :val",
                        ExpressionAttributeNames={"#tk": ticker},
                        ExpressionAttributeValues={":val": Decimal(str(valor))}
                    )
                except ClientError as e:
                    # Si la fecha no existe en la tabla (raro, pero posible), se salta
                    print(f"No se pudo actualizar {ticker} para la fecha {fila['Timestamp']}")
                    
                    
# --- 7.2 ACTUALIZACIÓN DE SESION_VOLUMES ---
tabla_vol = dynamodb.Table('sesion_volumes')

# A. Borrar día antiguo en volumen (usando la misma fecha_min de precios)
print(f"🗑️ Borrando {fecha_min.date()} de DynamoDB (Volúmenes)...")
with tabla_vol.batch_writer() as batch:
    for _, fila in filas_borrar.iterrows(): # Usamos filas_borrar ya definida
        batch.delete_item(Key={'Date': str(fila['Date'])})

# B. Subir día nuevo (volumen_hoy)
print(f"📤 Subiendo volúmenes de {fecha_str}...")
# Asegúrate que 'volumen_hoy' tenga la columna 'Date' tras el reset_index anterior
datos_vol_nuevos = volumenes_hoy.to_dict(orient='records') 

with tabla_vol.batch_writer() as batch:
    for f in datos_vol_nuevos:
        item = {}
        for k, v in f.items():
            if pd.notna(v):
                if k == 'Date':
                    item[k] = str(v)
                elif isinstance(v, (float, int)):
                    # El volumen suele ser entero, pero Decimal(str(v)) cubre ambos
                    item[k] = Decimal(str(v))
                else:
                    item[k] = v
        
        if len(item) > 1:
            batch.put_item(Item=item)

# C. Inyectar historial de volumen para nuevos tickers
if diccionario_hist:
    print("💉 Inyectando historial de VOLUMEN de nuevos tickers en DynamoDB...")
    for ticker, df_h in diccionario_hist.items():
        # Filtramos igual que en precios
        df_h_s = df_h.set_index('Timestamp').between_time('09:30', '16:00').reset_index()
        for _, fila in df_h_s.iterrows():
            valor_vol = fila['Volume'] # <--- Usamos columna Volume
            
            if pd.notna(valor_vol):
                try:
                    tabla_vol.update_item(
                        Key={'Date': str(fila['Timestamp'])},
                        UpdateExpression="SET #tk = :val",
                        ExpressionAttributeNames={"#tk": ticker},
                        ExpressionAttributeValues={":val": Decimal(str(valor_vol))}
                    )
                except ClientError as e:
                    pass # Mismo criterio que en precios


print("Todo listo. Tablas sincronizadas y NaNs filtrados.")

if __name__ == "__main__":
    # Aquí puedes llamar a la función principal que contiene todo tu código
    # Si metiste todo el código en una función llamada 'main' o 'handler'
    print("Iniciando ejecución desde GitHub Actions...")