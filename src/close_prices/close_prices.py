import boto3
import pandas as pd
import requests
import time
import pytz
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# Asegurar formatos de fecha
sp500_actualizado['Date'] = pd.to_datetime(sp500_actualizado['Date'])
clean_changes_sp500['Effective Date'] = pd.to_datetime(clean_changes_sp500['Effective Date'])
precios_cierre_sesion_historico['Date'] = pd.to_datetime(precios_cierre_sesion_historico['Date'])

# --- 3. OBTENER LISTA DE TICKERS PARA HOY ---
# Fecha más reciente de composición
fecha_comp_max = sp500_actualizado['Date'].max()
tickers_string = sp500_actualizado.loc[sp500_actualizado['Date'] == fecha_comp_max, 'Ticker'].values[0]
lista_tickers = [t.strip() for t in tickers_string.split(',')]

# Mapeo de nombres especiales
mapeo_alpaca = {'BF-B': 'BF.B', 'BRK-B': 'BRK.B'}
lista_tickers = [mapeo_alpaca.get(item, item) for item in lista_tickers]

# Lógica de Deletions (margen +1 día)
hoy_simulado = datetime(2026, 5, 4) # Según tu ejemplo
todos_los_deletions = clean_changes_sp500[clean_changes_sp500['Action'] == 'Deletion']

for _, fila in todos_los_deletions.iterrows():
    ticker = fila['Ticker']
    fecha_limite = fila['Effective Date'] + timedelta(days=1)
    if fila['Effective Date'] <= hoy_simulado <= fecha_limite:
        if ticker not in lista_tickers:
            lista_tickers.append(ticker)

# --- 4. CONFIGURACIÓN DE DESCARGA ALPACA ---
fecha_str = "2026-05-04"
tz_ny = pytz.timezone("America/New_York")
apertura_ny = tz_ny.localize(datetime.strptime(f"{fecha_str} 09:30:00", "%Y-%m-%d %H:%M:%S"))
cierre_ny = tz_ny.localize(datetime.strptime(f"{fecha_str} 16:00:00", "%Y-%m-%d %H:%M:%S"))
START = apertura_ny.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
END = cierre_ny.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# --- CONFIGURACIÓN DE SESIÓN ROBUSTA PARA AWS ---
session = requests.Session()
# Esto configura reintentos automáticos a nivel de red (más eficiente que un bucle manual)
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Headers para parecer un navegador real
HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def descargar_ticker(ticker, start_date=None, end_date=None):
    s = start_date if start_date else START
    e = end_date if end_date else END
    
    # ESTA ES LA URL CORRECTA PARA DATOS
    url = "https://alpaca.markets"
    
    params = {
        "symbols": ticker,
        "timeframe": "1Min",
        "start": s,
        "end": e,
        "limit": 10000,
        "feed": "sip", 
        "adjustment": "all",
    }

    all_bars = []
    page_token = None

    while True:
        if page_token: params["page_token"] = page_token
        try:
            r = session.get(url, headers=HEADERS, params=params, timeout=15)
            
            # Si Alpaca responde algo que no es 200, no intentamos el .json()
            if r.status_code != 200:
                print(f"⚠️ Error {r.status_code} en {ticker}: {r.text[:50]}")
                break

            data = r.json()
            bars = data.get("bars", {}).get(ticker, [])
            all_bars.extend(bars)

            page_token = data.get("next_page_token")
            if not page_token: break
        except Exception as err:
            # Aquí es donde te saltaba el "Expecting value"
            print(f"❌ Fallo en {ticker}: {err}")
            break
            
    return all_bars



# def descargar_ticker222(ticker, start_date=None, end_date=None):
#     s = start_date if start_date else START
#     e = end_date if end_date else END
#     url = "https://alpaca.markets"
#     headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
#     params = {"symbols": ticker, "timeframe": "1Min", "start": s, "end": e, "feed": "sip", "adjustment": "all"}
    
#     all_bars = []
#     page_token = None
    
#     while True:
#         if page_token: params["page_token"] = page_token
        
#         # --- Lógica de Reintentos ---
#         r = None
#         for intento in range(1, 4): # Intentará hasta 3 veces
#             try:
#                 r = requests.get(url, headers=headers, params=params, timeout=30)
                
#                 if r.status_code == 200:
#                     break # Éxito, salimos del bucle de reintentos
                
#                 elif r.status_code == 429: # Rate Limit (vas muy rápido)
#                     print(f"Rate limit en {ticker}. Reintento {intento}/3... esperando 5s")
#                     time.sleep(1.5)
                
#                 else:
#                     print(f"Error {r.status_code} en {ticker}. Reintento {intento}/3...")
#                     time.sleep(1)
                    
#             except Exception as e:
#                 print(f"Error de conexión en {ticker}: {e}. Reintento {intento}/3...")
#                 time.sleep(1)
        
#         # Si después de 3 intentos no funcionó, devolvemos lo que tengamos
#         if r is None or r.status_code != 200:
#             print(f"Imposible descargar {ticker} tras 3 intentos.")
#             return all_bars

#         # --- Decodificación segura ---
#         try:
#             data = r.json()
#             bars = data.get("bars", {}).get(ticker, [])
#             all_bars.extend(bars)
            
#             page_token = data.get("next_page_token")
#             if not page_token: 
#                 break
#         except Exception:
#             print(f"Error de formato JSON en {ticker}. Respuesta: {r.text[:100]}")
#             return all_bars
            
#     return all_bars


# --- 5. DESCARGA DEL DÍA ACTUAL ---
print(f"Descargando precios para {fecha_str}...")
diccionario_precios = {}
for i, ticker in enumerate(lista_tickers, 1):
    print(f"[{i}/{len(lista_tickers)}] {ticker}", end="\r")
    bars = descargar_ticker(ticker)
    if bars:
        df_t = pd.DataFrame(bars).rename(columns={'t': 'Timestamp', 'c': 'Close'})
        df_t['Timestamp'] = pd.to_datetime(df_t['Timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
        diccionario_precios[ticker] = df_t
    time.sleep(0.4)

# Crear precios_cierre_hoy
lista_cols = []
for ticker, df_t in diccionario_precios.items():
    temp = df_t[['Timestamp', 'Close']].set_index('Timestamp')
    temp.columns = [ticker]
    lista_cols.append(temp)

precios_cierre_hoy = pd.concat(lista_cols, axis=1).rename(columns={'BF.B': 'BF-B', 'BRK.B': 'BRK-B'})
precios_cierre_hoy.index.name = 'Date'
precios_cierre_hoy = precios_cierre_hoy.between_time('09:30', '16:00').reset_index()

# --- 6. UNIFICACIÓN Y TICKERS NUEVOS ---
precios_cierre_sesion_actualizado = pd.concat([precios_cierre_sesion_historico, precios_cierre_hoy], axis=0, sort=False).reset_index(drop=True)

columnas_nuevas = set(precios_cierre_sesion_actualizado.columns) - set(precios_cierre_sesion_historico.columns)
if 'Date' in columnas_nuevas: columnas_nuevas.remove('Date')

diccionario_hist = {}
if columnas_nuevas:
    print(f"\n✨ Nuevos tickers detectados: {columnas_nuevas}. Bajando 21 días...")
    ref = pd.to_datetime("2026-05-05")
    S_HIST = (ref - timedelta(days=21)).strftime('%Y-%m-%dT00:00:00Z')
    E_HIST = (ref - timedelta(days=1)).strftime('%Y-%m-%dT23:59:59Z')
    
    for ticker in columnas_nuevas:
        bars = descargar_ticker(ticker, start_date=S_HIST, end_date=E_HIST)
        if bars:
            df_h = pd.DataFrame(bars).rename(columns={'t': 'Timestamp', 'c': 'Close'})
            df_h['Timestamp'] = pd.to_datetime(df_h['Timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
            diccionario_hist[ticker] = df_h

    # Update local para rellenar NaNs
    precios_cierre_sesion_actualizado.set_index('Date', inplace=True)
    for ticker, df_h in diccionario_hist.items():
        df_h_ready = df_h.set_index('Timestamp')[['Close']].between_time('09:30', '16:00')
        df_h_ready.columns = [ticker]
        precios_cierre_sesion_actualizado.update(df_h_ready)
    precios_cierre_sesion_actualizado.reset_index(inplace=True)

# --- 7. ACTUALIZACIÓN DE DYNAMODB ---
tabla_dest = dynamodb.Table('precios_cierre_sesion')

# A. Borrar día antiguo
fecha_min = precios_cierre_sesion_actualizado['Date'].min()
filas_borrar = precios_cierre_sesion_actualizado[precios_cierre_sesion_actualizado['Date'].dt.date == fecha_min.date()]
print(f"Borrando {fecha_min.date()} de DynamoDB...")
with tabla_dest.batch_writer() as batch:
    for _, fila in filas_borrar.iterrows():
        batch.delete_item(Key={'Date': str(fila['Date'])})

# B. Subir día nuevo
print(f"Subiendo {fecha_str}...")
datos_nuevos = precios_cierre_hoy.to_dict(orient='records')
with tabla_dest.batch_writer() as batch:
    for f in datos_nuevos:
        item = {k: v for k, v in f.items() if pd.notna(v)}
        item['Date'] = str(item['Date'])
        batch.put_item(Item=item)

# C. Inyectar columnas nuevas en historial nube
if diccionario_hist:
    print("💉 Inyectando historial de nuevos tickers en DynamoDB...")
    for ticker, df_h in diccionario_hist.items():
        df_h_s = df_h.set_index('Timestamp').between_time('09:30', '16:00').reset_index()
        for _, fila in df_h_s.iterrows():
            tabla_dest.update_item(
                Key={'Date': str(fila['Timestamp'])},
                UpdateExpression="SET #tk = :val",
                ExpressionAttributeNames={"#tk": ticker},
                ExpressionAttributeValues={":val": fila['Close']}
            )

print("Todo listo. Tablas sincronizadas.")
