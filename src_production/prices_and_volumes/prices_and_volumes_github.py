import boto3
import pandas as pd
import requests
import time
import pytz
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from decimal import Decimal

# Guardo mis keys de Alpaca
API_KEY = "PKAPICHWO6YCDQQP7J54WS5VSB"
API_SECRET = "7pfkxGpxU16pgz836kSt2QHvZKjVKkmzGEUxmgHagE4u"

# Congifuro el dynamodb
dynamodb = boto3.resource('dynamodb')

# Descargo las tablas de dynamodb y las transformo a df
def descargar_tabla_completa(nombre_tabla):
    tabla = dynamodb.Table(nombre_tabla)
    response = tabla.scan()
    data = response['Items']
    while 'LastEvaluatedKey' in response:
        response = tabla.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    return pd.DataFrame(data)

# Cargo las tablas de interes y las guardo como df
sp500_actualizado = descargar_tabla_completa('historic_composition_sp500')
clean_changes_sp500 = descargar_tabla_completa('clean_changes_sp500')
precios_cierre_sesion_historico = descargar_tabla_completa('sesion_close_prices')
volumenes_sesion_historico = descargar_tabla_completa('sesion_volumes')

# Aseguro el fomato datetime a datos de fecha
sp500_actualizado['Date'] = pd.to_datetime(sp500_actualizado['Date'])
clean_changes_sp500['Effective Date'] = pd.to_datetime(clean_changes_sp500['Effective Date'])
precios_cierre_sesion_historico['Date'] = pd.to_datetime(precios_cierre_sesion_historico['Date'])
volumenes_sesion_historico['Date'] = pd.to_datetime(volumenes_sesion_historico['Date'])


# Obtengo los tickers de hoy
fecha_comp_max = sp500_actualizado['Date'].max()
tickers_string = sp500_actualizado.loc[sp500_actualizado['Date'] == fecha_comp_max, 'Ticker'].values[0]
lista_tickers = [t.strip() for t in tickers_string.split(',')]

# Cambios nombres de 2 tickers especiales para llamarlos desde Alpaca
mapeo_alpaca = {'BF-B': 'BF.B', 'BRK-B': 'BRK.B'}
lista_tickers = [mapeo_alpaca.get(item, item) for item in lista_tickers]

# Guardo la fecha de hoy para traer los precios y volumenes de hoy
hoy_simulado = datetime.now() 
# Filtro las empresas eliminadas del SP500
todos_los_deletions = clean_changes_sp500[clean_changes_sp500['Action'] == 'Deletion']

# Si hay una empresa de salida con fecha efectiva hoy, se mantiene para traer precios y volumnes para hoy y mañana
for _, fila in todos_los_deletions.iterrows():
    ticker = fila['Ticker']
    fecha_limite = fila['Effective Date'] + timedelta(days=1)
    if fila['Effective Date'] <= hoy_simulado <= fecha_limite:
        if ticker not in lista_tickers:
            lista_tickers.append(ticker)

# Confirguor el rango de fecha y hora para descargar precios y volumnes
fecha_str = hoy_simulado.strftime("%Y-%m-%d")
# Configuro hora de Nueva York para los datos
tz_ny = pytz.timezone("America/New_York")
apertura_ny = tz_ny.localize(datetime.strptime(f"{fecha_str} 09:30:00", "%Y-%m-%d %H:%M:%S"))
cierre_ny = tz_ny.localize(datetime.strptime(f"{fecha_str} 16:00:00", "%Y-%m-%d %H:%M:%S"))
START = apertura_ny.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
END = cierre_ny.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# Descargo ticker por ticker los datos OLHCV
def descargar_ticker(ticker, start_date=None, end_date=None):
    s = start_date if start_date else START
    e = end_date if end_date else END
    
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


# Aplicado la funcion de descarga y los guardo en un diccionario
print(f"Descargando datos para {fecha_str}")
diccionario_datos = {} 
for i, ticker in enumerate(lista_tickers, 1):
    print(f"[{i}/{len(lista_tickers)}] {ticker}", end="\r")
    bars = descargar_ticker(ticker)
    if bars:
        df_t = pd.DataFrame(bars).rename(columns={'t': 'Timestamp', 'c': 'Close', 'v': 'Volume'})
        df_t['Timestamp'] = pd.to_datetime(df_t['Timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
        diccionario_datos[ticker] = df_t
    time.sleep(0.5)

# Creo el df de precios de cierre
lista_cols_precios = []
for ticker, df_t in diccionario_datos.items():
    temp = df_t[['Timestamp', 'Close']].set_index('Timestamp')
    temp.columns = [ticker]
    lista_cols_precios.append(temp)

# Vuelvo a cambiar los tickers que antes se modifico solo para traer precios y volumnes
precios_cierre_hoy = pd.concat(lista_cols_precios, axis=1).rename(columns={'BF.B': 'BF-B', 'BRK.B': 'BRK-B'})
precios_cierre_hoy.index.name = 'Date'
precios_cierre_hoy = precios_cierre_hoy.between_time('09:30', '16:00').reset_index()

# Creo el df de volumnes
lista_cols_volumes = []
for ticker, df_t in diccionario_datos.items():
    # Ahora 'Volume' sí existe en df_t
    temp = df_t[['Timestamp', 'Volume']].set_index('Timestamp')
    temp.columns = [ticker]
    lista_cols_volumes.append(temp)

# Vuelvo a cambiar los tickers que antes se modifico solo para traer precios y volumnes
volumenes_hoy = pd.concat(lista_cols_volumes, axis=1).rename(columns={'BF.B': 'BF-B', 'BRK.B': 'BRK-B'})
volumenes_hoy.index.name = 'Date'
volumenes_hoy = volumenes_hoy.between_time('09:30', '16:00').reset_index()

# Añado estos nuevos precios y volumnes al historico
precios_cierre_sesion_actualizado = pd.concat([precios_cierre_sesion_historico, precios_cierre_hoy], axis=0, sort=False).reset_index(drop=True)
volumenes_sesion_actualizado = pd.concat([volumenes_sesion_historico, volumenes_hoy], axis=0, sort=False).reset_index(drop=True)

# Me quedo con los nuevos tickers, porque descargare mas datos de este
columnas_nuevas = set(precios_cierre_sesion_actualizado.columns) - set(precios_cierre_sesion_historico.columns)
if 'Date' in columnas_nuevas: columnas_nuevas.remove('Date')

# Descargo precios de cierre y volumnes para un historicos de 35 dias antes para los nuevos tickers
diccionario_hist = {}
if columnas_nuevas:
    print(f"\nNuevos tickers detectados: {columnas_nuevas}. Bajando historico de 35 días")
    ref = hoy_simulado
    S_HIST = (ref - timedelta(days=35)).strftime('%Y-%m-%dT00:00:00Z')
    E_HIST = (ref - timedelta(days=1)).strftime('%Y-%m-%dT23:59:59Z')
    
    for ticker in columnas_nuevas:
        bars = descargar_ticker(ticker, start_date=S_HIST, end_date=E_HIST)
        if bars:
            df_h = pd.DataFrame(bars).rename(columns={'t': 'Timestamp', 'c': 'Close', 'v': 'Volume'})
            df_h['Timestamp'] = pd.to_datetime(df_h['Timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
            diccionario_hist[ticker] = df_h

    # Coloco el Date como indice para precios cierre
    precios_cierre_sesion_actualizado.set_index('Date', inplace=True)
    # # Coloco el Date como indice para volumnes
    volumenes_sesion_actualizado.set_index('Date', inplace=True)

    # Actualizo el df de precios cierre y volumnes con el historicos de los nuevos tickers
    for ticker, df_h in diccionario_hist.items():
        # Proceso precios cierre
        df_p_ready = df_h.set_index('Timestamp')[['Close']].between_time('09:30', '16:00')
        df_p_ready.columns = [ticker]
        precios_cierre_sesion_actualizado.update(df_p_ready)
        
        # Proceso volumenes
        df_v_ready = df_h.set_index('Timestamp')[['Volume']].between_time('09:30', '16:00')
        df_v_ready.columns = [ticker]
        volumenes_sesion_actualizado.update(df_v_ready)

    # Devuelvo el Date como nueva columna para precios cierre
    precios_cierre_sesion_actualizado.reset_index(inplace=True)
    # Devuelvo el Date como nueva columna para volumenes
    volumenes_sesion_actualizado.reset_index(inplace=True)


# Actualizo la tabla de precios cierre de dynamodb
tabla_dest = dynamodb.Table('sesion_close_prices')

# Borro los precios cierre de la fecha mas antigua, asi no sobrecargo la tabla de dynamodb
fecha_min = precios_cierre_sesion_actualizado['Date'].min()
filas_borrar = precios_cierre_sesion_actualizado[precios_cierre_sesion_actualizado['Date'].dt.date == fecha_min.date()]
print(f"Borrando precios cierre de {fecha_min.date()} en dynamodb")
with tabla_dest.batch_writer() as batch:
    for _, fila in filas_borrar.iterrows():
        batch.delete_item(Key={'Date': str(fila['Date'])})

# Subo los precios cierre de hoy, filtrando NaNs y convirtiendo a decimal
print(f"Subiendo precios cierre de {fecha_str}")
datos_nuevos = precios_cierre_hoy.to_dict(orient='records')

with tabla_dest.batch_writer() as batch:
    for f in datos_nuevos:
        item = {}
        for k, v in f.items():
            if pd.notna(v): 
                if k == 'Date':
                    item[k] = str(v)
                elif isinstance(v, (float, int)):
                    item[k] = Decimal(str(v))
                else:
                    item[k] = v
        
        if len(item) > 1:
            batch.put_item(Item=item)

# Subo columnas nuevas de precios cierre en dynamodb, filtrando NaNs y usando decimal
if diccionario_hist:
    print("Inyectando historial de precios de cierre de nuevos tickers en Dynamodb")
    for ticker, df_h in diccionario_hist.items():
        df_h_s = df_h.set_index('Timestamp').between_time('09:30', '16:00').reset_index()
        for _, fila in df_h_s.iterrows():
            valor = fila['Close']
            
            if pd.notna(valor):
                try:
                    tabla_dest.update_item(
                        Key={'Date': str(fila['Timestamp'])},
                        UpdateExpression="SET #tk = :val",
                        ExpressionAttributeNames={"#tk": ticker},
                        ExpressionAttributeValues={":val": Decimal(str(valor))}
                    )
                except ClientError as e:
                    print(f"No se pudo actualizar {ticker} para la fecha {fila['Timestamp']}")
                    
                    
# Actualizo la tabla de volumenes de dynamodb
tabla_vol = dynamodb.Table('sesion_volumes')

# Borro los volumenes de la fecha mas antigua, asi no sobrecargo la tabla de dynamodb
print(f"Borrando volumenes de {fecha_min.date()} en dynamodb")
with tabla_vol.batch_writer() as batch:
    for _, fila in filas_borrar.iterrows(): 
        batch.delete_item(Key={'Date': str(fila['Date'])})

# Subo los volumenes de hoy, filtrando NaNs y convirtiendo a decimal
print(f"Subiendo volumenes de {fecha_str}")
datos_vol_nuevos = volumenes_hoy.to_dict(orient='records') 

with tabla_vol.batch_writer() as batch:
    for f in datos_vol_nuevos:
        item = {}
        for k, v in f.items():
            if pd.notna(v):
                if k == 'Date':
                    item[k] = str(v)
                elif isinstance(v, (float, int)):
                    item[k] = Decimal(str(v))
                else:
                    item[k] = v
        
        if len(item) > 1:
            batch.put_item(Item=item)

# Subo columnas nuevas de volumenes en dynamodb, filtrando NaNs y usando decimal
if diccionario_hist:
    print("Inyectando historial de volumenes de nuevos tickers en Dynamodb")
    for ticker, df_h in diccionario_hist.items():
        df_h_s = df_h.set_index('Timestamp').between_time('09:30', '16:00').reset_index()
        for _, fila in df_h_s.iterrows():
            valor_vol = fila['Volume'] 
            
            if pd.notna(valor_vol):
                try:
                    tabla_vol.update_item(
                        Key={'Date': str(fila['Timestamp'])},
                        UpdateExpression="SET #tk = :val",
                        ExpressionAttributeNames={"#tk": ticker},
                        ExpressionAttributeValues={":val": Decimal(str(valor_vol))}
                    )
                except ClientError as e:
                    pass 

print("Todo listo. Tablas sincronizadas y NaNs filtrados")

if __name__ == "__main__":
    # Ejecuto en Github Actions porque tarda mucho en Lambda
    print("Iniciando ejecución desde GitHub Actions.")