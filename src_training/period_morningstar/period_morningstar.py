import yfinance as yf
import pandas as pd
import time
import json
import boto3

# Instancio el dyanmodb
dynamodb = boto3.resource("dynamodb")
tabla_sp500_in_out = dynamodb.Table("sp500_in_out")

# Descargo la tabla
items = []
response = tabla_sp500_in_out.scan()
items.extend(response.get("Items", []))

# Manejo varias ventanas de la tabla
while "LastEvaluatedKey" in response:
    response = tabla_sp500_in_out.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
    items.extend(response.get("Items", []))

# Transformo la tabla a df
sp500_in_out = pd.DataFrame(items)

# Aseguro formato datime a las fechas
sp500_in_out["Date Added"] = pd.to_datetime(
    sp500_in_out["Date Added"], errors="coerce"
)
sp500_in_out["Date Removed"] = pd.to_datetime(
    sp500_in_out["Date Removed"], errors="coerce"
)

# Los tickers, company y duration aseguro el formato string
string_cols = ["Ticker", "Company Name", "Duration"]
for col in string_cols:
    if col in sp500_in_out.columns:
        sp500_in_out[col] = sp500_in_out[col].astype(str)
        
        
        
# Me quedo con Ticker y company de sp500_in_out
ticker_name =  sp500_in_out.iloc[:,:2].copy()
# Para leer sectores de Yahoo Finance, necesito adaptar el nombre de 2 tickers
ticker_name["Ticker"] = ticker_name["Ticker"].replace({
    "BF.B": "BF-B",
    "BRK.B": "BRK-B"
})

# Paso a lista los ticker para iterar sobre ellos
tickers = ticker_name["Ticker"].tolist()

# Mediante la libreria de Yahoo Finance, descargo tickers, nombre, sector e industria de Morningstar
datos_lista = []

for t in tickers:
    try:
        info = yf.Ticker(t).info
        
        datos_lista.append({
            "Ticker": t,
            "Yahoo Company Name": info.get("longName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry")
        })
    except Exception as e:
        print(f"Error con {t}: {e}")
        datos_lista.append({"Ticker": t, "Yahoo Company Name": None, "Sector": None, "Industry": None})

# Transformo los resultados a df
sectores_empresa = pd.DataFrame(datos_lista)

# Uno tickers y nombres con los sectores e industrias de Morningstar obtenidos
df_sectores = pd.merge(ticker_name, sectores_empresa, on="Ticker", how="left")

# Comparo nombres de empresas con los de yahoo para ver si efectivamente me traje el ticker correcto
sectores_yahoo_encontrados = df_sectores[~df_sectores.isna().any(axis=1)]

sectores_yahoo_encontrados['Yahoo Comparison'] = (sectores_yahoo_encontrados['Company Name'].str.split().str[0].str.lower() == 
                     sectores_yahoo_encontrados['Yahoo Company Name'].str.split().str[0].str.lower())


# Reviso los NaN de datos que no se encontrar en Yahoo Finance 
df_sectores_NA = df_sectores[df_sectores.isna().any(axis=1)]

# Procedo con la busqueda de tickers sin sectores
tickers_sin_sector = df_sectores_NA["Ticker"]

# Otra fuente bastan fiable para estos tickers faltantes que son historicos es Digrin
def get_digrin_data(ticker):
    url = f"https://www.digrin.com/stocks/detail/{ticker}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        
        if r.status_code == 404 or soup.find("h2", string=lambda t: t and "404" in t):
            print(f"  {ticker}: página no encontrada")
            return None, None, None
        
        title = soup.find("h1")
        company_digrin = title.text.strip() if title else None
        company_digrin = company_digrin.replace(f"({ticker})", "").replace("Dividends", "").strip() if company_digrin else None

        sector, industry = None, None
        for a in soup.find_all("a", href=True):
            if f"/stocks/list/sector/" in a["href"] and a["href"] != "/stocks/list/sector/":
                sector = a.text.strip()
            if f"/stocks/list/industry/" in a["href"] and a["href"] != "/stocks/list/industry/":
                industry = a.text.strip()
        
        return company_digrin, sector, industry
    
    except Exception as e:
        print(f"  Error {ticker}: {e}")
        return None, None, None

# Inserto el nombre enonctrado en Digrin
df_sectores.insert(3, "Digrin Company Name", None)

# Aplico la busqueda en Digrin de los tickers no encontados 
for ticker in tickers_sin_sector:
    print(f"Buscando {ticker}...")
    company_digrin, sector, industry = get_digrin_data(ticker)
    
    mask = df_sectores["Ticker"] == ticker
    df_sectores.loc[mask, "Digrin Company Name"] = company_digrin
    df_sectores.loc[mask, "Sector"] = sector
    df_sectores.loc[mask, "Industry"] = industry
    
    print(f"  → {company_digrin} | {sector} | {industry}")
    time.sleep(0.5)

# Miro los tickers no encontrados 
sectores_digrin_encontrados = df_sectores[df_sectores['Ticker'].isin(tickers_sin_sector)]

# Reviso si hay NaN despues de buscar en Digrin y si hay uno, es CXO
columnas_interes = ["Digrin Company Name", "Sector", "Industry"]
sectores_digrin_encontrados[sectores_digrin_encontrados[columnas_interes].isna().any(axis=1)]

# Añado sus datos manualmente, porque es dificil encontrarlo 
df_sectores.loc[142, ['Digrin Company Name','Sector', 'Industry']] = ['Concho Resources', 'Energy', 'Oil & Gas E&P']
df_sectores.drop(columns=['Yahoo Company Name', 'Digrin Company Name'], inplace=True)

# Reemplazo "Insurance Brokers" por "Insurance - Brokers" porque daba error y error por una tonteria
df_sectores['Industry'] = df_sectores['Industry'].replace('Insurance Brokers', 'Insurance - Brokers')







# Merge y reordenamiento en un solo bloque
df_merged = df_sectores.merge(
    df_ref[['Sector', 'Industry Group', 'Industry']], 
    on=['Sector', 'Industry'], 
    how='left'
)

# Mover Industry Group a la cuarta posición (índice 3)
col = df_merged.pop('Industry Group')
df_merged.insert(3, 'Industry Group', col)
