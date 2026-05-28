import yfinance as yf
import pandas as pd
import time
import json

# DESCARGAREL SP500_HISTORICO

# Reemplaza los valores dentro de la columna 'Ticker'
ticker_name =  sp500_historico.iloc[:,:2].copy()
ticker_name["Ticker"] = ticker_name["Ticker"].replace({
    "BF.B": "BF-B",
    "BRK.B": "BRK-B"
})

tickers = ticker_name["Ticker"].tolist()

datos_lista = []

for t in tickers:
    try:
        # Llamamos a .info UNA sola vez y lo guardamos en una variable
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

# Creamos el DataFrame desde la lista de diccionarios
sectores_empresa = pd.DataFrame(datos_lista)

# Unimos con tu DataFrame original (ticker_name) usando Ticker como llave
df_sectores = pd.merge(ticker_name, sectores_empresa, on="Ticker", how="left")

sectores_yahoo_encontrados = df_sectores[~df_sectores.isna().any(axis=1)]

sectores_yahoo_encontrados['Yahoo Comparison'] = (sectores_yahoo_encontrados['Company Name'].str.split().str[0].str.lower() == 
                     sectores_yahoo_encontrados['Yahoo Company Name'].str.split().str[0].str.lower())


# Muestra solo las filas donde falte algún dato (Ticker, Name, Sector o Industry)
df_sectores_NA = df_sectores[df_sectores.isna().any(axis=1)]

tickers_sin_sector = df_sectores_NA["Ticker"]

def get_digrin_data(ticker):
    url = f"https://www.digrin.com/stocks/detail/{ticker}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Si la página no existe, Digrin muestra un h2 con "404"
        if r.status_code == 404 or soup.find("h2", string=lambda t: t and "404" in t):
            print(f"  {ticker}: página no encontrada")
            return None, None, None
        
        # Nombre de la empresa
        title = soup.find("h1")
        company_digrin = title.text.strip() if title else None
        company_digrin = company_digrin.replace(f"({ticker})", "").replace("Dividends", "").strip() if company_digrin else None

        sector, industry = None, None
        for a in soup.find_all("a", href=True):
            # Excluir links del menú de navegación usando la URL completa
            if f"/stocks/list/sector/" in a["href"] and a["href"] != "/stocks/list/sector/":
                sector = a.text.strip()
            if f"/stocks/list/industry/" in a["href"] and a["href"] != "/stocks/list/industry/":
                industry = a.text.strip()
        
        return company_digrin, sector, industry
    
    except Exception as e:
        print(f"  Error {ticker}: {e}")
        return None, None, None

# Insertar columna "Digrin Company Name" en la tercera posición
df_sectores.insert(3, "Digrin Company Name", None)

for ticker in tickers_sin_sector:
    print(f"Buscando {ticker}...")
    company_digrin, sector, industry = get_digrin_data(ticker)
    
    mask = df_sectores["Ticker"] == ticker
    df_sectores.loc[mask, "Digrin Company Name"] = company_digrin
    df_sectores.loc[mask, "Sector"] = sector
    df_sectores.loc[mask, "Industry"] = industry
    
    print(f"  → {company_digrin} | {sector} | {industry}")
    time.sleep(0.5)

sectores_digrin_encontrados = df_sectores[df_sectores['Ticker'].isin(tickers_sin_sector)]


# 2. Filtramos para ver solo las filas donde alguna de las 'columnas_interes' es NA
columnas_interes = ["Digrin Company Name", "Sector", "Industry"]
sectores_digrin_encontrados[sectores_digrin_encontrados[columnas_interes].isna().any(axis=1)]

df_sectores.loc[142, ['Digrin Company Name','Sector', 'Industry']] = ['Concho Resources', 'Energy', 'Oil & Gas E&P']

df_sectores.drop(columns=['Yahoo Company Name', 'Digrin Company Name'], inplace=True)

# Reemplazar "Insurance Brokers" por "Insurance - Brokers"
df_sectores['Industry'] = df_sectores['Industry'].replace('Insurance Brokers', 'Insurance - Brokers')

# 1. Cargar el JSON (Asegúrate de que el archivo tenga la nueva estructura de objetos)
with open('morningstar_2025.json', 'r', encoding='utf-8') as f:
    classification = json.load(f)

# 2. Listas para recolectar los datos
data = []

for sector, groups in classification.items():
    for group, inds in groups.items():
        # 'inds' ahora es un diccionario: {"NombreIndustria": {"description": "..."}}
        for industry_name, details in inds.items():
            data.append({
                'Sector': sector,
                'Industry Group': group,
                'Industry': industry_name,
            })

# 3. Crear el DataFrame
df_ref = pd.DataFrame(data)

# Merge y reordenamiento en un solo bloque
df_merged = df_sectores.merge(
    df_ref[['Sector', 'Industry Group', 'Industry']], 
    on=['Sector', 'Industry'], 
    how='left'
)

# Mover Industry Group a la cuarta posición (índice 3)
col = df_merged.pop('Industry Group')
df_merged.insert(3, 'Industry Group', col)
