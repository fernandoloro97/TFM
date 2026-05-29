import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from urllib.parse import urljoin
import re
from dateutil import parser
import boto3


# Inicializado el dynamodb
dynamodb = boto3.resource("dynamodb")
tabla_sp500_actual = dynamodb.Table("sp500_actual")
tabla_sp500_in_out = dynamodb.Table("sp500_in_out")
tabla_sp500_period_historic = dynamodb.Table("sp500_period_historic")

# Descargo la tabla
items = []
response = tabla_sp500_actual.scan()
items.extend(response.get("Items", []))

# Itero para distintas paginas de la tabla
while "LastEvaluatedKey" in response:
    response = tabla_sp500_actual.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
    items.extend(response.get("Items", []))

# Lo guardo en un un df
sp500_actual = pd.DataFrame(items)

# Limpios las variables de la tabla: string, numero y datetime
sp500_actual["CIK"] = pd.to_numeric(sp500_actual["CIK"], errors="coerce")

# Convertierto a datetime
sp500_actual["Date added"] = pd.to_datetime(
    sp500_actual["Date added"], errors="coerce"
)

# Convierto a string
string_columns = [
    "Ticker",
    "Company Name",
    "GICS Sector",
    "GICS Sub-Industry",
    "Headquarters Location",
    "Founded",
]
for col in string_columns:
    if col in sp500_actual.columns:
        sp500_actual[col] = sp500_actual[col].astype(str)


# URL de SP GLOBAL PRESS RELEASE
BASE_URL = "https://press.spglobal.com/index.php?s=2429&l=100&year={year}&keywords=Join"
ROOT = "https://press.spglobal.com/"

#
# Extraigo los cambios respecto de periodo
def get_relevant_news(year):
    url = BASE_URL.format(year=year)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    records = []

    for item in soup.select(".wd_item"):
        title_tag = item.select_one(".wd_title a")
        date_tag = item.select_one(".wd_date")

        if not title_tag or not date_tag:
            continue

        title = title_tag.get_text(strip=True)
        publish_date = date_tag.get_text(strip=True)
        href = title_tag["href"]

        if "Set to Join S&P 500" in title or "Set to S&P 500" in title:
            full_url = urljoin(ROOT, href)
            records.append({
                "Publish Date": publish_date,
                "Title": title,
                "URL": full_url
            })

    return records

# Leo el nombre de las compañias citadas en los anunncios
def extract_company(title):
    match = re.match(r"(.+?)\s+Set to Join", title)
    if match:
        return match.group(1).strip()
    return None

# Orquestador de las funciones anteriores, obtengo el df de cambios del SP500
def collect_news(start_year=2020, end_year=2026):
    all_records = []

    for year in range(start_year, end_year + 1):
        all_records.extend(get_relevant_news(year))

    df_news = pd.DataFrame(all_records)
    df_news["Company"] = df_news["Title"].apply(extract_company)
    df_news["Publish Date"] = pd.to_datetime(df_news["Publish Date"])
    df_news = df_news.sort_values(["Company", "Publish Date"], ascending=[True, False])
    df_news = df_news.drop_duplicates(subset="Company", keep="first")

    return df_news.reset_index(drop=True)

# Extraigo las tablas con los datos de las modificaciones del SP500
def extract_sp500_table_raw(url):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    table = soup.find("table")
    if table is None:
        return pd.DataFrame()

    rows = []
    current_date = None

    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue

        texts = []
        for td in cells:
            span = td.find("span", class_="prnews_span")
            if span:
                texts.append(span.get_text(strip=True))
            else:
                texts.append(td.get_text(strip=True))

        if all(t == "" for t in texts):
            continue

        if "Effective Date" in texts[0]:
            continue

        if texts[0] != "":
            current_date = texts[0]

        if len(texts) == 7:
            _, index_name, action, company_name, ticker, sector = texts[0], texts[1], texts[2], texts[4], texts[5], texts[6]

        elif len(texts) == 6:
            _, index_name, action, company_name, ticker, sector = texts

        else:
            continue

        rows.append({
            "Effective Date": current_date,
            "Index Name": index_name,
            "Action": action,
            "Company Name": company_name,
            "Ticker": ticker,
            "GICS Sector": sector
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Filtrar solo S&P 500
    df = df[df["Index Name"].str.contains("S&P 500", case=False, na=False)]

    return df.reset_index(drop=True)


# Orquestador que lee las noticias y entra a cada uno para extraer las tablas con datos
def scrape_sp500_raw(start_year=2020, end_year=2026):
    df_news = collect_news(start_year, end_year)

    all_rows = []

    for _, row in df_news.iterrows():
        df_table = extract_sp500_table_raw(row["URL"])
        if df_table.empty:
            continue

        df_table["Publish Date"] = row["Publish Date"]
        df_table["Title"] = row["Title"]
        df_table["URL"] = row["URL"]

        all_rows.append(df_table)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)

# Ejecuto la busqueda y extraccion de cambios del periodo
sp500_cambio_brutos = scrape_sp500_raw(2020, 2026)

# Dado que defini una fecha donde tengo el SP500 del momento, solo me interesa de esa fecha hacia atras
sp500_cambio_brutos = sp500_cambio_brutos[sp500_cambio_brutos['Publish Date'] < '2026-02-18']

# Normalizo la fecha
def normalize_date(x):
    if pd.isna(x):
        return None
    
    x = str(x).strip()
    x = re.sub(r"\.", "", x)
    x = x.replace("Sept", "Sep")

    try:
        return parser.parse(x, dayfirst=False)
    except:
        return None

# Limpio las tablas sucias
def clean_sp500_dataframe(df_raw):

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Normalizo la fecha
    df["Effective Date"] = df["Effective Date"].apply(normalize_date)

   # Elimino las columnas no deseadas
    df = df.drop(columns=[c for c in ["Index Name", "Publish Date", "Title", "URL"] if c in df.columns])

    # Filtro desde 2021 en adelante, porque es mi periodo de analisis
    df = df[df["Effective Date"] >= pd.Timestamp("2021-01-01")]
    
    # Aseguro formato datetime
    df["Effective Date"] = pd.to_datetime(df["Effective Date"])

    # Convierto a solo fecha, sin horas
    df["Effective Date"] = df["Effective Date"].dt.date

    # Ordeno las fechas
    df = df.sort_values("Effective Date", ascending=False)

    return df.reset_index(drop=True)

# Aplico la limpieza a las tablas brutas
sp500_cambios = clean_sp500_dataframe(sp500_cambio_brutos)


# Dado que hay ticker que duraron menos que el periodo, me interesa verlos, buscado tickers repetidos
tickers_repetidos = sp500_cambios["Ticker"][sp500_cambios["Ticker"].duplicated()].unique()

# Ademas, necesito saber el orden de entrada y salida, porque el rango de vida en el indice seria distinto
resultados = []

for t in tickers_repetidos:
    df_t = sp500_cambios[sp500_cambios["Ticker"] == t].sort_values("Effective Date")

    acciones = df_t["Action"].tolist()

    if acciones == ["Addition", "Deletion"]:
        estado = True   
    elif acciones == ["Deletion", "Addition"]:
        estado = False 
    else:
        estado = None   

    resultados.append({
        "Ticker": t,
        "Fechas": pd.to_datetime(df_t["Effective Date"]).dt.strftime('%Y-%m-%d').tolist(),
        "Acciones": acciones,
        "Addition_before_Deletion": estado
    })

# Reviso y veo salieron despues de entrar, todo correcto
orden_entrada_salida = pd.DataFrame(resultados)


# Tengo que limpiar aun mas el df con los cambios del SP500
# En una fila hay 2 tickers unidos
fila_original_UA = sp500_cambios[sp500_cambios["Ticker"] == "UA/UAA"].iloc[0]

# Separo el ticker en sus respectivo ticker
fila_UA = fila_original_UA.copy()
fila_UA["Ticker"] = "UA"

fila_UAA = fila_original_UA.copy()
fila_UAA["Ticker"] = "UAA"

# Convierto a df para concaternarlos
fila_UA = fila_UA.to_frame().T
fila_UAA = fila_UAA.to_frame().T

# Elimino la fila original erronea
sp500_cambios = sp500_cambios[sp500_cambios["Ticker"] != "UA/UAA"]

# El ticker CDAY cambio de ticker a DAY, lo corrijo
sp500_cambios.loc[sp500_cambios['Ticker'] == 'CDAY', ['Company Name', 'Ticker', 'GICS Sector']] = ["Dayforce", "DAY", "Industrials"]
                                                                                         

# Falta las filas DISCA, DISCK y WBD y por eso las creo manualmente
fila_DISCA = pd.DataFrame([{
    "Effective Date": pd.Timestamp("2022-04-11"),
    "Action": "Deletion",
    "Company Name": "Discovery, Inc",
    "Ticker": "DISCA",
    "GICS Sector": "Communication Services"
}])

fila_DISCK = pd.DataFrame([{
    "Effective Date": pd.Timestamp("2022-04-11"),
    "Action": "Deletion",
    "Company Name": "Discovery, Inc.",
    "Ticker": "DISCK",
    "GICS Sector": "Communication Services"
}])

fila_WBD = pd.DataFrame([{
    "Effective Date": pd.Timestamp("2022-04-11"),
    "Action": "Addition",
    "Company Name": "Warner Bros. Discovery",
    "Ticker": "WBD",
    "GICS Sector": "Communication Services"
}])

# Concanteno todo
sp500_cambios = pd.concat(
    [sp500_cambios, fila_UA, fila_UAA, fila_DISCA, fila_DISCK, fila_WBD],
    ignore_index=True
)

# Asegur el formato datetime para fecha
sp500_cambios["Effective Date"] = pd.to_datetime(sp500_cambios["Effective Date"])

# Reordeno por fecha
sp500_cambios = sp500_cambios.sort_values("Effective Date", ascending=False).reset_index(drop=True)

# Solo me quedo con la fecha, elimino la hora
sp500_cambios["Effective Date"] = sp500_cambios["Effective Date"].dt.date


# Genero un ticker por fila con fecha de entada y salida original
rows = []

# Agrupo por ticker
for ticker, df_t in sp500_cambios.groupby("Ticker"):

    # Ordeno por fecha
    df_t = df_t.sort_values("Effective Date")

    # Inicializo valores
    date_added = pd.NaT
    date_removed = pd.NaT
    company = df_t["Company Name"].iloc[0]

    # Recorro los tickers
    for _, row in df_t.iterrows():
        if row["Action"] == "Addition":
            date_added = row["Effective Date"]
        elif row["Action"] == "Deletion":
            date_removed = row["Effective Date"]

    rows.append({
        "Company Name": company,
        "Ticker": ticker,
        "Date Added": date_added,
        "Date Removed": date_removed
    })

# Transformo a df todos los resultados añadidos
sp500_cambios_estructurado = pd.DataFrame(rows)

# Ajusto la entrada y salida de los tickers al periodo de analisis
inicio_periodo = pd.to_datetime("2021-01-01")
fin_periodo = pd.to_datetime("2025-12-31")

df_hist = sp500_cambios_estructurado.copy()

# Aseguro formato datetime de fechas
df_hist["Date Added"] = pd.to_datetime(df_hist["Date Added"], errors="coerce")
df_hist["Date Removed"] = pd.to_datetime(df_hist["Date Removed"], errors="coerce")

# No me interesa empresas que entraron despues del periodo
df_hist = df_hist[
    (df_hist["Date Added"].isna()) |
    (df_hist["Date Added"] <= fin_periodo)
]

# Tickers con salidas posterior al periodo, se pone NaN
df_hist.loc[
    df_hist["Date Removed"] > fin_periodo,
    "Date Removed"
] = pd.NaT

# Reviso los tickers que no entraron durante el periodo, sino antes
tickers_actuales = set(sp500_actual["Ticker"])
tickers_con_cambios = set(sp500_cambios_estructurado["Ticker"])
tickers_sin_cambios = tickers_actuales - tickers_con_cambios

df_sin_cambios = sp500_actual[
    sp500_actual["Ticker"].isin(tickers_sin_cambios)
][["Ticker", "Company Name"]].copy()

# Los tickers que estuvieron todo el periodo, no se ni su entrada ni su salida, pongo NaN
df_sin_cambios["Date Added"] = pd.NaT
df_sin_cambios["Date Removed"] = pd.NaT

# Concateno todo
sp500_historico_bruto = pd.concat([df_hist, df_sin_cambios], ignore_index=True)


# Guardo un copia por seguridad
sp500_historico = sp500_historico_bruto.copy()

# Relleno todos los NaN de entrada y salida a los extremos del periodo, por necesidad para hacer la composicion diaria despues
sp500_historico["Date Added"] = sp500_historico["Date Added"].fillna(pd.Timestamp("2021-01-01"))
sp500_historico["Date Removed"] = sp500_historico["Date Removed"].fillna(pd.Timestamp("2025-12-31"))

# Calculo la duracion en el SP500 dentro del periodo de analisis
sp500_historico["Duration"] = sp500_historico["Date Removed"] - sp500_historico["Date Added"]

# Ordeno segun nombre de la compañia
sp500_historico = sp500_historico.sort_values("Ticker").reset_index(drop=True)

sp500_historico

# Aseguro formato datetime para las fechas
sp500_historico['Date Added'] = pd.to_datetime(sp500_historico['Date Added'])
sp500_historico['Date Removed'] = pd.to_datetime(sp500_historico['Date Removed'])

# Defino el periodo de analisis
rango_fechas = pd.date_range(start='2021-01-01', end='2025-12-31', freq='D')

resultados = []

# Genero el universo de la composicion del SP500 diarios
for dia in rango_fechas:

    if dia == pd.Timestamp('2025-12-31'):
        mask = (sp500_historico['Date Added'] <= dia) & (sp500_historico['Date Removed'] >= dia)
    else:
        mask = (sp500_historico['Date Added'] <= dia) & (sp500_historico['Date Removed'] > dia)

    nombres = sp500_historico.loc[mask, 'Ticker'].unique().tolist()

    resultados.append({
        'Date': dia,
        'Ticker': ", ".join(nombres),
        'Total Companies': len(nombres)
    })

# Tranformo a df los resultados
sp500_diario = pd.DataFrame(resultados)


# Convierto el df a un diccionario de string
dicc_sp500_historico = sp500_historico.astype(str).to_dict(orient="records")

# Subo el diccionario a la tabla de dynamodb
with tabla_sp500_in_out.batch_writer() as batch:
    for row in dicc_sp500_historico:
        batch.put_item(Item=row)

print("Tabla_sp500_in_out subida")

# Trasnformo el df a formato aceptados por dynamodb
df_preparado = sp500_diario.copy()
df_preparado["Date"] = df_preparado["Date"].astype(str)
df_preparado["Ticker"] = df_preparado["Ticker"].astype(str)
df_preparado["Total Companies"] = df_preparado["Total Companies"].astype(int)



# Verifico si hay algun dato en la tabla
chequeo_tabla = tabla_sp500_period_historic.scan(Limit=1)

# Si hay datos, no subo nada
if chequeo_tabla.get('Items'):
    print("La tabla sp500_period_historic ya contiene datos. Se cancela la subida para evitar duplicados", flush=True)
else:
    print("La tabla está vacia. Iniciando la subida de datos", flush=True)
    
    # Convierto a diccionario
    dicc_sp500_diario = df_preparado.to_dict(orient="records")

    # Subo el diccionario a la tabla de dynamodb
    with tabla_sp500_period_historic.batch_writer() as batch:
        for row in dicc_sp500_diario:
            batch.put_item(Item=row)

    print("Tabla sp500_period_historic subida")