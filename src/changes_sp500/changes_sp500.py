import boto3
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from dateutil import parser


BASE_URL = "https://press.spglobal.com/index.php?s=2429&l=100&year={year}&keywords=Join"
ROOT = "https://press.spglobal.com/"


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

        #if "Set to Join S&P 500" in title:
        if "Set to Join S&P 500" in title or "Set to S&P 500" in title:
            full_url = urljoin(ROOT, href)
            records.append({
                "Publish Date": publish_date,
                "Title": title,
                "URL": full_url
            })

    return records

def extract_company(title):
    match = re.match(r"(.+?)\s+Set to Join", title)
    if match:
        return match.group(1).strip()
    return None


def collect_news(start_year=2020, end_year=2026):
    all_records = []

    for year in range(start_year, end_year + 1):
        all_records.extend(get_relevant_news(year))

    df_news = pd.DataFrame(all_records)

    # Extraer empresa
    df_news["Company"] = df_news["Title"].apply(extract_company)

    # Convertir fecha
    df_news["Publish Date"] = pd.to_datetime(df_news["Publish Date"])

    # Ordenar por empresa y fecha
    df_news = df_news.sort_values(["Company", "Publish Date"], ascending=[True, False])

    # Quedarse con la noticia más reciente por empresa
    df_news = df_news.drop_duplicates(subset="Company", keep="first")

    return df_news.reset_index(drop=True)

def extract_sp500_table_row(url):

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

        # Extraer texto limpio de cada celda
        texts = []
        for td in cells:
            span = td.find("span", class_="prnews_span")
            if span:
                texts.append(span.get_text(strip=True))
            else:
                texts.append(td.get_text(strip=True))

        # Saltar filas vacías
        if all(t == "" for t in texts):
            continue

        # Detectar encabezado
        if "Effective Date" in texts[0]:
            continue

        # Detectar fecha (solo aparece en la primera fila del bloque)
        if texts[0] != "":
            current_date = texts[0]

        # Caso típico: 7 columnas por culpa del colspan
        if len(texts) == 7:
            _, index_name, action, company_name, ticker, sector = texts[0], texts[1], texts[2], texts[4], texts[5], texts[6]

        # Caso normal: 6 columnas
        elif len(texts) == 6:
            _, index_name, action, company_name, ticker, sector = texts

        else:
            # Si algo raro ocurre, lo saltamos
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


# -------------------------------------------------
# 3. Pipeline BRUTO
# -------------------------------------------------

def scrape_sp500_raw(start_year=2020, end_year=2026):
    df_news = collect_news(start_year, end_year)

    all_rows = []

    for _, row in df_news.iterrows():
        df_table = extract_sp500_table_row(row["URL"])
        if df_table.empty:
            continue

        df_table["Publish Date"] = row["Publish Date"]
        df_table["Title"] = row["Title"]
        df_table["URL"] = row["URL"]

        all_rows.append(df_table)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)

def normalize_date(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    # Quitar puntos en abreviaturas: "Sept." → "Sept"
    x = re.sub(r"\.", "", x)

    # Reemplazar meses abreviados raros
    x = x.replace("Sept", "Sep")

    try:
        return parser.parse(x, dayfirst=False)
    except:
        return None


def clean_sp500_dataframe(df_raw):

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # -----------------------------
    # 1. Normalizar y convertir Effective Date
    # -----------------------------
    df["Effective Date"] = df["Effective Date"].apply(normalize_date)

    # -----------------------------
    # 3. Eliminar columnas no deseadas
    # -----------------------------
    df = df.drop(columns=[c for c in ["Index Name", "GICS Sector", "Title", "URL"] if c in df.columns])

    # -----------------------------
    # 4. Filtrar solo desde 01/01/2021
    # -----------------------------
    df = df[df["Effective Date"] >= pd.Timestamp("2021-01-01")]
    
    # 1. Asegurar que sea Datetime para poder operar
    df["Effective Date"] = pd.to_datetime(df["Effective Date"])

    # 2. Convertir a "Solo Fecha" (elimina el 00:00:00 visual)
    df["Effective Date"] = df["Effective Date"].dt.date

    # -----------------------------
    # 5. Ordenar por fecha (más reciente → más antiguo)
    # -----------------------------
    df = df.sort_values("Effective Date", ascending=True)

    return df.reset_index(drop=True)

def handler(event, context):
    try:
        # 1. Configuración DynamoDB
        TABLE_NAME = "clean_changes_sp500"
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(TABLE_NAME)

        # 2. Ejecutar Scraping del año actual
        actual_year = datetime.now().year
        raw_data = scrape_sp500_raw(actual_year, actual_year)
        
        if raw_data.empty:
            return {"status": "success", "message": "No hay noticias nuevas en la web."}

        df_new = clean_sp500_dataframe(raw_data)

        # 3. Leer lo que ya existe en DynamoDB para no duplicar
        # Nota: Scan es aceptable si la tabla es pequeña (pocos cambios al año)
        response = table.scan()
        existing_items = response.get('Items', [])
        df_existing = pd.DataFrame(existing_items)

        # 4. Comparar y filtrar solo lo nuevo
        if not df_existing.empty:
            # Creamos una columna temporal de ID para comparar fácilmente
            df_new['temp_id'] = df_new['Ticker'] + df_new['Effective Date'].astype(str)
            df_existing['temp_id'] = df_existing['Ticker'] + df_existing['Effective Date'].astype(str)
            
            # Solo nos quedamos con los que NO están en la base de datos
            df_to_save = df_new[~df_new['temp_id'].isin(df_existing['temp_id'])].copy()
            df_to_save = df_to_save.drop(columns=['temp_id'])
        else:
            df_to_save = df_new

        # 5. Guardar en DynamoDB
        if df_to_save.empty:
            print("No hay registros nuevos para añadir.")
            return {"status": "success", "message": "Todo al día."}

        for _, row in df_to_save.iterrows():
            item = {
                'Ticker': str(row['Ticker']),
                'Effective Date': str(row['Effective Date']),
                'Action': str(row['Action']),
                'Company Name': str(row['Company Name']),
                'Publish Date': str(row['Publish Date'])
            }
            table.put_item(Item=item)

        print(f"Se han añadido {len(df_to_save)} filas nuevas.")
        return {"status": "success", "added": len(df_to_save)}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}
