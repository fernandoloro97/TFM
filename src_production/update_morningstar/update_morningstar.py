import boto3
import requests
import re
import pandas as pd
import time
from bs4 import BeautifulSoup
from thefuzz import process, fuzz
from datetime import datetime
from boto3.dynamodb.conditions import Attr

# Obtengo sector e indutria de Digrin
def get_digrin_data(ticker):
    url = f"https://digrin.com{ticker}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        if r.status_code == 404: return None, None, None
        title = soup.find("h1")
        company_digrin = title.text.replace(f"({ticker})", "").replace("Dividends", "").strip() if title else None
        sector, industry = None, None
        for a in soup.find_all("a", href=True):
            if "/stocks/list/sector/" in a["href"]: sector = a.text.strip()
            if "/stocks/list/industry/" in a["href"]: industry = a.text.strip()
        return company_digrin, sector, industry
    except:
        return None, None, None

# Manejo de actualizacion de tabla del dynamodb llamado companys_morningstar_sectors
def handler(event, context):
    try:
        # Configuro el dynamodb
        dynamodb = boto3.resource('dynamodb')
        
        # Creo las referencias a las tablas del dyanmodb
        table_changes = dynamodb.Table('clean_changes_sp500')
        table_morningstar = dynamodb.Table('morningstar_classification')
        table_sectors = dynamodb.Table('companys_morningstar_sectors')

        # Cargo y transformo las tablas a df
        morningstar = pd.DataFrame(table_morningstar.scan()['Items'])
        companys_morningstar_sectors = pd.DataFrame(table_sectors.scan()['Items'])

        # Filtro los tickers nuevos de hoy
        today = pd.Timestamp.now().normalize()
        res_changes = table_changes.scan(FilterExpression=Attr('Action').eq('Addition'))
        clean_changes_sp500 = pd.DataFrame(res_changes['Items'])

        if clean_changes_sp500.empty:
            return {"status": "success", "message": "No hay adiciones en clean_changes_sp500"}

        clean_changes_sp500['Effective Date'] = pd.to_datetime(clean_changes_sp500['Effective Date'])
        new_tickers = clean_changes_sp500[clean_changes_sp500['Effective Date'] <= today]
        
        # Me quedo con los tickers que realmente no tengo
        real_new_tickers = new_tickers[~new_tickers['Ticker'].isin(companys_morningstar_sectors['Ticker'])]

        if real_new_tickers.empty:
            return {"status": "success", "message": "No hay tickers nuevos para categorizar"}

        # Ejecuto el webscrapping a Digrin
        results_list = []
        for ticker in real_new_tickers["Ticker"]:
            company, sector, industry = get_digrin_data(ticker)
            results_list.append({
                "Ticker": ticker, "Company Name": company,
                "Sector": sector, "Industry": industry
            })
            time.sleep(0.5)
        # Lo guardo a df
        new_sectors = pd.DataFrame(results_list)

        # Me quedo con valores unicos
        sectors_officias_list = morningstar['Sector'].unique()
        industrys_officias_list = morningstar['Industry'].unique()

        # apluzo Fuzz matching para cazar los sectores e industri de Digrin con los oficial de Morningstar
        def look_for_official(value, official_list):
            if pd.isna(value) or value is None: return None
            match, score = process.extractOne(str(value), official_list, scorer=fuzz.token_sort_ratio)
            return match if score >= 90 else value

        # Aplico el mapeo
        new_sectors['Sector'] = new_sectors['Sector'].apply(lambda x: look_for_official(x, sectors_officias_list))
        new_sectors['Industry'] = new_sectors['Industry'].apply(lambda x: look_for_official(x, industrys_officias_list))

        # Añado el grupo industrial
        final_new_data = new_sectors.merge(
            morningstar[['Sector', 'Industry Group', 'Industry']], 
            on=['Sector', 'Industry'], 
            how='left'
        )

        # Guardo solo las filas nuevas en el dynamodb
        for _, row in final_new_data.iterrows():
            table_sectors.put_item(Item=row.to_dict())

        return {"status": "success", "added": len(final_new_data)}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}
