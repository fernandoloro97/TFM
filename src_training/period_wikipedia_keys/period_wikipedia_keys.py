import requests
import re
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
from thefuzz import process, fuzz
from datetime import datetime
from collections import Counter
from groq import Groq
import json
import boto3


# Instancio el dyanmodb
dynamodb = boto3.resource("dynamodb")
tabla_sp500_in_out = dynamodb.Table("sp500_in_out")
tabla_wikipedia_keys = dynamodb.Table("period_wikipedia_keys")

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
        
            
# Webscrapping a wikipedia
HEADERS = {"User-Agent": "MiScriptDePrueba/1.0 (contacto@tuemail.com)"}

# Obtengo la URL de cada empresa
def get_wiki_url(company_name):
    r = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, params={
        "action": "query", "list": "search",
        "srsearch": company_name, "srlimit": 1, "format": "json"
    }).json()
    results = r.get("query", {}).get("search", [])
    if not results:
        return None
    title = results[0]["title"].replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{title}"

# Webscrapeo las categorias de interes de cada URL
def scrape_wikipedia_infobox(company_name):
    empty = {k: None for k in ["Wikipedia Url", "Founders", "Predecessor", "Key People", "Products",
                            "Services", "Brands", "Divisions", "Subsidiaries"]}
    try:
        url = get_wiki_url(company_name)
        if not url:
            return empty

        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        infobox = soup.find("table", class_=lambda c: c and "infobox" in c)

        result = {"Founders": None, "Predecessor": None, "Key People": None,
        "Subsidiaries": None, "Products": None, "Services": None, "Brands": None,
        "Divisions": None}

        if not infobox:
            return result

        for row in infobox.find_all("tr"):
            header = row.find("th")
            data   = row.find("td")
            if not header or not data:
                continue

            label = header.get_text(strip=True).lower()
            items = data.find_all("li")
            if items:
                value = ", ".join(li.get_text(strip=True) for li in items)
            else:
                value = data.get_text(separator=" ", strip=True)

            if not value:
                continue

            if "founder"     in label: result["Founders"]      = value
            elif "predecessor" in label: result["Predecessor"] = value
            elif "key"       in label: result["Key People"]     = value
            elif "subsidiar" in label: result["Subsidiaries"]   = value
            elif "product"   in label: result["Products"]       = value
            elif "service"   in label: result["Services"]       = value
            elif "brand"     in label: result["Brands"]       = value
            elif "division"  in label: result["Divisions"]      = value


        return result

    except Exception as e:
        print(f"  Error {company_name}: {e}")
        return empty

# Ejecuto el webscrapeo para solo los nuevos tickers
rows = []
total = len(sp500_in_out)
for _, row in sp500_in_out.iterrows():
    data = scrape_wikipedia_infobox(row["Company Name"])
    rows.append({"Ticker": row["Ticker"], "Company Name": row["Company Name"], **data})
    time.sleep(0.3)

# Creo que df de nuevos tickers con sus categorias
new_wikipedia_keys = pd.DataFrame(rows, columns=[
    "Ticker", "Company Name", "Predecessor",
    "Products", "Services", "Brands", "Divisions", "Subsidiaries"
])
    
# Limpio y estandarizo los datos brutos webscrappeados
def clean_names(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    names = re.findall(r"[A-Z][a-zA-Z']+(?:\s[A-Z][a-zA-Z',.]+)+", text)
    if not names:
        return None
    seen = set()
    unique_names = [n for n in names if not (n in seen or seen.add(n))]
    return ", ".join(unique_names)

# Solo limpio
def clean_garbage(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*,", ",", text)
    return text.strip().strip(",").strip()

# Aplico la limpieza y estandarizado segun corresponda
new_wikipedia_keys["Predecessor"]   = new_wikipedia_keys["Predecessor"].apply(clean_garbage)
new_wikipedia_keys["Products"]   = new_wikipedia_keys["Products"].apply(clean_garbage)
new_wikipedia_keys["Services"]   = new_wikipedia_keys["Services"].apply(clean_garbage)
new_wikipedia_keys["Brands"]   = new_wikipedia_keys["Brands"].apply(clean_garbage)
new_wikipedia_keys["Divisions"]   = new_wikipedia_keys["Divisions"].apply(clean_garbage)
new_wikipedia_keys["Subsidiaries"]   = new_wikipedia_keys["Subsidiaries"].apply(clean_garbage)

# Conteo de palabras unicas
def get_unique_sorted_by_freq_multi(df, columns):

    combined_series = df[columns].stack().astype(str)
    all_items = [item.strip() for sublist in combined_series.str.split(",") for item in sublist]
    all_items = [item for item in all_items if item.lower() != 'nan' and item != '']
    counts = Counter(all_items)

    return [word for word, _ in counts.most_common()]

# Ejecuto el conteo para productos y servicios
problem_columns= ["Products", "Services"]
keywords = get_unique_sorted_by_freq_multi(new_wikipedia_keys, problem_columns)

# Para clasificar que es una palabra comun, utilizo un prompt en ingles de LLM
client = Groq(api_key="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")
# Prompt para clasiicar
def classify_batch(terms):
    numbered = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])

    prompt = f"""You are a classification assistant. For each term below, determine if it is a PROPER NOUN (specific brand, product, or named entity) or a COMMON NOUN (generic category or descriptive phrase).

Respond ONLY with a JSON array with one object per term, in the same order. No markdown, no explanation.

Example format:
[
{{"term": "Apple TV", "is_proper_noun": true}},
{{"term": "Apartments", "is_proper_noun": false}}
]

Terms to classify:
{numbered}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0
    )

    text = response.choices[0].message.content.strip()
    results = json.loads(text)
    return results

# Proceso por batches el prompt
def classify_all(terms, batch_size=20):
    all_results = []
    for i in range(0, len(terms), batch_size):
        batch = terms[i:i+batch_size]
        results = classify_batch(batch)
        all_results.extend(results)
    return all_results

# Guardo y transformo a df
results = classify_all(keywords)
common_words = pd.DataFrame(results)

# Creo un diccionario de las palabras comunes
proper_noun_set = set(
    common_words[common_words['is_proper_noun'] == True]['term'].tolist()
)

# Me quedo solo con los nombres propios
def filter_proper_nouns(cell_value):
    if pd.isna(cell_value) or cell_value == 'None':
        return cell_value

    terms = [t.strip() for t in str(cell_value).split(',')]
    filtered = [t for t in terms if t in proper_noun_set]

    return ', '.join(filtered) if filtered else None

# Filtro los nombres propios y guardo si sufieron cambios 
df_review = new_wikipedia_keys[['Ticker', 'Company Name', 'Products', 'Services']].copy()
df_review['Products_clean'] = new_wikipedia_keys['Products'].apply(filter_proper_nouns)
df_review['Services_clean'] = new_wikipedia_keys['Services'].apply(filter_proper_nouns)

df_review['Products_changed'] = df_review['Products'] != df_review['Products_clean']
df_review['Services_changed'] = df_review['Services'] != df_review['Services_clean']

# Reordeno para comparar mas rapido
df_review = df_review[['Ticker', 'Company Name',
                        'Products', 'Products_clean', 'Products_changed',
                        'Services', 'Services_clean', 'Services_changed']]

# Actualizo los productos y servicios, mostrando solo los que son nombres propios, el resto NaN
df_review_changed = df_review[
    df_review['Products_changed'] | df_review['Services_changed']
]

new_wikipedia_keys['Products'] = df_review['Products_clean']
new_wikipedia_keys['Services'] = df_review['Services_clean']

# Limpio los nombres de las empresas
def clean_company_name(name):
    if not isinstance(name, str):
        return name, ""

    suffixes = [
    r",?\s+Group\s+Holdings?",
    r",?\s+&\s+Company",
    r",?\s+&\s+Co\.?",
    r",?\s+and\s+Company",
    r",?\s+and\s+Co\.?",
    r",?\s+Incorporated",
    r",?\s+Corporation",
    r",?\s+Companies",
    r",?\s+Inc\.?",
    r",?\s+Corp\.?",
    r",?\s+Ltd\.?",
    r",?\s+Limited",
    r",?\s+LLC\.?",
    r",?\s+L\.L\.C\.?",
    r",?\s+PLC\.?",
    r",?\s+plc\.?",
    r",?\s+Holdings?",
    r",?\s+Company",            
    r",?\s+Enterprises?",
    r",?\s+Partners?",
    ]
    cleaned = name
    removed = []
    for suffix in suffixes:
        match = re.search(suffix + r"$", cleaned, flags=re.IGNORECASE)
        if match:
            removed.append(match.group().strip().strip(",").strip())
            cleaned = re.sub(suffix + r"$", "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip().strip(",").strip()
    return cleaned, ", ".join(removed) if removed else ""

# Copia de seguridad
df_review_names = new_wikipedia_keys[['Company Name']].copy()

# Aplico la limpieza para crear dos columnas: nombre limpio y lo que se limpio
df_review_names[['Company Name Clean', 'Removed']] = df_review_names['Company Name'].apply(
    lambda x: pd.Series(clean_company_name(x))
)

# Creo otra columna que me dice si hubo cambios en el nombre
df_review_names['changed'] = df_review_names['Company Name'] != df_review_names['Company Name Clean']

# Insert los nombres limpios al df principal
clean_names = df_review_names['Company Name Clean']
new_wikipedia_keys.insert(2, 'Company Name Clean', clean_names)

# Reviso si en el webscrappeo se leyo mal y falta comas, teniendo 2 o mas elementos en uno solo
columns = [
    'Predecessor', 'Products', 'Services',
    'Brands', 'Divisions', 'Subsidiaries'
]

# Reviso elementos de mas de 3 palabras sin comas y lo marco como sospechoso
def is_suspicious(text):
    if pd.isna(text):
        return False

    texto = str(text).strip()
    palabras = text.split()
    
    if len(palabras) > 3 and ',' not in text:
        return True
    return False

# Aplico el filtro de sospecha y lo guardo en una columna
new_wikipedia_keys['Need Revision'] = new_wikipedia_keys[columns].apply(lambda row: any(is_suspicious(val) for val in row), axis=1)

# Seleccion columnas con elementos sospechosos
columnas = [
    'Predecessor', 'Products', 'Services',
    'Brands', 'Divisions', 'Subsidiaries'
]

# Cojo la columna del elemeto sospechoso 
def get_incorrect_columns(row):
    cols_sospechosas = []
    for col in columnas:
        texto = str(row[col]).strip()
        if pd.isna(row[col]) or texto == "" or texto.lower() == 'nan':
            continue

        palabras = texto.split()
        if len(palabras) > 3 and ',' not in texto:
            cols_sospechosas.append(col)

    return cols_sospechosas if cols_sospechosas else None

# Creo una columna para columnas con elementos sospechosos
new_wikipedia_keys['Incorrect Columns'] = new_wikipedia_keys.apply(get_incorrect_columns, axis=1)
# Me quedo con las filas que si tiene columnas con elementos sospechosos
rows_to_correct = new_wikipedia_keys[new_wikipedia_keys['Incorrect Columns'].notna()]

# Aplico un prompt en ingles de LLM para comprobar si realmente los elementos sospechosos son reales y si son, corregirlos
client = Groq(api_key="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")

# Prompt para corregir los elementos sospechosos
def fix_merged_keywords(row, suspicious_columns):

    corrections = {}

    for col in suspicious_columns:
        value = row.get(col)
        if pd.isna(value) or value == 'None' or not value:
            continue

        prompt = f"""You are a data cleaning assistant. You are looking at a Wikipedia-scraped dataset about S&P 500 companies.

Company: {row.get('Company Name', 'Unknown')}
Column type: {col}
Raw value: "{value}"

This value was scraped from Wikipedia and may contain multiple items merged together without commas (e.g., "Apple TV Apple Vision Pro HomePod" should be "Apple TV, Apple Vision Pro, HomePod").

Your task:
1. Determine if this is a single item or multiple items merged without commas.
2. If multiple items, separate them with commas.
3. If it's genuinely a single item (even if long), return it as-is.

Respond ONLY with a JSON object, no markdown:
{{
"is_merged": true or false,
"corrected_value": "item1, item2, item3" or original value if single,
"reason": "one short sentence explaining your decision"
}}
"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        result = json.loads(text)
        corrections[col] = result

    return corrections

results = []

# Aplico el promot para las filas sospechosas
for idx, row in rows_to_correct.iterrows():
    suspicious_cols = row['Incorrect Columns'] 

    corrections = fix_merged_keywords(row, suspicious_cols)

    for col, result in corrections.items():
        results.append({
            'index': idx,
            'ticker': row['Ticker'],
            'column': col,
            'original': row[col],
            'corrected': result['corrected_value'],
            'is_merged': result['is_merged'],
            'reason': result['reason']
        })
# Lo guardo en df
df_corrections = pd.DataFrame(results)

# Aplico las correcciones directo al df principal
for _, fix in df_corrections.iterrows():
    new_wikipedia_keys.loc[fix['index'], fix['column']] = fix['corrected']

new_wikipedia_keys = new_wikipedia_keys.drop(columns=[ "Need Revision", "Incorrect Columns"])


# Subida del df a la tabla del dynomdb
# Copia de seguridad para modificaciones
df_preparado = new_wikipedia_keys.copy()

# Todas las columnas las paso a texto
for col in df_preparado.columns:
    df_preparado[col] = df_preparado[col].astype(str)

# Reemplazo NaN por vacio
df_preparado = df_preparado.replace(
    {"nan": None, "None": None, "": None, np.nan: None}
)

# Transformo el df a diccionario
items_to_upload = [
    {k: v for k, v in row.items() if v is not None}
    for row in df_preparado.to_dict(orient="records")
]

# 4. Subir el diccionario a la tabla
with tabla_wikipedia_keys.batch_writer() as batch:
    for item in items_to_upload:
        batch.put_item(Item=item)

print("Tabla period_wikipeda_keys subida")