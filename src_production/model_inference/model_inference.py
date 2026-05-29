import os
import io
import json
import ast
import re
import math
import time
import unicodedata
import requests
import boto3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from decimal import Decimal
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, date, time as dt_time
from datetime import time as dt_time
from collections import defaultdict, Counter

import spacy
from ctransformers import AutoModelForCausalLM
from rapidfuzz import process, fuzz
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset



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

# Leo tablas de interes del dynamodb
composicion_sp500_actualizado = get_table_df('historic_composition_sp500')
wikipedia_actualizado = get_table_df('update_wikipedia_keys')
morningstar = get_table_df('morningstar_classification')
empresas_sectores_morningstar = get_table_df('companys_morningstar_sectors')

# Descargo los pesos de mi red neuronal optimizada
bucket_name = 'trained-neuronal-model'
key = 'NN_41_weights.pth'

buffer = io.BytesIO()
s3.download_fileobj(bucket_name, key, buffer)
buffer.seek(0)


# Extraccion de noticias
API_KEY = "1sO8gR66gfd7wAU8HaGYxqDMGhROZH0VsLMBY7WEXzjF6VgV"

# Descargo noticias del NEW YORK TIMES segun un rango de fechas seleccionadas
def descargar_nyt_periodo(fecha_inicio, fecha_final, query="news", max_paginas=100, sleep_time=6.0, reintentos=3):
    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    noticias = []

    begin_date = fecha_inicio.replace("-", "")
    end_date = fecha_final.replace("-", "")

    for page in range(max_paginas):
        intento = 0
        while intento < reintentos:
            params = {
                "q": query,
                "begin_date": begin_date,
                "end_date": end_date,
                "sort": "oldest",
                "page": page,
                "api-key": API_KEY
            }
            try:
                r = requests.get(url, params=params, timeout=30)
            except Exception as e:
                print(f"Error de conexión: {e}")
                intento += 1
                time.sleep(5)
                continue

            if r.status_code == 200:
                data = r.json()
                # CORRECCIÓN AQUÍ: Manejo seguro de la respuesta
                response = data.get("response", {})
                docs = response.get("docs", []) if response else []

                if not docs: # Si es None o lista vacía
                    print(f"No hay más artículos en página {page}. Fin de la búsqueda.")
                    return finalizar_dataframe(noticias)

                for d in docs:
                    noticias.append({
                        "Date": d.get("pub_date"),
                        "Section": d.get("section_name"),
                        "Title": d.get("headline", {}).get("main"),
                        "Content": d.get("abstract")
                    })

                print(f"Página {page} descargada ({len(docs)} artículos).")
                time.sleep(sleep_time) 
                break

            elif r.status_code in [429, 500]:
                intento += 1
                wait = 12 * intento 
                print(f"Error {r.status_code} (Rate Limit/Server) en pág {page}, reintento {intento}/{reintentos}...")
                time.sleep(wait)
            else:
                print(f"Error HTTP {r.status_code} en página {page}: {r.text}")
                return finalizar_dataframe(noticias)

    return finalizar_dataframe(noticias)

# Ajusto la hora de publicacion de noticias a la hora de Nueva York
def finalizar_dataframe(noticias):
    df = pd.DataFrame(noticias)
    if not df.empty:
        df["Date"] = (
            pd.to_datetime(df["Date"], utc=True, format='ISO8601')
            .dt.tz_convert(ZoneInfo("America/New_York"))
            .dt.tz_localize(None)
        )
        df = df.sort_values("Date").reset_index(drop=True)
    return df


# Guardo la fecha y dia de semana en el momento de ejecucion del codigo
hoy = datetime.now()
dia_semana = hoy.weekday()

print(f"Fecha de hoy: {hoy.strftime('%Y-%m-%d')} (día de la semana: {dia_semana})")

# Establezco el rango de dias a leer segun el dia de la semana
if dia_semana == 0:
    # Si es lunes, retrocedo 3 dias para cubrir el fin de semana
    dias_atras = 3
else:
    # Si no es lunes, solo retrocedo un dia
    dias_atras = 1

# Calculo el rango de fechas
start = (hoy - timedelta(days=dias_atras)).strftime("%Y-%m-%d") 
end = (hoy + timedelta(days=1)).strftime("%Y-%m-%d")   

# Descargo noticias
df_nyt = descargar_nyt_periodo(start, end, query="news")

# Convierto a data frame
df_nyt['Date'] = pd.to_datetime(df_nyt['Date'])

# Fijo las horas buscadas para el rango de fecha. Inicia al final del cierre del dia anterior hasta el cierre de hoy
start_period = datetime.combine(hoy.date() - pd.Timedelta(days=dias_atras), dt_time(18, 0, 0))
end_period = datetime.combine(hoy.date(), dt_time(18, 0, 0))

print(f"Rango de filtrado de noticias: Desde {start_period} hasta {end_period}")

# Filtro las noticias con el rango con fecha y hora antes calculado
last_news = df_nyt[(df_nyt['Date'] >= start_period) & (df_nyt['Date'] <= end_period)].copy()

# Ordeno por fecha 
last_news = last_news.sort_values(by='Date')

# Elimino las filas con NaN
last_news = last_news.dropna(subset=['Section', 'Title'])

# Reviso  duplicados
duplicate_news = last_news[last_news.duplicated(subset=["Title", "Content"], keep=False)]

# Elimino los duplicados manteniendo solo la primera fila encontrada
unique_news = last_news.drop_duplicates(subset=["Title", "Content"], keep='first').reset_index(drop=True)

# Lista de sección si rigos financiero
black_list = [
    'Crosswords & Games', 'Gameplay', 'Movies', 'Arts', 'Theater', 'Books',
    'Book Review', 'Briefing', 'Today’s Paper', 'Times Insider', 'Corrections',
    'Admin', 'Reader Center', 'Homepage', 'Video', 'Multimedia/Photos',
    'The Learning Network', 'Education', 'Parenting', 'Well', 'At Home',
    'Smarter Living', 'Neediest Cases', 'Giving', 'Sports', 'Obituaries',
    'Weather', 'Travel', 'Podcasts', 'En español', 'en Español', 'New York',
    'International Home', 'Lens', 'Universal', 'Home & Garden'
]

# Quito las noticias ruidosas
relevant_last_news = unique_news[~unique_news['Section'].isin(black_list)]

# Reseteo indice
relevant_last_news = relevant_last_news.reset_index(drop=True)

# Creo una nueva columna: seccion + titulo + contenido
relevant_last_news["Full Text"] = (
    "[SECTION] " + relevant_last_news["Section"] + "\n" +
    "[TITLE] " + relevant_last_news["Title"] + "\n" +
    "[CONTENT] " + relevant_last_news["Content"]
)

# Extraccion de palabras claves de empresas con NER
# Descargo NER
nlp = spacy.load("en_core_web_trf")

# Extraigo organizaciones, personas y producto. Personas al final no lo usaré
def extract_entities(text):
    doc = nlp(text)

    entities = {"ORG": set(), "PERSON": set(), "PRODUCT": set()}

    for ent in doc.ents:
        if ent.label_ in ("ORG", "PERSON", "PRODUCT"):
            entities[ent.label_].add(ent.text)

    return entities
 
# Aplico NER por bloques del total de noticias
def run_ner_and_process(df, chunk_size=5):
    total = len(df)
    n_chunks = math.ceil(total / chunk_size)

    print(f"Total noticias: {total} → {n_chunks} bloques de ~{chunk_size}")

    all_chunks = []

    for i in range(n_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size].copy()
        resultados = {"ORG": [], "PERSON": [], "PRODUCT": [], "ALL_ENTITIES": []}

        start = time.time()

        for text in chunk["Full Text"]:
            ents = extract_entities(str(text))
            orgs, persons, products = ents["ORG"], ents["PERSON"], ents["PRODUCT"]

            resultados["ORG"].append(orgs)
            resultados["PERSON"].append(persons)
            resultados["PRODUCT"].append(products)
            resultados["ALL_ENTITIES"].append(", ".join(sorted(orgs | persons | products)))

        elapsed = time.time() - start
        print(f"Procesado bloque {i}: {len(chunk)} noticias en {elapsed:.1f}s")

        # Creo el dataFrame del bloque actual
        df_chunk = pd.DataFrame({
            "Date": chunk["Date"].values,
            "Full Text": chunk["Full Text"].values,
            "Organization": [", ".join(sorted(x)) if x else None for x in resultados["ORG"]],
            "Person": [", ".join(sorted(x)) if x else None for x in resultados["PERSON"]],
            "Product": [", ".join(sorted(x)) if x else None for x in resultados["PRODUCT"]],
            "All Names": [x if x else None for x in resultados["ALL_ENTITIES"]],
        })

        # Guardo el df en la lista
        all_chunks.append(df_chunk)

    # Uno todos los bloques en un solo df
    df_final = pd.concat(all_chunks, ignore_index=True)
    return df_final

# Ejecucion de la extraccion de NER
noticias_ner_mapeados = run_ner_and_process(relevant_last_news, chunk_size=5)


# Extraccion de palabras claves de sectores con LLM
# Cargo el LLM mistral
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=0, 
    context_length=2048
)


# Prompt en ingles para extraer menciones financieras, canales economicos y pais afectado 
def build_prompt(news_text):
    prompt = f"""
<s>[INST]
You are a financial and economic event extractor for quantitative investment analysis.

Your job is to extract ONLY explicit economic or financial information.


1) FINANCIAL EVENT DETECTION (STRICT BUT COMPLETE)

Determine whether the article describes a concrete economic or financial event.

A valid financial event must involve at least one of the following:

• Corporate transactions (merger, acquisition, asset sale, debt issuance)
• Financial performance (revenue, profit, losses, earnings growth)
• Capital allocation (share buybacks, dividend changes, funding rounds)
• Government economic policy (tariffs, interest rate changes, fiscal measures)
• Macroeconomic data (GDP growth, inflation reports, unemployment figures)
• Commodity price movements (oil, gas, metals, agricultural commodities)
• Major operational financial disruptions affecting market activity

A financial mention MUST:

• Be a short complete phrase describing the economic event
• Include the financial action and context
• Reflect a measurable, transactional, or market-relevant development

Correct examples:
- "quarterly net income fell 12%"
- "company issued $2 billion in new bonds"
- "central bank raised interest rates by 50 basis points"
- "automaker announced a $5 billion share buyback program"
- "copper prices dropped 8% amid supply concerns"

Financial mentions must be written as natural descriptive phrases.
Do NOT use structured labels such as "company:", "transaction:", or similar prefixes.

Incorrect examples:
- Company names alone
- Person names alone
- Single words such as "deal", "profits", or "investment"
- Legal investigations without financial consequences
- Political statements without economic measures

If no concrete financial event is present,
RETURN ALL EMPTY LISTS.

Do NOT speculate.
Do NOT infer future outcomes.
unless explicit financial language is used.


2) ECONOMIC CHANNELS (INDUSTRY MAPPING REQUIRED)

If a financial event is identified,
map it to the relevant productive industries,
economic sectors, or market activities.

This step is REQUIRED if a financial event exists.

Guidelines:

• Identify which industry or economic activity is structurally involved.
• Include industries directly mentioned or clearly implied by the event.
• Prefer specific productive activities over broad aggregates.
• NEVER return geographic regions, continents, or countries as economic channels.
  - Example: "Latin American stocks" → return "equity market", not "Latin America"
  - Example: "European energy funds" → return "renewable energy market", not "Europe"
• Focus on the type of market, product, or sector affected.

Examples of correct mappings:

- Car manufacturer reports supply chain disruption → "automobile manufacturing", "automotive market"
- Pharma firm receives FDA approval for drug → "pharmaceutical industry", "biotechnology sector"
- Airline reduces flights due to fuel costs → "air transportation", "travel industry"
- Retail chain reports weak holiday sales → "retail sector", "consumer discretionary market"
- Steel tariffs announced → "steel industry", "manufacturing supply chains"
- Tech company releases new AI chip → "semiconductor industry", "artificial intelligence"
- Renewable energy project receives investment → "renewable energy market", "clean energy sector"
- Bank revises interest rates on mortgages → "banking sector", "mortgage market"

Rules:

• ALWAYS return industries, sectors, or economic activities.
• NEVER return locations, regions, continents, or countries as economic channels.
• Return 1–6 concise phrases.
• Every financial event must be associated with at least one relevant economic sector
  based on the nature of the entity or activity involved.
• If a financial event exists, an industry mapping is REQUIRED.


3) COUNTRIES / LOCATIONS

Extract locations only when clearly identifiable.
- Include continents, countries, regions, cities mentioned in the article.
- Map political leaders to their country (e.g., Trump → United States).
- Locations that appear in text but are not countries (e.g., Latin America, Europe) should also be included.
- Do NOT place any location in economic_channels.

If no location is explicitly mentioned, return an empty list.


4) OUTPUT FORMAT (MANDATORY)

You MUST return all keys exactly as shown.
If a category has no data, return [].
Do NOT omit keys.
Do NOT add extra keys.
Do NOT modify key names.

Return strictly and only this JSON:

{{
"financial_mentions": [],
"economic_channels": [],
"countries_involved": []
}}

Article:
{news_text}
[/INST]
"""
    return prompt


# Extraigo las palabras claves para las noticias
def extract_events_batch(news_texts):
    all_results = []
    for text in news_texts:
        prompt = f"[INST] {build_prompt(text)} [/INST]"
        respuesta = model(prompt, max_new_tokens=250, temperature=0)
        all_results.append(respuesta)
        
    return all_results

# Limpio la salidad del prompt
def parse_llm_output_clean(text):
    try:
        clean_text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
    except:
        data = {}

    def format_list(val):
        if isinstance(val, list) and len(val) > 0:
            return ", ".join(map(str, val))
        return None

    return {
        "Financial Mentions": format_list(data.get("financial_mentions")),
        "Economic Channels": format_list(data.get("economic_channels")),
        "Countries Involved": format_list(data.get("countries_involved"))
    }

# Extraccion y limpieza del LLM uno a uno 
def procesar_todo_en_variable(df_original):
    total = len(df_original)
    resultados_lista = []
    inicio = time.time()

    print(f"Procesando {total} noticias en CPU...")

    for i, (idx, row) in enumerate(df_original.iterrows()):
        texto_noticia = [row["Full Text"]]
        respuesta_raw = extract_events_batch(texto_noticia)

        datos_limpios = parse_llm_output_clean(respuesta_raw[0])

        # Guardo en la lista
        resultados_lista.append({
            "Date": row["Date"],
            "Full Text": row["Full Text"],
            **datos_limpios
        })

        print(f"la noticia {i+1}/{total} ha terminado")

    df_final = pd.DataFrame(resultados_lista)
    # Imprimo el tiempo porque suele demorar mucho
    print(f"\nTiempo total: {time.time() - inicio:.2f}s")
    
    return df_final

# Aplico la extraccion y limpieza
noticias_sectores_mapeados = procesar_todo_en_variable(relevant_last_news)

# Añado al df anterior una columna con las palabras claves de empresas obtenidas con NER
noticias_keywords_mapeados = noticias_sectores_mapeados.copy()
noticias_keywords_mapeados['All Names'] = noticias_ner_mapeados['All Names']


# Transformacion las palabras claves de empresas de Wikipedia a un diccionario
# Columnas de interes
COLUMNAS_KEYWORDS = ["Company Name", "Company Name Clean", "Predecessor", "Products",
                     "Services", "Brands", "Divisions", "Subsidiaries"]

keyword_index = {}  

# Transformo el df de palabras claves, tambien las limpio, a diccionario
for _, row in wikipedia_actualizado.iterrows():
    ticker = row["Ticker"]

    keywords = []
    for col in COLUMNAS_KEYWORDS:
        if isinstance(row[col], str):
            items = [k.strip() for k in row[col].split(",") if k.strip()]
            keywords.extend(items)

    for kw in keywords:
        kw_lower = kw.lower()

        if kw_lower not in keyword_index:
            keyword_index[kw_lower] = []
        if ticker not in keyword_index[kw_lower]:
            keyword_index[kw_lower].append(ticker)


# Keywords repetidas solo de productos y servicios
keyword_prod_serv = {}

for _, row in wikipedia_actualizado.iterrows():
    ticker = row["Ticker"]

    for col in ["Products", "Services"]:
        if isinstance(row[col], str):
            items = [k.strip() for k in row[col].split(",") if k.strip()]
            for kw in items:
                kw_lower = kw.lower()
                if kw_lower not in keyword_prod_serv:
                    keyword_prod_serv[kw_lower] = []
                if ticker not in keyword_prod_serv[kw_lower]:
                    keyword_prod_serv[kw_lower].append(ticker)

# Muestro solo los productos y servicios repetidos
repetidas = {kw: tickers for kw, tickers in keyword_prod_serv.items() if len(tickers) > 1}


# Columnas de interes
COLUMNAS_KEYWORDS = ["Predecessor",  "Products", "Services",
                    "Brands", "Divisions", "Subsidiaries"]

# Ultimo filtro de repetidas por columna
for col in COLUMNAS_KEYWORDS:
    keyword_col = {}
    for _, row in wikipedia_actualizado.iterrows():
        ticker = row["Ticker"]
        if isinstance(row[col], str):
            items = [k.strip() for k in row[col].split(",") if k.strip()]
            for kw in items:
                kw_lower = kw.lower()
                if kw_lower not in keyword_col:
                    keyword_col[kw_lower] = []
                if ticker not in keyword_col[kw_lower]:
                    keyword_col[kw_lower].append(ticker)

    repetidas_col = {kw: tickers for kw, tickers in keyword_col.items() if len(tickers) > 1}

keywords_a_eliminar = set(repetidas.keys()) 

# Columnas de interes
OTRAS_COLUMNAS = ["Predecessor", "Brands", "Divisions", "Subsidiaries"]

# Elimino repetidas que aparezcan en mas de 4 tickers
for col in OTRAS_COLUMNAS:
    keyword_col = {}
    for _, row in wikipedia_actualizado.iterrows():
        ticker = row["Ticker"]
        if isinstance(row[col], str):
            items = [k.strip() for k in row[col].split(",") if k.strip()]
            for kw in items:
                kw_lower = kw.lower()
                if kw_lower not in keyword_col:
                    keyword_col[kw_lower] = []
                if ticker not in keyword_col[kw_lower]:
                    keyword_col[kw_lower].append(ticker)

    for kw, tickers in keyword_col.items():
        if len(tickers) > 4:
            keywords_a_eliminar.add(kw)

# Construyo el df limpiecito
keyword_index_clean = {kw: tickers for kw, tickers in keyword_index.items()
                       if kw not in keywords_a_eliminar}


# Mapeo de noticias empresariales
# Copia porque se usara ese df para el mapeo de noticias sectoriales tambien
noticias_con_NER = noticias_keywords_mapeados.copy()

# Me quedo con las palabras claves unicas obtenidas con NER 
def get_unique_sorted_by_freq(df, column):
    all_items = df[column].dropna().str.split(", ")
    flat_list = [item.strip() for sublist in all_items for item in sublist]
    counts = Counter(flat_list)

    return [word for word, _ in counts.most_common()]

# Obtengo las palabra unicas
unique_names = get_unique_sorted_by_freq(noticias_con_NER, "All Names")
keywords = [x.lower() for x in unique_names]

# Mapeo por igualdad de palabras entre las palabras claves de NER y las oficial de las empresas (wikipedia)
NER_mapeado = pd.DataFrame({
    "original": keywords,
    "mapped": [keyword_index_clean.get(x) for x in keywords]
})

keys = list(keyword_index_clean.keys())

# Ahora el mapeo es mediante parecido
def get_match_info(x):
    match = process.extractOne(x, keys, scorer=fuzz.token_sort_ratio)
    if match:
        return match  
    return (None, 0, None)

# Ejecuto y veo los grados de similitud
NER_mapeado["match"], NER_mapeado["score"], _ = zip(*NER_mapeado["original"].apply(get_match_info))

# Umbral de similitud definida
THRESHOLD = 90
NER_mapeado["mapped_fuzzy"] = NER_mapeado.apply(
    lambda row: keyword_index_clean.get(row["match"]) if row["score"] >= THRESHOLD else None,
    axis=1
)

# Guardo las palabras no mapeados menores al 90% de coincidencia
NER_bajo_score = NER_mapeado[NER_mapeado["score"] < THRESHOLD].copy()

# Del df anterior, primero busco si comparten al menos un palabra igual
def has_common_word(row):
    if not isinstance(row["original"], str) or not isinstance(row["match"], str):
        return False

    original_words = set(row["original"].lower().split())
    match_words = set(row["match"].lower().split())

    return len(original_words & match_words) > 0

# Ejecuto para encontrar palabras en comun
NER_bajo_score["common_word"] = NER_bajo_score.apply(has_common_word, axis=1)

# Solo pasare al siguiente filtro las palabras que tuvieron al menos un palabra igual
NER_candidatos = NER_bajo_score[NER_bajo_score["common_word"] == True].copy()

# Utilizo un LLM para filtrar si realmente las palabras candidatas anteriors realmente reprensentan a una empresa
groc_client = Groq(api_key="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")

# Prompt para el mapeo final
def build_groc_prompt(original, match):
    prompt = f"""
You are an expert in company name recognition and entity matching.

Determine if the two names below refer to the same company or organization. Answer only with a JSON object, no markdown or explanation.

Original name: "{original}"
Match name: "{match}"

Rules:
- Return "Yes" if they refer to the same company/entity.
- Return "No" if they do not refer to the same entity.
- Return "Maybe" if it is unclear (e.g., ambiguous or partial match).

Respond ONLY with a JSON object like this:
{{
  "result": "Yes" | "No" | "Maybe",
  "reason": "one sentence explaining your reasoning"
}}
"""
    return prompt

# Aplico el prompt a las candidatas
def check_relation_groc(original, match):
    prompt = build_groc_prompt(original, match)

    response = groc_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"result": "Maybe", "reason": "Could not parse response"}

# Transformo el output del prompt a true y false
def parse_llm_output(row):
    llm_result = check_relation_groc(row["original"], row["match"])
    result_bool = llm_result.get("result", "Maybe") == "Yes"
    reason = llm_result.get("reason", "")

    return pd.Series([result_bool, reason])

# Aplico la funcion anterior
NER_candidatos[["LLM_check_bool", "LLM_reason"]] = NER_candidatos.apply(parse_llm_output, axis=1)

# Creo nueva columna de palabras mapeadas de wikipeda con LLM
NER_candidatos["mapped_llm"] = NER_candidatos.apply(
    lambda row: keyword_index_clean.get(row["match"]) if row["LLM_check_bool"] else None,
    axis=1
)

# Quito NaN
mask = NER_candidatos['mapped_llm'].notna()
NER_mapeado.loc[NER_candidatos[mask].index, 'mapped_fuzzy'] = NER_candidatos.loc[mask, 'mapped_llm']

# Creo diccionario con las palabras mapeadas
lookup = NER_mapeado.dropna(subset=['mapped_fuzzy']).set_index('original')['mapped_fuzzy'].to_dict()

# Añado el ticker por palabra de NER mapeado, sin repeticiones
def map_names_to_tickers(all_names):
    if pd.isna(all_names) or not all_names:
        return [] 

    names = [n.strip() for n in all_names.split(',')]
    tickers = []

    for name in names:
        ticker = lookup.get(name.lower())
        if ticker:
            if isinstance(ticker, list):
                tickers.extend(ticker)
            else:
                tickers.append(ticker)

    tickers = list(dict.fromkeys(tickers))

    return tickers

# Df con los tickers mapeados
noticias_con_NER['Tickers'] = noticias_con_NER['All Names'].apply(map_names_to_tickers)

# Auditoria de palabras claves de noticias con Tickers
# Busco la razon de mapeo de dichas palabras con los tickers
def get_mapping_con_evidencia(tickers_noticia, all_names_noticia, df_keywords, df_ref):
    if isinstance(all_names_noticia, str):
        all_names_list = [n.strip().lower() for n in all_names_noticia.split(',')]
    else:
        all_names_list = [str(n).strip().lower() for n in all_names_noticia]

    df_temp = df_keywords.copy()
    df_temp['original_lower'] = df_temp['original'].str.lower().str.strip()

    mapping_entries = []

    for ticker in tickers_noticia:
        nombre_oficial = df_ref.loc[df_ref['Ticker'] == ticker, 'Company Name'].values
        nombre_oficial = nombre_oficial[0] if len(nombre_oficial) > 0 else "Unknown Entity"

        mask = (df_temp['original_lower'].isin(all_names_list)) & \
               (df_temp['mapped_fuzzy'].apply(lambda x: ticker in x if isinstance(x, list) else ticker == str(x)))

        palabras_match = df_temp[mask]['original'].unique().tolist()
        evidencia = ", ".join(palabras_match) if palabras_match else "Contextual match"

        mapping_entries.append(f"- {ticker} ({nombre_oficial}). Razón: keyword '{evidencia}'")

    return "\n".join(mapping_entries)

# Prompt del LLM de auditaria financiera para controlar mapping sin el contexto de las noticias
def audit_tickers_with_groq(full_text, mapping_evidencia):
   
    system_prompt = """Act as a strict Financial Entity Auditor. Your goal is to filter a list of stock tickers by validating the "Evidence Mapping" against the "News Text".

CORE AUDIT RULES:
1. STRICT IDENTITY: A Ticker is ONLY valid if its "Trigger Keyword" is a direct synonym, brand, subsidiary, or current/former name of that specific corporate entity.
   - DO NOT keep a ticker based on industry association (e.g., do not keep a competitor or a related company in the same sector if it is not explicitly mentioned).
   - If Keyword 'A' triggered Ticker 'B', but Ticker 'B' is a different company than Keyword 'A', REMOVE IT.

2. STRATEGIC CONTEXT (apply with caution): In news regarding high-stakes litigation, mergers, or corporate takeovers involving a famous individual, KEEP the tickers of the primary companies they currently lead or own — BUT ONLY IF those companies are not overshadowed by a more specific entity mentioned in the news.
   - CRITICAL EXCEPTION: If the news is primarily and explicitly about a SPECIFIC company owned by that individual (e.g., SpaceX, xAI, X/Twitter), and that specific company is NOT in the suggested ticker list, DO NOT substitute it with another company owned by the same person. The absence of the primary entity from the universe does not justify keeping a different entity as a proxy.
   - Example: News about SpaceX → do NOT keep TSLA just because Elon Musk is mentioned. SpaceX is the subject; Tesla is irrelevant.

3. OPERATIONAL SPECIFICITY: If the news is strictly about the internal operations (e.g., taxes, manufacturing, local permits, contracts, government funding) of a specific named entity, REMOVE tickers of other companies owned by the same person if they are not involved in that specific operational event.

4. DUAL CLASS: Always keep both tickers for the same legal entity (e.g., GOOG/GOOGL).

5. SEMANTIC DISAMBIGUATION: Remove a ticker if its Keyword refers to a person or thing that is not a company (e.g., a person's name matching a company name but they are unrelated).

6. PRIMARY SUBJECT RULE: Identify the PRIMARY corporate subject of the news (the company the story is actually about). If that primary subject is NOT in the Evidence Mapping (because it is not in the investable universe), return an EMPTY list rather than keeping a different company as a proxy for it.

OUTPUT:
Return ONLY a valid Python list of strings. No reasoning. No headers.

INPUT:
- News Text: {full_text}
- Evidence Mapping: {mapping_evidencia}

"""

    user_content = f"News Text: {full_text}\n\nSuggested Mapping for Audit:\n{mapping_evidencia}"

    try:
        completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0
        )

        res = completion.choices[0].message.content.strip()

        if "```" in res:
            res = res.split("```")[-2].replace("python", "").strip()

        return ast.literal_eval(res)
    except Exception as e:
        print(f"Error en auditoría: {e}")
        return []

# Proceso la auditoria para cada noticias con palabras claves mapeadas
def procesar_noticias_muchos_tickers(df, df_keywords, df_ref, delay=0.5):
    resultado = df[['Full Text', 'All Names', 'Tickers']].copy()
    resultado['Tickers Filtrados'] = None
    resultado['Cambios'] = None

    errores = []
    total = len(df)
    ultima_fila_ok = None

    for i, idx in enumerate(df.index):
        try:
            noticia = df.loc[idx, 'Full Text']
            names   = df.loc[idx, 'All Names']
            tickers = df.loc[idx, 'Tickers']

            if isinstance(tickers, list):
                tickers_originales = sorted([str(t).strip() for t in tickers])
            elif isinstance(tickers, str):
                tickers_originales = sorted([t.strip() for t in tickers.strip('[]').split(',')])
            else:
                tickers_originales = []

            propuesta         = get_mapping_con_evidencia(tickers, names, df_keywords, df_ref)
            tickers_filtrados = audit_tickers_with_groq(noticia, propuesta)

            if isinstance(tickers_filtrados, list):
                tickers_filtrados_sorted = sorted([str(t).strip() for t in tickers_filtrados])
            else:
                tickers_filtrados_sorted = []
                tickers_filtrados = []

            hubo_cambio = tickers_originales != tickers_filtrados_sorted

            resultado.at[idx, 'Tickers Filtrados'] = tickers_filtrados
            resultado.at[idx, 'Cambios']           = hubo_cambio

            ultima_fila_ok = idx

            time.sleep(delay)

        except Exception as e:
            print(f"\n [{i+1}/{total}] idx={idx} | ERROR: {e}")
            print(f" Proceso interrumpido. Última fila procesada correctamente: idx={ultima_fila_ok} (posición {i}/{total})")
            errores.append({'idx': idx, 'error': str(e)})
            resultado.at[idx, 'Tickers Filtrados'] = []
            resultado.at[idx, 'Cambios']           = None
            break 

    procesadas = resultado['Cambios'].notna().sum()
    print(f"\n{'Completado' if not errores else 'Parcial'}. Filas procesadas: {procesadas}/{total}")
    if ultima_fila_ok:
        print(f"Última fila guardada: idx={ultima_fila_ok}")

    return resultado

# Filtro noticias con mas de un ticker mapeado
noticias_muchos_tickers = noticias_con_NER[noticias_con_NER['Tickers'].apply(lambda x: len(x) > 1)]


# Si no hay candidatos para auditar, devuelvo el df final vacio
if noticias_muchos_tickers.empty:
    print("El DataFrame 'noticias_muchos_tickers' está vacío. Me salto el procesamiento.")
    noticias_auditadas = pd.DataFrame() 
else:
    # Divido en 6 partes
    partes = np.array_split(noticias_muchos_tickers, 6)

    # Ejecuto la auditoria por partes
    for i, parte in enumerate(partes):

        if not parte.empty:
            print(f"Parte {i+1}: {len(parte)} filas | indices {parte.index[0]} a {parte.index[-1]}")
        else:
            print(f"Parte {i+1}: 0 filas (vacia)")

    api_keys = [
        "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", 
        "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", 
        "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", 
        "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", 
        "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", 
        "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"
    ]

    resultados = []
    for i, (parte, key) in enumerate(zip(partes, api_keys)):
        # Solo procesamos si la parte tiene contenido
        if parte.empty:
            continue

        print(f"\n Procesando parte {i+1}/6 con key {i+1}")
        client = Groq(api_key=key)
        resultado = procesar_noticias_muchos_tickers(parte, NER_mapeado, wikipedia_actualizado)
        resultados.append(resultado)

    # Uno los resultados, si es que hay
    if resultados:
        noticias_auditadas = pd.concat(resultados)
    else:
        noticias_auditadas = pd.DataFrame()


# Actualizo los tickers corregidos por la auditoria, si es que hay noticias auditadas
if not noticias_auditadas.empty:
    noticias_con_NER.loc[noticias_auditadas.index, 'Tickers'] = noticias_auditadas['Tickers Filtrados']
    print(f"Se actualizaron {len(noticias_auditadas)} filas.")
else:
    print("No hay noticias auditadas para actualizar.")


# Aseguro el formato datetime para las fechas
noticias_con_NER['Date_only'] = pd.to_datetime(noticias_con_NER['Date']).dt.date
composicion_sp500_actualizado['Date'] = pd.to_datetime(composicion_sp500_actualizado['Date']).dt.date

# Convierto los tickers de mi historico a un diccionario
universo_dict = composicion_sp500_actualizado.set_index('Date')['Ticker'].str.split(', ').apply(set).to_dict()

# Me quedo con los tickers mapeados que estuvieron ese dia en el SP500
def filtrar_tickers(row):
    fecha = row['Date_only']
    tickers_fila = row['Tickers']

    if not tickers_fila or fecha not in universo_dict:
        return []

    universo_dia = universo_dict[fecha]

    return [t for t in tickers_fila if t in universo_dia]

# Aplico el filtro
noticias_con_NER['Tickers'] = noticias_con_NER.apply(filtrar_tickers, axis=1)

# Elimino la columna con sola la fecha
noticias_con_NER = noticias_con_NER.drop(columns=['Date_only'])

# Mapeo de noticias sectoriales
# Creo una copia 
noticias_con_sector = noticias_keywords_mapeados.copy()

# Obtengo valores uncos de canales economicos
def get_unique_sorted_by_freq(df, column):
    all_items = df[column].dropna().str.split(", ")
    flat_list = [item.strip() for sublist in all_items for item in sublist]

    counts = Counter(flat_list)

    # solo devolver las palabras ordenadas por frecuencia
    return [word for word, _ in counts.most_common()]

# Ejecuto la anterior funcion
unique_channels = get_unique_sorted_by_freq(noticias_keywords_mapeados, "Economic Channels")

# Mapeo de palabras claves sectoriales con las oficial del clasificador de Morningstar
groq_client = Groq(api_key="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")

# Transformo el df de grupos estructurados del clasificador de Morningstar a diccionario
def parse_morningstar_df(df: pd.DataFrame) -> dict:
    sectors = {sector: "" for sector in df["Sector"].unique()}

    industries = df.rename(columns={
        "Sector": "sector",
        "Industry Group": "industry_group",
        "Industry": "industry"
    }).to_dict(orient="records")

    for item in industries:
        if "description" not in item:
            item["description"] = ""

    return {"sectors": sectors, "industries": industries}

# Construyo una catalago de 3 niveles: sectores, grupos industriales e industrias
def build_catalog(parsed: dict) -> dict:

    industries    = parsed["industries"]
    catalog_sectors = []
    catalog_groups  = []
    catalog_inds    = []

    groups = defaultdict(list)
    for ind in industries:
        groups[(ind["sector"], ind["industry_group"])].append(ind)

    sector_groups = defaultdict(list)
    for (sector, group), members in groups.items():
        sector_groups[sector].append((group, members))

    # Nivel industria
    for ind in industries:
        catalog_inds.append({
            "level":          "industry",
            "sector":         ind["sector"],
            "industry_group": ind["industry_group"],
            "industry":       ind["industry"],
            "text":           ind["industry"]
        })

    # Nivel grupo industrial
    for (sector, group), members in groups.items():
        catalog_groups.append({
            "level":          "industry_group",
            "sector":         sector,
            "industry_group": group,
            "industry":       None,
            "text":           group
        })

    # Nivel sector
    for sector, groups_list in sector_groups.items():
        catalog_sectors.append({
            "level":          "sector",
            "sector":         sector,
            "industry_group": None,
            "industry":       None,
            "text":           sector
        })

    return {
        "sectors":    catalog_sectors,
        "groups":     catalog_groups,
        "industries": catalog_inds
    }


# Transformo a embedding para los grupos economicos del catalogo
def build_embeddings(catalog: dict, model: SentenceTransformer):
    embeddings = {}
    for level, items in catalog.items():
        texts = [item["text"] for item in items]

        print(f"Generando embeddings para {level} ({len(texts)} entradas)")
        embeddings[level] = model.encode(texts, show_progress_bar=True)

    return embeddings

# Limpio y normalizo
def clean_nlp_input(text: str) -> str:
    noise   = r'\b(sector|industry|market)\b'
    cleaned = re.sub(noise, '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.lower()


# Clasifico las palabras claves de sectores con los grupos de Morningstar con un prompt en ingles de LLM
# Uso esta funcion para un nivel de similitud de coseno bajo, poco fiable
def classify_with_llm(nlp_input: str, catalog: dict, probable_sectors: list = None) -> dict:

    sectors_list = "\n".join(
        f"- {item['sector']}"
        for item in catalog["sectors"]
    )

    groups_list = "\n".join(
        f"- {item['sector']} > {item['industry_group']}"
        for item in catalog["groups"]
    )

    if probable_sectors:
        filtered_industries = [
            item for item in catalog["industries"]
            if item["sector"] in probable_sectors
        ]
    else:
        filtered_industries = catalog["industries"]

    industries_list = "\n".join(
        f"- {item['sector']} > {item['industry_group']} > {item['industry']}"
        for item in filtered_industries
    )

    prompt = f"""You are a financial sector classification expert using the Morningstar taxonomy.

Given this financial news topic: "{nlp_input}"

Classify it using ONLY the options below. Choose the most specific level possible.

SECTORS:
{sectors_list}

INDUSTRY GROUPS (format: Sector > Industry Group):
{groups_list}

INDUSTRIES (format: Sector > Industry Group > Industry):
{industries_list}

Rules:
- ALWAYS return the most specific level possible. If ANY industry matches,
  you MUST return industry level, never stop at industry_group.
- If the topic refers to a broad economic concept, aggregate market activity, or
  geographic region rather than a specific company-level industry or sector,
  return unclassified. This includes:
  * Any government policy or aggregate economic measure (spending, deficits,
    growth, employment levels, monetary or fiscal policy)
  * Any financial market concept not tied to a specific industry (equity markets,
    bond markets, capital flows, liquidity)
  * Any geographic region, continent, or country-level concept
  * When in doubt: if no real company could be classified under this topic,
    return unclassified

Respond ONLY with a JSON object, no explanation, no markdown:
{{
  "sector": "exact sector name from the list or null",
  "industry_group": "exact industry group name from the list or null",
  "industry": "exact industry name from the list or null",
  "level": "sector or industry_group or industry or unclassified",
  "reason": "one sentence explanation"
}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    try:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            return {"sector": None, "industry_group": None, "industry": None,
                    "level": "unclassified", "reason": "JSON parsing error"}

        result = json.loads(json_match.group())

        sector         = result.get("sector")
        industry_group = result.get("industry_group")
        industry       = result.get("industry")
        level          = result.get("level", "unclassified")

        if industry_group and ">" in industry_group:
            parts          = industry_group.split(">")
            sector         = parts[0].strip()
            industry_group = parts[1].strip()

        if industry and ">" in industry:
            parts    = industry.split(">")
            industry = parts[-1].strip()

        valid_sectors    = {item["sector"] for item in catalog["sectors"]}
        valid_groups     = {item["industry_group"] for item in catalog["groups"]}
        valid_industries = {item["industry"] for item in catalog["industries"]}

        if sector and sector not in valid_sectors:
            return {"sector": None, "industry_group": None, "industry": None,
                    "level": "unclassified",
                    "reason": f"unknown sector: {sector}"}

        if industry_group and industry_group not in valid_groups:
            return {"sector": None, "industry_group": None, "industry": None,
                    "level": "unclassified",
                    "reason": f"unknown group: {industry_group}"}

        if industry and industry not in valid_industries:
            return {"sector": None, "industry_group": None, "industry": None,
                    "level": "unclassified",
                    "reason": f"unknown industry: {industry}"}

        if sector and industry_group and industry:
            level = "industry"
        elif sector and industry_group and not industry:
            level = "industry_group"
        elif sector and not industry_group and not industry:
            level = "sector"
        else:
            level = "unclassified"

        if level == "industry" and sector and industry_group and industry:
            label = f"{sector} > {industry_group} > {industry}"
        elif level == "industry_group" and sector and industry_group:
            label = f"{sector} > {industry_group}"
        elif level == "sector" and sector:
            label = sector
        else:
            label = None
            level = "unclassified"

        return {
            "sector":         sector,
            "industry_group": industry_group,
            "industry":       industry,
            "level":          level,
            "label":          label,
            "reason":         result.get("reason", "")
        }

    except Exception as e:
        return {"sector": None, "industry_group": None, "industry": None,
                "level": "unclassified", "reason": f"error: {str(e)}"}

# Construyo y estandarizo un diccionario de los resultados directos obtenido por un alto de nivel de similitud de coseno
def build_result_embedding(nlp_output, cleaned, best_item, level,
                  confidence, best_per_level, top_n,
                  llm_used, llm_reason, source):

    if best_item is None:
        return {
            "input":          nlp_output,
            "cleaned":        cleaned,
            "label":          None,
            "sector":         None,
            "industry_group": None,
            "industry":       None,
            "level":          "unclassified",
            "confidence":     round(confidence, 3),
            "llm_used":       llm_used,
            "llm_reason":     llm_reason,
            "source":         source,
            "best_per_level": {l: {"score": round(best_per_level[l]["score"], 3),
                                   "label": best_per_level[l]["top_n"][0]["label"]}
                               for l in best_per_level},
            "top_n":          top_n
        }

    if best_item.get("level") == "industry":
        label = f"{best_item['sector']} > {best_item['industry_group']} > {best_item['industry']}"
    elif best_item.get("level") == "industry_group":
        label = f"{best_item['sector']} > {best_item['industry_group']}"
    else:
        label = best_item["sector"]

    return {
        "input":          nlp_output,
        "cleaned":        cleaned,
        "label":          label,
        "sector":         best_item.get("sector"),
        "industry_group": best_item.get("industry_group"),
        "industry":       best_item.get("industry"),
        "level":          level,
        "confidence":     round(confidence, 3),
        "llm_used":       llm_used,
        "llm_reason":     llm_reason,
        "source":         source,
        "best_per_level": {l: {"score": round(best_per_level[l]["score"], 3),
                               "label": best_per_level[l]["top_n"][0]["label"]}
                           for l in best_per_level},
        "top_n":          top_n
    }

# Construyo y estandarizo los resultado del LLM
def build_result_LLM(nlp_output, cleaned, llm_result, confidence,
                    best_per_level, top_n, llm_used, llm_reason, source):

  sector         = llm_result.get("sector")
  industry_group = llm_result.get("industry_group")
  industry       = llm_result.get("industry") 
  level          = llm_result.get("level", "unclassified")
  label          = llm_result.get("label")

  return {
      "input":          nlp_output,
      "cleaned":        cleaned,
      "label":          label,
      "sector":         sector,
      "industry_group": industry_group,
      "industry":       industry,  
      "level":          level,
      "confidence":     round(confidence, 3),
      "llm_used":       llm_used,
      "llm_reason":     llm_reason,
      "source":         source,
      "best_per_level": {l: {"score": round(best_per_level[l]["score"], 3),
                              "label": best_per_level[l]["top_n"][0]["label"]}
                          for l in best_per_level},
      "top_n":          top_n
  }

# Mapeo las palabras claves de sector con las del clasificador de Morningstar, viendo primero la similitud de coseno
# Si es mayor a 0.65, lo doy por bueno, pero si es menor, pasa por un prompt de LLM que termine de mapearlo
def normalize_sector(nlp_output: str,
                     catalog: dict,
                     embeddings: dict,
                     model: SentenceTransformer,
                     threshold_high: float = 0.65,
                     threshold_min:  float = 0.25,
                     top_n:          int   = 3) -> dict:

    cleaned   = clean_nlp_input(nlp_output)
    query_emb = model.encode([cleaned])

    best_per_level = {}

    for level in ["sectors", "groups", "industries"]:
        items  = catalog[level]
        embs   = embeddings[level]
        scores = cosine_similarity(query_emb, embs)[0]

        top_idx  = scores.argsort()[::-1][:top_n]
        best_idx = int(scores.argmax())

        best_per_level[level] = {
            "item":  items[best_idx],
            "score": float(scores[best_idx]),
            "top_n": [
                {
                    "label": (
                        f"{items[i]['sector']} > {items[i]['industry_group']} > {items[i]['industry']}"
                        if items[i]["level"] == "industry"
                        else f"{items[i]['sector']} > {items[i]['industry_group']}"
                        if items[i]["level"] == "industry_group"
                        else items[i]["sector"]
                    ),
                    "score": round(float(scores[i]), 3)
                }
                for i in top_idx
            ]
        }

    winner_level = max(best_per_level, key=lambda l: best_per_level[l]["score"])
    winner       = best_per_level[winner_level]
    best_score   = winner["score"]
    best_item    = winner["item"]

    # Cojo solo los 2 mejores sectores de similitud
    sector_items  = catalog["sectors"]
    sector_embs   = embeddings["sectors"]
    sector_scores = cosine_similarity(query_emb, sector_embs)[0]
    top2_idx      = sector_scores.argsort()[::-1][:2]
    probable_sectors = [sector_items[i]["sector"] for i in top2_idx]

    level_map = {
        "sectors":    "sector",
        "groups":     "industry_group",
        "industries": "industry"
    }

    # Un score demasiado bajo:LLM clasifica directamente
    if best_score < threshold_min:
        llm_result = classify_with_llm(cleaned, catalog, probable_sectors)
        return build_result_LLM(nlp_output, cleaned, llm_result,
                                best_score, best_per_level, winner["top_n"],
                                llm_used=True, llm_reason=llm_result["reason"],
                                source="llm_direct")

    # Un score alto: embedding directo sin LLM
    if best_score >= threshold_high:
        return build_result_embedding(nlp_output, cleaned, best_item,
                             level_map[winner_level], best_score,
                             best_per_level, winner["top_n"],
                             llm_used=False, llm_reason="high confidence",
                             source="embedding")

    # Un score medio: LLM clasifica
    llm_result = classify_with_llm(cleaned, catalog, probable_sectors)

    if llm_result["level"] != "unclassified":
        return build_result_LLM(nlp_output, cleaned, llm_result,
                                best_score, best_per_level, winner["top_n"],
                                llm_used=True, llm_reason=llm_result["reason"],
                                source="llm_medium")
    else:
        return build_result_embedding(nlp_output, cleaned, None, "unclassified",
                             best_score, best_per_level, winner["top_n"],
                             llm_used=True, llm_reason=llm_result["reason"],
                             source="llm_medium")



# Ejecuto todo el pipeline de mapeo para noticias sectoriales
parsed     = parse_morningstar_df(morningstar) 
catalog    = build_catalog(parsed)
model      = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = build_embeddings(catalog, model)

levels = Counter(item["level"] for level_items in catalog.values()
                 for item in level_items)

resultados = []

for i, s in enumerate(unique_channels):
    try:
        r = normalize_sector(s, catalog, embeddings, model)
        resultados.append(r)

        if (i + 1) % 100 == 0:
            print(f"Procesados: {i+1} keywords")

    except Exception as e:
        if "429" in str(e):
            # Guardo lo avanzado
            print(f"Rate limit en keyword {i+1}: {s}")
            print(f"Guardando {len(resultados)} resultados")
            break
        else:
            raise e

# Muestro los resultados en un df
clasificacion_sector = pd.DataFrame([{
    "input":      r["input"],
    "label":      r["label"],
    "level":      r["level"],
    "output":     r["industry"] or r["industry_group"] or r["sector"],
    "confidence": r["confidence"],
    "source":     r["source"],
    "llm_reason": r["llm_reason"]
} for r in resultados])

# Diccionario de mapeo
mapeo = {}
for _, row in clasificacion_sector.iterrows():
    if row["level"] not in ["unclassified", "error"]:
        mapeo[row["input"]] = row["output"]

# Genero los grupos economicos oficiales sin repetir dado los canales economicos por noticia
def mapear_canales(economic_channels_str, mapeo):
    if pd.isna(economic_channels_str):
        return []

    canales = [c.strip().lower() for c in economic_channels_str.split(",")]

    # Obtenemos los resultados únicos
    resultados = list(dict.fromkeys(
        mapeo[canal] for canal in canales if canal in mapeo
    ))

    # CAMBIO: Si no hay resultados, devolvemos [] en lugar de None
    return resultados if resultados else []

# Aplico la funcion anterior para crear la columna Morningstar
noticias_con_sector["Morningstar"] = noticias_con_sector["Economic Channels"].apply(
    lambda x: mapear_canales(x, mapeo)
)

# Diccionario de sectores, grupos indutriales e industrias por ticker
empresa_por_clasificador = {}

for _, row in empresas_sectores_morningstar.iterrows():
    for clasificador in [row["Sector"], row["Industry Group"], row["Industry"]]:
        if pd.notna(clasificador):
            if clasificador not in empresa_por_clasificador:
                empresa_por_clasificador[clasificador] = []
            empresa_por_clasificador[clasificador].append(row["Ticker"])


# Mapeo de tickers por los grupos economicos oficial mapeaos de Morningstar
def obtener_tickers(morningstar_list):
    if not isinstance(morningstar_list, list) or not morningstar_list:
        return []

    tickers = []
    for clasificador in morningstar_list:
        encontrados = empresa_por_clasificador.get(clasificador, [])
        tickers.extend(encontrados)

    resultado = list(dict.fromkeys(tickers))

    return resultado

# Aplico el mapeo con tickers
noticias_con_sector["Tickers"] = noticias_con_sector["Morningstar"].apply(obtener_tickers)


# Aseguro el formato datetime para las fechas
noticias_con_sector['Date_only'] = pd.to_datetime(noticias_con_sector['Date']).dt.date
composicion_sp500_actualizado['Date'] = pd.to_datetime(composicion_sp500_actualizado['Date']).dt.date

# Convierto los tickers de mi historico a un diccionario
universo_dict = composicion_sp500_actualizado.set_index('Date')['Ticker'].str.split(', ').apply(set).to_dict()

# Filtro los tickers que no estaban en el SP500 el dia de la publicacion de la noticia
def filtrar_tickers(row):
    fecha = row['Date_only']
    tickers_fila = row['Tickers']

    if not tickers_fila or fecha not in universo_dict:
        return []

    universo_dia = universo_dict[fecha]

    return [t for t in tickers_fila if t in universo_dia]

# Aplico la funcion de filtro de tickers
noticias_con_sector['Tickers'] = noticias_con_sector.apply(filtrar_tickers, axis=1)

# Elimino la columnas auxiliar de solo fechas
noticias_con_sector = noticias_con_sector.drop(columns=['Date_only'])


# Contruyo mi df de inputs para el analisis gramatical
noticias_input = noticias_con_sector.drop(columns=['Morningstar', 'Tickers'], errors='ignore').copy()

# Añado los tickers mapeados tanto de noticias empresariales como sectoriales
noticias_input['Tickers Sector'] = noticias_con_sector['Tickers']
noticias_input['Tickers Empresa'] = noticias_con_NER['Tickers']

# Creo un columnas con los tickers totales mapeados
noticias_input['Tickers Combinados'] = [
    sorted(list(set(s + e))) for s, e in zip(noticias_input['Tickers Sector'], noticias_input['Tickers Empresa'])
]

# Condicion para noticias empresariales: Debe tener palabras claves obtenidas en NER, mencion financieras y tickers mapeados
condicion_1 = (
    noticias_input['All Names'].notna() &
    noticias_input['Financial Mentions'].notna() &
    (noticias_input['Tickers Empresa'].str.len() > 0)
)

# Condicion para noticias sectoriales: Debe tener canales economicos y tickers mapeados
condicion_2 = (
    noticias_input['Economic Channels'].notna() &
    (noticias_input['Tickers Sector'].str.len() > 0)
)

# Filtro final, me quedo con las noticias que cumplieron al menos una condicion
noticias_input_filtrado = noticias_input[condicion_1 | condicion_2].copy()
noticias_input_filtrado = noticias_input_filtrado.reset_index(drop=True)

# Reviso si no se encontraron noticias que superaran ese filtro
print(f"¿Hay noticias candidatas para analizarlas gramaticalmente?: {noticias_input_filtrado.empty}") 

# Analisis gramatical de noticias empresariales y sectoriales

# Descargo el modelo de mistral
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=0,
    context_length=2048
)

# Genero una lista de palabras claves de empresas obtenidas con NER que tienen un ticker mapeado
def get_keywords_empresa(tickers_noticia, all_names_noticia, df_keywords):
    if isinstance(all_names_noticia, str):
        all_names_list = [n.strip().lower() for n in all_names_noticia.split(',')]
    else:
        all_names_list = [str(n).strip().lower() for n in all_names_noticia]

    df_temp = df_keywords.copy()
    df_temp['original_lower'] = df_temp['original'].str.lower().str.strip()

    keywords_encontradas = []

    for ticker in tickers_noticia:
        mask = (df_temp['original_lower'].isin(all_names_list)) & \
               (df_temp['mapped_fuzzy'].apply(lambda x: ticker in x if isinstance(x, list) else ticker == str(x)))

        palabras_match = df_temp[mask]['original'].unique().tolist()
        keywords_encontradas.extend(palabras_match)

    return list(dict.fromkeys(keywords_encontradas))


# Diccionario del mapeo de noticias sectoriales
mapeo = {}
for _, row in clasificacion_sector.iterrows():
    if row["level"] not in ["unclassified", "error"]:
        mapeo[row["input"]] = row["output"]

# Genero una lista de palabras claves de sectores obtenidas con el LLM que tienen un ticker mapeado
def get_keywords_sector(economic_channels, mapeo):
    if not economic_channels or (isinstance(economic_channels, float)):
        return []

    if isinstance(economic_channels, str):
        channels_list = [c.strip().lower() for c in economic_channels.split(',')]
    else:
        channels_list = [str(c).strip().lower() for c in economic_channels]

    channels_validos = []
    for channel in channels_list:
        if mapeo.get(channel):
            channels_validos.append(channel)

    return channels_validos

# Prompt em ingles para extraer las categorias de analisis gramatical para noticias empresariales
def extract_event_empresa(full_text, tickers_empresa, tickers_sector, economic_channels, all_names, df_keywords, mapeo):

    keywords_empresa = get_keywords_empresa(tickers_empresa, all_names, df_keywords) if tickers_empresa and len(tickers_empresa) > 0 else []
    keywords_sector = get_keywords_sector(economic_channels, mapeo) if economic_channels and not isinstance(economic_channels, float) else []

    if not keywords_empresa:
        return []

    participantes = "BLOCK A - Entity keywords explicitly found in the news:\n"
    participantes += "\n".join([f'- "{kw}"' for kw in keywords_empresa])

    contexto_sector = ""
    if keywords_sector:
        contexto_sector = f"\nECONOMIC CONTEXT (for reference only — do NOT include in output):\n"
        contexto_sector += "\n".join([f'- "{kw}"' for kw in keywords_sector])

    prompt = f"""<s>[INST]
You are a specialized financial news analyst.
The news below follows this structure:
- TITLE: the main headline — defines the PRIMARY financial event.
- CONTENT: additional context — contains details, names, and entities involved.

NEWS:
"{full_text}"

MAPPED KEYWORDS (use ONLY these in agent_keywords, patient_keywords, and affected_keywords):
{participantes}
{contexto_sector}

TASK:
Extract exactly ONE financial event based on the TITLE.

STEP 1 - READ THE TITLE:
The TITLE defines the PRIMARY financial event. Identify the main action and its participants.
If a keyword only appears in supporting context (CONTENT), it is likely AFFECTED.

STEP 2 - ASSIGN EACH KEYWORD A ROLE:
For each keyword in MAPPED KEYWORDS, read TITLE and CONTENT and decide:

- Use the EXACT keyword string as it appears in MAPPED KEYWORDS. Do not paraphrase, modify, or abbreviate it.

Assign EXACTLY ONE of these roles:

- AGENT → performs the main action in the TITLE
- PATIENT → receives the main action
- AFFECTED → mentioned in the news but does NOT directly perform or receive the main action (e.g., background, origin, affiliation, comparison, investor, alumni, former employer, etc.)

RULES:
- A keyword MUST appear in exactly ONE of the three roles.
- A keyword CANNOT appear in more than one role.
- If multiple keywords share the same role, group them together.
- If a keyword is not clearly agent or patient, assign it to AFFECTED.
- NEVER force a keyword into agent or patient if the role is weak or indirect.

STEP 3 - SUMMARIZE:
Write ONE sentence using the assigned keyword roles. Print it before the JSON.

STEP 4 - FILL IN:
- verb: - Must contain ONLY ONE verb. Never include multiple verbs. ONE pure infinitive in English. No conjugations, no auxiliaries, no modifiers.
  WRONG: "reported", "is acquiring", "can make", "couldn't recover", "has declined"
  RIGHT: "report", "acquire", "make", "recover", "decline"
  Extract only the main action verb. If unclear → null.

- context: capture the main topic of the news in 2-4 words maximum. Be specific and concise.
  Examples: "earnings report", "product recall", "merger agreement", "stock decline", "drug approval", "government contract", "leadership change"
  If unclear → null.

- status: use ONLY one of these:
  CONFIRMED POSITIVE  → verified fact, favorable outcome (increase, approval, deal closed, profit)
  CONFIRMED NEGATIVE  → verified fact, unfavorable outcome (decline, rejection, loss, fine)
  CONFIRMED NEUTRAL   → verified fact, no clear direction
  SPECULATIVE         → rumor, prediction, anonymous source, forecast, discussion
  IN PROGRESS         → ongoing negotiation, open investigation, pending approval

  CRITICAL:
- Return ONE JSON if all keywords relate to the same event.
- Return MULTIPLE JSONs only if keywords are involved in clearly distinct actions OR represent different moments in time — in that case, put the most recent event FIRST.
- If multiple sub-events exist for the same moment, merge them into ONE by grouping keywords that perform the same action in agent_keywords.
- If different keywords are associated with different verbs, they represent separate events.
- Only use keywords from MAPPED KEYWORDS in agent_keywords, patient_keywords, and affected_keywords. Never invent new ones.
- Every MAPPED KEYWORD must appear in EXACTLY ONE of:
  agent_keywords, patient_keywords, or affected_keywords.
- No headers. No markdown.

Summary: <your one sentence summary here>
[
  {{
    "agent_keywords": ["keyword"] or null,
    "patient_keywords": ["keyword"] or null,
    "affected_keywords": ["keyword"] or null,
    "verb": "pure infinitive",
    "context": "concise financial context" or null,
    "status": "CONFIRMED POSITIVE|CONFIRMED NEGATIVE|CONFIRMED NEUTRAL|SPECULATIVE|IN PROGRESS" or null
  }}
]
[/INST]
"""
    try:
        response = llm(  #llm
            prompt,
            max_new_tokens=400,
            temperature=0,
            stop=["[/INST]", "</s>"]
        )

        response = response.strip()

        matches = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
        if not matches:
            return []

        return [json.loads(m) for m in matches]

    except Exception as e:
        print(f"Error en la generación o parseo: {e}")
        return []

# Trasnformo a minuscula
def normalize_to_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, list):
        return [str(i).strip().lower() for i in x if i is not None and not pd.isna(i)]

    return [str(x).strip().lower()]

# Estandarizacion
def clean_text(x):
    if x is None:
        return None
    return str(x).strip().lower()

# Transformo el ouput del prompt a df
def eventos_a_df_fast(resultado, indice_noticia, keywords_validas):

    filas = []
    append = filas.append

    if not resultado:
        append((
            indice_noticia,
            1,
            [],
            [],
            [],
            None,
            None,
            None
        ))

        return pd.DataFrame(
            filas,
            columns=[
                "Fila Noticia", "Evento", "Agente", "Paciente",
                "Afectado", "Verbo", "Contexto", "Status"
            ]
        )

    for i, event in enumerate(resultado, 1):

        agent = normalize_to_list(event.get("agent_keywords"))
        patient = normalize_to_list(event.get("patient_keywords"))
        affected = normalize_to_list(event.get("affected_keywords"))

        verb = clean_text(event.get("verb"))
        context = clean_text(event.get("context"))
        status = clean_text(event.get("status"))

        append((
            indice_noticia,
            i,
            keywords_validas,
            agent,
            patient,
            affected,
            verb,
            context,
            status
        ))

    return pd.DataFrame(
        filas,
        columns=[
            "Fila Noticia", "Evento", "Keywords Validas", "Agente", "Paciente",
            "Afectado", "Verbo", "Contexto", "Status"
        ]
    )

# Orquesto las funciones anteriores, desde pasarlo por el prompt hasta obtener un df con los resultados estructurados
def procesar_noticias_completo(noticias):
    """
    Procesa todas las noticias y devuelve un DataFrame con los eventos.
    No guarda archivos en disco.
    """
    inicio_total = time.time()
    all_dfs = []
    tiempos = []

    for i in noticias.index:
        fila = noticias.loc[i]
        t0 = time.time()

        # Aplico el LLM para el analisis gramatical
        resultado = extract_event_empresa(
            full_text=fila['Full Text'],
            tickers_empresa=fila['Tickers Empresa'],
            tickers_sector=fila['Tickers Sector'],
            economic_channels=fila['Economic Channels'],
            all_names=fila['All Names'],
            df_keywords=NER_mapeado,
            # df_ref=sp500_historico,
            mapeo=mapeo
        )

        # Obtengo las keywords con tickers mapeados
        keywords_filtradas = get_keywords_empresa(
            tickers_noticia=fila['Tickers Empresa'],
            all_names_noticia=fila['All Names'],
            df_keywords=NER_mapeado
        )

        # Convierto a df
        df_eventos = eventos_a_df_fast(
            resultado=resultado,
            indice_noticia=i,
            keywords_validas=keywords_filtradas
        )

        all_dfs.append(df_eventos)

        # Registo del tiempo de ejecucion
        duracion = time.time() - t0
        tiempos.append(duracion)

        if (len(tiempos) % 5 == 0):
            print(f"Procesadas {len(tiempos)}/{len(noticias)} - Última: {duracion:.2f}s")

    # Unifico todos los resultados en un solo df
    if all_dfs:
        df_final = pd.concat(all_dfs, ignore_index=True)
    else:
        df_final = pd.DataFrame()

    fin_total = time.time()

    # Dado el tiempo dilato del procesamiento, registro los tiempos
    print(f"Noticias totales: {len(noticias)}")
    print(f"Tiempo total: {(fin_total - inicio_total)/60:.2f} min")
    print(f"Tiempo medio por noticia: {np.mean(tiempos):.2f} s")

    return df_final

# Selecciono solo noticias empresariales
mask = noticias_input_filtrado['Tickers Empresa'].apply(lambda x: len(x) > 0)
noticias_con_empresas = noticias_input_filtrado[mask].copy()

# Si hay dichas noticias, ejecuto el analisis gramatical
if not noticias_con_empresas.empty:
    print(f"Iniciando proceso para {len(noticias_con_empresas)} noticias")
    resultado_empresa = procesar_noticias_completo(noticias_con_empresas)
else:
    print("No se encontraron noticias con empresas. Devuelvo un df vacio")
    # Creamos un df vacio
    resultado_empresa = pd.DataFrame(columns=[
        "Fila Noticia", "Evento", "Keywords Validas", "Agente", "Paciente",
        "Afectado", "Verbo", "Contexto", "Status"
    ])
    
 
# Prompt em ingles para extraer las categorias de analisis gramatical para noticias sectoriales   
def extract_event_sector(full_text, tickers_sector, economic_channels, mapeo):
    keywords_sector = get_keywords_sector(economic_channels, mapeo) if economic_channels and not isinstance(economic_channels, float) else []

    if not keywords_sector:
        return []

    participantes = "SECTOR KEYWORDS (economic sectors identified as relevant to this news):\n"
    participantes += "\n".join([f'- "{kw}"' for kw in keywords_sector])

    prompt = f"""<s>[INST]
You are a specialized financial news analyst.
The news below follows this structure:
- TITLE: the main headline — defines the PRIMARY financial event.
- CONTENT: additional context — contains details, names, and entities involved.

NEWS:
"{full_text}"

SECTOR KEYWORDS (economic sectors previously identified as relevant to this news):
{participantes}

These sector keywords do NOT appear literally in the news — they represent economic sectors that a previous system determined are economically relevant to this event.

TASK:
Extract exactly ONE financial event based on the TITLE and assign each sector keyword a role.

STEP 1 - READ THE TITLE:
The TITLE defines the PRIMARY financial event. Identify in one sentence: who does what to whom.

STEP 2 - VALIDATE EACH KEYWORD:
Before assigning a role, check if the keyword represents a productive or economic sector (e.g. "banking sector", "automotive industry", "pharmaceutical market").
- If the keyword is NOT a productive/economic sector (e.g. it is a social concept, a verb, a vague term) → assign it directly to AFFECTED.
- If the keyword IS a productive/economic sector → proceed to STEP 3.

STEP 3 - ASSIGN EACH SECTOR KEYWORD A ROLE:
For each valid sector keyword, decide its role based on your understanding of the news:
- AGENT: the sector is grammatically the subject performing or triggering the action in the news.
- PATIENT: the sector is grammatically the object directly receiving the action in the news.
- AFFECTED: the sector is a valid economic sector but does not appear grammatically as subject or object of the main action — it is economically impacted but not a direct participant.
A keyword can only have ONE role. If it is AGENT or PATIENT, do NOT also put it in affected.
- If the real agent or patient found in the news refers to the same economic sector as the keyword (even with a different name), use the exact keyword string in the output instead.

STEP 4 - SUMMARIZE:
Write ONE sentence summarizing the main financial event using the real grammatical subject and object from the news. Print it before the JSON.

STEP 5 - FILL IN:
- agent: list of the real grammatical subject(s) of the main action as they appear in the news. This can be ANY entity — companies, people, governments, natural phenomena, abstract forces, etc. null only if truly no subject exists in the news.
- patient: list of the real grammatical object(s) of the main action as they appear in the news. This can be ANY entity. null only if truly no object exists in the news.
- affected: list of sector keywords that are valid economic sectors but do not directly participate grammatically. null if none.
- verb: the main action verb in infinitive form. Extract only the core verb, no auxiliaries. null if unclear.
- context: capture the main topic of the news in 2-4 words maximum.
  Examples: "earnings report", "product recall", "merger agreement", "stock decline", "drug approval", "government contract"
  null if unclear.
- status: use ONLY one of these:
  CONFIRMED POSITIVE  → verified fact, favorable outcome
  CONFIRMED NEGATIVE  → verified fact, unfavorable outcome
  CONFIRMED NEUTRAL   → verified fact, no clear direction
  SPECULATIVE         → rumor, prediction, anonymous source, forecast
  IN PROGRESS         → ongoing negotiation, open investigation, pending approval

CRITICAL:
- Every sector keyword must appear in EITHER agent, patient OR affected. Never omit any.
- agent and patient must ALWAYS be lists (e.g. ["Federal Reserve"]) or null. Never plain strings.
- agent and patient are free text from the news or exact sector keyword if semantically equivalent. affected only contains sector keywords.
- Return ONE JSON if all keywords share the same role.
- Return MULTIPLE JSONs only if keywords are involved in clearly distinct actions — put most recent FIRST.
- No headers. No markdown.

Summary: <your one sentence summary here>
[
  {{
    "agent": ["real entity or sector keyword"] or null,
    "patient": ["real entity or sector keyword"] or null,
    "affected": ["sector keyword"] or null,
    "verb": "pure infinitive" or null,
    "context": "2-4 words" or null,
    "status": "CONFIRMED POSITIVE|CONFIRMED NEGATIVE|CONFIRMED NEUTRAL|SPECULATIVE|IN PROGRESS" or null
  }}
]
[/INST]"""

    response = llm(
        prompt,
        max_new_tokens=300,
        temperature=0, 
        stop=["</s>"]
    )

    try:
        matches = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
        if not matches:
            return []

        result = [json.loads(m.replace("'", '"')) for m in matches]
        return result
    except Exception as e:
        print(f"Error parseando JSON: {e}")
        return []

# Transformo a minuscula
def normalize_to_list(x):

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, list):
        return [str(i).strip().lower() for i in x if i is not None and not pd.isna(i)]

    return [str(x).strip().lower()]

# Estandarizacion
def clean_text(x):
    if x is None:
        return None
    return str(x).strip().lower()

# Transformo el ouput del prompt a df
def eventos_a_df_fast(resultado, indice_noticia, keywords_validas):

    kw_set = set(kw.lower() for kw in keywords_validas)

    filas = []
    append = filas.append

    if not resultado:
        append((
            indice_noticia,
            1,
            keywords_validas,  
            [],
            [],
            [],
            None,
            None,
            None
        ))

        return pd.DataFrame(
            filas,
            columns=[
                "Fila Noticia", "Evento", "Keywords Validas", "Agente", "Paciente",
                "Afectado", "Verbo", "Contexto", "Status"
            ]
        )

    for i, event in enumerate(resultado, 1):

        agent = normalize_to_list(event.get("agent"))
        patient = normalize_to_list(event.get("patient"))
        affected = normalize_to_list(event.get("affected"))

        verb = clean_text(event.get("verb"))
        context = clean_text(event.get("context"))
        status = clean_text(event.get("status"))

        agent_f = [kw for kw in agent if kw in kw_set]
        patient_f = [kw for kw in patient if kw in kw_set]
        affected_f = [kw for kw in affected if kw in kw_set]

        append((
            indice_noticia,
            i,
            keywords_validas,  
            agent_f,
            patient_f,
            affected_f,
            verb,
            context,
            status
        ))

    return pd.DataFrame(
        filas,
        columns=[
            "Fila Noticia", "Evento", "Keywords Validas", "Agente", "Paciente",
            "Afectado", "Verbo", "Contexto", "Status"
        ]
    )


# Orquesto las funciones anteriores, desde pasarlo por el prompt hasta obtener un df con los resultados estructurados
def procesar_noticias(noticias, mapeo):
    inicio_total = time.time()
    todos_los_dfs = []
    tiempos = []

    for i in noticias.index:
        fila = noticias.loc[i]
        t0 = time.time()

        # Aplico el LLM para el analisis gramatical
        resultado = extract_event_sector(
            full_text         = fila['Full Text'],
            tickers_sector    = fila['Tickers Sector'],
            economic_channels = fila['Economic Channels'],
            mapeo             = mapeo
        )

        # Obtengo keywords con tickers mapeados
        keywords_filtradas = get_keywords_sector(
            economic_channels = fila['Economic Channels'],
            mapeo = mapeo
        )

        # Convierto a df
        df_eventos = eventos_a_df_fast(
            resultado=resultado,
            indice_noticia=fila.name,
            keywords_validas=keywords_filtradas
        )

        todos_los_dfs.append(df_eventos)
        tiempos.append(time.time() - t0)

        if (len(todos_los_dfs) % 10) == 0:
            print(f"Procesadas {len(todos_los_dfs)}/{len(noticias)} noticias")

    # Consolido todo en un df
    df_final = pd.concat(todos_los_dfs, ignore_index=True)

    fin_total = time.time()

    # Dado el tiempo dilato del procesamiento, registro los tiempos
    print(f"Noticias totales: {len(noticias)}")
    print(f"Tiempo total: {(fin_total - inicio_total)/60:.2f} min")
    print(f"Tiempo medio por noticia: {np.mean(tiempos):.2f} s")

    return df_final

# Selecciono solo noticias sectoriales
mask = noticias_input_filtrado['Tickers Sector'].apply(lambda x: len(x) > 0)
noticias_con_sectores = noticias_input_filtrado[mask].copy()

# Si hay dichas noticias, ejecuto el analisis gramatical
if not noticias_con_sectores.empty:
    print(f"Iniciando proceso para {len(noticias_con_sectores)} noticias")
    resultado_sector = procesar_noticias(noticias_con_sectores, mapeo)
    
else:
    print("No se encontraron noticias con sectores. Devuelvo un df vacio.")
    # Creo un df vacio
    resultado_sector = pd.DataFrame(columns=[
        "Fila Noticia", "Evento", "Keywords Validas", "Agente", "Paciente",
        "Afectado", "Verbo", "Contexto", "Status"
    ])
    

# Limpieza de los resultados del analisis gramatical
# Empiezo con los roles de noticias empresariales   
ner  = resultado_empresa.copy()
# Me cargo los NaN
ner = ner.dropna()

# Reviso si hay roles repetidos
def es_repetida(row):
    keywords = list(row['Keywords Validas']) if isinstance(row['Keywords Validas'], (list, np.ndarray)) else []

    agente = list(row['Agente']) if isinstance(row['Agente'], (list, np.ndarray)) else []
    paciente = list(row['Paciente']) if isinstance(row['Paciente'], (list, np.ndarray)) else []
    afectado = list(row['Afectado']) if isinstance(row['Afectado'], (list, np.ndarray)) else []

    pool_palabras = agente + paciente + afectado

    for kw in keywords:
        if pool_palabras.count(kw) > 1:
            return True
    return False

# Reviso si hay repeticion, si hay noticias empresariales sin NaN
if not ner.empty:
    repetidos_empresa = ner[ner.apply(es_repetida, axis=1)]
else:
    # Devuelvo un df vacio
    repetidos_empresa = pd.DataFrame(columns=ner.columns)

# Ejecuto la limpieza si hay repetidos
if not repetidos_empresa.empty:
    print(f"Limpiando {len(repetidos_empresa)} filas con duplicados")

    # Si hay roles en agente y otros mas, me quedo con el de agente
    def limpiar_duplicados(row):
        agente = set(row['Agente']) if isinstance(row['Agente'], (list, np.ndarray)) else set()

        paciente_limpio = [x for x in row['Paciente'] if x not in agente]
        afectado_limpio = [x for x in row['Afectado'] if x not in agente]

        return pd.Series([paciente_limpio, afectado_limpio])

    # Aplicamos la limpieza
    ner[['Paciente', 'Afectado']] = ner.apply(limpiar_duplicados, axis=1)
else:
    print("No se detectaron repetidos. Saltando limpieza.")

# Añado un diferencial que es una noticia empresarial, guarandolo como True
ner["Mencionado"] = True


# Continuo con la limpieza de roles de las noticias sectoriales
sector  = resultado_sector.copy()
# Me cargo los NaN
sector = sector.dropna()

# Si hay noticias sin NaN, verifico si tengo roles repetidos
if not ner.empty:
    repetidos_sector = sector[sector.apply(es_repetida, axis=1)]
else:
    # Genero df vacio
    repetidos_sector = pd.DataFrame(columns=sector.columns)

# Me cargo el rol afectado si aparece tambien en otro rol
def limpiar_afectado(row):
    agente = set(row["Agente"])
    paciente = set(row["Paciente"])

    existentes = agente.union(paciente)

    row["Afectado"] = [
        x for x in row["Afectado"]
        if x not in existentes
    ]

    return row

# Aplico la funcion anterior
sector = sector.apply(limpiar_afectado, axis=1)

# Si hay roles en agente y paciente, me quedo con agente
def limpiar_agente_paciente(row):
    agente_set = set(row["Agente"])
    paciente_set = set(row["Paciente"])

    overlap = agente_set.intersection(paciente_set)

    if overlap:
        row["Paciente"] = [x for x in row["Paciente"] if x not in overlap]

    return row

# Aplico la funcion anterior
sector = sector.apply(limpiar_agente_paciente, axis=1)

# Añado un diferencial que es una noticia sectorial, guarandolo como False
sector["Mencionado"] = False

# Uno en un solo df las noticias empresariales y sectoriales
inputs_totales = pd.concat([ner, sector], ignore_index=True)

# Organizo por Fila Noticas y dentro de ella por evento
inputs_totales = inputs_totales.sort_values(by=["Fila Noticia", "Evento"]).reset_index(drop=True)

# Genero una fila por palabra clave o keyword
rows = []

for _, r in inputs_totales.iterrows():
    fila = r["Fila Noticia"]

    for rol in ["Agente", "Paciente", "Afectado"]:
        keywords = r[rol]

        if isinstance(keywords, (list, np.ndarray)):
            for kw in keywords:
                rows.append({
                    "Fila Noticia": fila,
                    "Evento": r["Evento"],
                    "Rol": rol.lower(),
                    "Keyword": str(kw).lower(), 
                    "Verbo": r["Verbo"],
                    "Contexto": r["Contexto"],
                    "Status": r["Status"],
                    "Mencionado": r["Mencionado"]
                })

inputs_totales_por_rol = pd.DataFrame(rows)

# Obtengo los tickers de las palabras claves
def get_tickers(row):
    keyword = row["Keyword"]
    fila = row["Fila Noticia"]

    # Noticias empresariales
    if row["Mencionado"]:
        ner_dict = NER_mapeado.set_index("original")["mapped_fuzzy"].to_dict()
        tickers_ner = ner_dict.get(keyword, [])
        tickers_news = noticias_input_filtrado["Tickers Empresa"].to_dict().get(fila, [])

        if not isinstance(tickers_ner, list):
            tickers_ner = []

        if not isinstance(tickers_news, list):
            tickers_news = []

        return list(set(tickers_ner) & set(tickers_news))

    # Noticias sectoriales
    else:
        sector = mapeo.get(keyword)

        if not sector:
            return []

        dict_sector = empresas_sectores_morningstar.groupby('Sector')['Ticker'].apply(list).to_dict()
        dict_group = empresas_sectores_morningstar.groupby('Industry Group')['Ticker'].apply(list).to_dict()
        dict_industry = empresas_sectores_morningstar.groupby('Industry')['Ticker'].apply(list).to_dict()

        empresa_por_morningstar = {**dict_sector, **dict_group, **dict_industry}

        tickers_sector = empresa_por_morningstar.get(sector, [])
        tickers_news = noticias_input_filtrado["Tickers Sector"].to_dict().get(fila, [])

        if isinstance(tickers_sector, (list, np.ndarray)):
            tickers_sector = list(tickers_sector)
        else:
            tickers_sector = []

        if isinstance(tickers_news, (list, np.ndarray)):
            tickers_news = list(tickers_news)
        else:
            tickers_news = []

        return list(set(tickers_sector) & set(tickers_news))

# Si hay filas con palabras claves, entonces continuo el proceso de generacion de inputs
if not inputs_totales_por_rol.empty:
    
    # Aplico la busqueda de tickers
    inputs_totales_por_rol["Tickers Mapeados"] = inputs_totales_por_rol.apply(get_tickers, axis=1)

    # Me guardo solo palabras claves con tickers
    inputs_totales_por_rol = inputs_totales_por_rol[
        inputs_totales_por_rol["Tickers Mapeados"].map(lambda x: len(x) > 0 if isinstance(x, list) else False)
    ]

    inputs_totales_por_rol = inputs_totales_por_rol.reset_index(drop=True)

    # Limipieza de estatus
    # Normalizo estatus
    def normalize_text(text):
        if pd.isna(text):
            return text

        text = text.lower()
        text = text.replace("_", " ")
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # Limpio el estatus
    def clean_status(text):
        text = normalize_text(text)

        if text is None:
            return text

        if "speculative" in text:
            return "speculative"

        if "progress" in text:
            return "in progress"

        if "confirmed" in text:
            if "positive" in text:
                return "confirmed positive"
            elif "negative" in text:
                return "confirmed negative"
            else:
                return "confirmed neutral"

        return text

    # Limpio estatus, porque el prompt alucino un poco
    inputs_totales_por_rol["Status"] = inputs_totales_por_rol["Status"].apply(clean_status)

    # Limpieza de verbos
    # Me quedo con los verbos problematicos: NaN o mas de un verbo
    mask_error = (
        ~inputs_totales_por_rol["Verbo"].apply(lambda x: isinstance(x, str)) 
        |
        inputs_totales_por_rol["Verbo"].astype(str).str.contains(",")        
    )

    df_verbos_problematicos = inputs_totales_por_rol[mask_error]

    # Si hay mas de un verbo, me quedo con el primero, suele ser mas preciso
    inputs_totales_por_rol["Verbo"] = inputs_totales_por_rol["Verbo"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) else x
    )

    # En caso de noticias y evento iguale para noticias empr. y sect., me quedo 
    # con el verbo, contexto y estatus de la noticia empresarial, suele ser mas preciso
    true_values = (
        inputs_totales_por_rol[inputs_totales_por_rol["Mencionado"] == True]
        .groupby(["Fila Noticia", "Evento"])
        .agg({
            "Verbo": "first",
            "Contexto": "first",
            "Status": "first"
        })
        .rename(columns={
            "Verbo": "Verbo_true",
            "Contexto": "Contexto_true",
            "Status": "Status_true"
        })
        .reset_index()
    )

    inputs_totales_por_rol = inputs_totales_por_rol.merge(
        true_values,
        on=["Fila Noticia", "Evento"],
        how="left"
    )

    # Reemplazo los verbos, contextos y estatus de las noticias empresariales sobre las sectoriales para misma noticia y evento
    # Verbos
    inputs_totales_por_rol["Verbo"] = inputs_totales_por_rol.apply(
        lambda x: x["Verbo_true"] if pd.notna(x["Verbo_true"]) else x["Verbo"],
        axis=1
    )
    # Contexto
    inputs_totales_por_rol["Contexto"] = inputs_totales_por_rol.apply(
        lambda x: x["Contexto_true"] if pd.notna(x["Contexto_true"]) else x["Contexto"],
        axis=1
    )
    # Estatus
    inputs_totales_por_rol["Status"] = inputs_totales_por_rol.apply(
        lambda x: x["Status_true"] if pd.notna(x["Status_true"]) else x["Status"],
        axis=1
    )

    inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Verbo_true", "Contexto_true", "Status_true"])


    # Normalizacion de verbos
    # Normalizo con un prompt en ingles de LLM
    def extract_main_verbs_batch(phrases, client):

        if isinstance(phrases, str):
            phrases = [phrases]

        numbered = "\n".join([f"{i+1}. {p}" for i, p in enumerate(phrases)])

        prompt = f"""You are a linguistic expert in Syntax and Semantics.

    Your goal is to isolate the HEAD VERB of each phrase.

    CORE LOGIC:
    1. IDENTIFY THE HEAD: Locate the primary action. If there is a chain of verbs (e.g., "agree to pay"), the final action verb is the head.
    2. PHRASAL VS. TRANSITIVE:
    - Keep particles ONLY if they are part of a Phrasal Verb (e.g., "carry out", "set up").
    - Discard all Objects, Nouns, Adjectives, and Adverbs. A verb should never be followed by a noun in your output.
    3. STRIP MODIFIERS: Remove all "junk" surrounding the verb, including articles (a, an, the) and prepositional phrases that act as objects.
    4. LEMMATIZE: Always return the verb in its dictionary infinitive form (e.g., "fund").

    NEGATIVE CONSTRAINTS:
    - No nouns in the output (unless part of a rare compound verb).
    - No auxiliary/modal verbs if a main action follows.

    Respond ONLY with a JSON array:
    [
    {{"text": "...", "verb": "..."}}
    ]

    Phrases:
    {numbered}
    """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        results = json.loads(text)

        return results

    verbs = (
        inputs_totales_por_rol["Verbo"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    verbs = list(verbs)
    len(list(verbs))

    api_keys = ["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", "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"]  
    clients = [Groq(api_key=k) for k in api_keys]

    # Divido la lista de verbos
    def split_list(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

    verb_parts = split_list(verbs, len(clients))

    # Divido en partes de tamaño 50
    def chunk_list(lst, size=50):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    all_results = []

    # Aplico el prompt en partes, porque tiene limite mi LLM
    for idx, (client, verb_subset) in enumerate(zip(clients, verb_parts)):
        print(f"\n Procesando parte {idx+1} con API key {idx+1}")

        chunks = list(chunk_list(verb_subset, 50))

        for i, chunk in enumerate(chunks):
            try:
                results = extract_main_verbs_batch(chunk, client)
                all_results.extend(results)

                print(f"Parte {idx+1} → Batch {i+1}/{len(chunks)}")

            except Exception as e:
                print(f"Error en parte {idx+1}, batch {i}: {e}")
                continue
    
    # Guardo un df los resultado encontrados
    verbos_procesados = pd.DataFrame(all_results)
    verbos_procesados = verbos_procesados.rename(columns={
        "text": "Verbo",
        "verb": "Verbo_Limpio"
    })

    # Uno los verbos normalizados al df principal
    inputs_totales_por_rol = inputs_totales_por_rol.merge(
        verbos_procesados,
        on="Verbo",
        how="left"
    )

    # Me quedo con verbos normalizados que dieron NaN
    verbs_missing = (
        inputs_totales_por_rol[
            inputs_totales_por_rol["Verbo_Limpio"].isna()
        ]["Verbo"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    verbs_missing = list(verbs_missing)

    # Vuelvo pasar por LLM estos verbos mal procesados
    missing_results = []
    client = Groq(api_key="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")

    def chunk_list(lst, size=50):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    for chunk in chunk_list(verbs_missing, 20):
        try:
            results = extract_main_verbs_batch(chunk, client)

            mapping = {r["text"]: r["verb"] for r in results}

            for v in chunk:
                missing_results.append({
                    "Verbo": v,
                    "Verbo_Limpio": mapping.get(v, None)
                })

        except Exception as e:
            print(f"Error en batch: {e}")

    # Me guardo los resultados de los verbos erroneos
    verbos_no_procesados = pd.DataFrame(missing_results)

    # Si no hay verbos erroneos, devuelvo df vacio
    if verbos_no_procesados.empty:
        verbos_no_procesados = pd.DataFrame([{'Verbo': None, 'Verbo_Limpio': None}])
        print("El DataFrame 'verbos_no_procesados' está vacío. Columnas inicializadas.")
    else:
        # Si hay verbos erroneos, vuelvo a usar un LLM
        mask = verbos_no_procesados['Verbo_Limpio'].isna()
        verbo_sucio = verbos_no_procesados.loc[mask, 'Verbo'].tolist()

        if verbo_sucio:
            resultados = extract_main_verbs_batch(verbo_sucio, client)
            verbos_ordenados = [item['verb'] for item in resultados]
            verbos_no_procesados.loc[mask, 'Verbo_Limpio'] = verbos_ordenados
            print(f"Actualización completada: {len(verbos_ordenados)} verbos procesados")
        else:
            print("No hay filas con Verbo_Limpio pendiente")


    # Fuciono los verbos correctos 
    df1 = verbos_procesados.set_index('Verbo')
    df2 = verbos_no_procesados.set_index('Verbo')
    verbos_procesados_finales = df1.combine_first(df2).reset_index()
    
    # Elimino la columna transitoria
    inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Verbo_Limpio"])

    # Reemplazo los verbos brutos por los normalizados
    inputs_totales_por_rol = inputs_totales_por_rol.merge(
        verbos_procesados_finales,
        on="Verbo",
        how="left"
    )

    inputs_totales_por_rol["Verbo"] = inputs_totales_por_rol["Verbo_Limpio"]
    inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Verbo_Limpio"])
    
    
    # Normalizacion de contexto
    # Normalizo con un prompt en ingles de LLM
    def extract_main_context_batch(phrases, client):

        if isinstance(phrases, str):
            phrases = [phrases]

        numbered = "\n".join([f"{i+1}. {p}" for i, p in enumerate(phrases)])

        prompt = f"""You are an expert in semantic abstraction and information compression.

    Your goal is to transform each input into ONE SINGLE, HIGH-LEVEL CONTEXT.

    CORE LOGIC:
    1. GENERALIZE: Convert specific phrases into a broad, generic concept.
    - Example: "stimulus plan, obamacare marketplaces" → "government policy"
    - Example: "brexit regulations, tariffs" → "trade regulation"

    2. MERGE MULTIPLE ELEMENTS:
    - If multiple items are separated by commas, DO NOT keep them separate.
    - Combine them into ONE unified concept.

    3. REMOVE SPECIFICITY:
    - Eliminate names, brands, events, and detailed descriptions.
    - Avoid long phrases. Keep it short (1–3 words ideally).

    4. ABSTRACT UPWARD:
    - Always move to a higher-level category (industry, policy, operations, demand, supply, regulation, etc.).

    5. SINGLE OUTPUT ONLY:
    - Even if the input tenga varios elementos → SOLO UN contexto final.

    6. LOWERCASE OUTPUT:
    - The context MUST be entirely in lowercase.

    NEGATIVE CONSTRAINTS:
    - No commas in output
    - No long phrases
    - No specific entities (e.g., "Obamacare", "Brexit")
    - No duplication of input wording unless already abstract
    - Output must be lowercase only

    OUTPUT FORMAT (JSON array):
    [
    {{"text": "...", "context": "..."}}
    ]

    Inputs:
    {numbered}
    """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        results = json.loads(text)

        return results

    api_keys = ["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", "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", "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", "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"]  # puedes añadir más
    clients = [Groq(api_key=k) for k in api_keys]

    contexts = (
        inputs_totales_por_rol["Contexto"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    contexts = list(contexts)
    
    # Divido la lista de contextos
    def split_list(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

    context_parts = split_list(contexts, len(clients))

    # Divido en partes tamaño 50
    def chunk_list(lst, size=50):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    # Aplico el prompt en partes, porque tiene limite mi LLM
    all_results = []

    for idx, (client, context_subset) in enumerate(zip(clients, context_parts)):
        print(f"\n Procesando parte {idx+1} con API key {idx+1}")

        chunks = list(chunk_list(context_subset, 50))

        for i, chunk in enumerate(chunks):
            try:
                results = extract_main_context_batch(chunk, client)
                all_results.extend(results)

                print(f"Parte {idx+1} → Batch {i+1}/{len(chunks)}")

            except Exception as e:
                print(f"Error en parte {idx+1}, batch {i}: {e}")
                continue
    
    # Guardo un df los resultado encontrados
    contextos_procesados = pd.DataFrame(all_results)
    contextos_procesados = contextos_procesados.rename(columns={
        "text": "Contexto",
        "context": "Contexto_Limpio"
    })

    # Uno los contextos normalizados al df principal
    inputs_totales_por_rol = inputs_totales_por_rol.merge(
        contextos_procesados,
        on="Contexto",
        how="left"
    )

    # Me quedo con contextos normalizados que dieron NaN
    contexts_missing = (
        inputs_totales_por_rol[
            inputs_totales_por_rol["Contexto_Limpio"].isna()
        ]["Contexto"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    contexts_missing = list(contexts_missing)

    # Vuelvo pasar por LLM estos contextos mal procesados
    missing_results = []
    client = Groq(api_key="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")

    def chunk_list(lst, size=50):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    for chunk in chunk_list(contexts_missing, 20):
        try:
            results = extract_main_context_batch(chunk, client)

            # IMPORTANTE: mapear por texto
            mapping = {r["text"]: r["context"] for r in results}

            for v in chunk:
                missing_results.append({
                    "Contexto": v,
                    "Contexto_Limpio": mapping.get(v, None)
                })

        except Exception as e:
            print(f"Error en batch: {e}")

    # Me guardo los resultados de los contextos erroneos
    contextos_no_procesados = pd.DataFrame(missing_results)

    # Si no hay contextos erroneos, devuelvo df vacio
    if contextos_no_procesados.empty:
        contextos_no_procesados = pd.DataFrame([{'Contexto': None, 'Contexto_Limpio': None}])
        print("El DataFrame 'contextos_no_procesados' está vacio. Columnas inicializadas.")
    else:
        # Si hay contextos erroneos, vuelvo a usar un LLM
        mask = contextos_no_procesados['Contexto_Limpio'].isna()
        contexto_sucio = contextos_no_procesados.loc[mask, 'Contexto'].tolist()

        if contexto_sucio:
            resultados = extract_main_context_batch(contexto_sucio, client)
            contextos_ordenados = [item['context'] for item in resultados]
            contextos_no_procesados.loc[mask, 'Contexto_Limpio'] = contextos_ordenados
            print(f"Actualización completada: {len(contextos_ordenados)} contextos procesados.")
        else:
            print("No hay filas con Contexto_Limpio pendiente")

    # Fuciono los contextos correctos 
    df1 = contextos_procesados.set_index('Contexto')
    df2 = contextos_no_procesados.set_index('Contexto')
    contextos_procesados_finales = df1.combine_first(df2).reset_index()

    # Elimino la columna transitoria
    inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Contexto_Limpio"])

    # Reemplazo los contextos brutos por los normalizados
    inputs_totales_por_rol = inputs_totales_por_rol.merge(
        contextos_procesados_finales,
        on="Contexto",
        how="left"
    )

    inputs_totales_por_rol["Contexto"] = inputs_totales_por_rol["Contexto_Limpio"]
    inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Contexto_Limpio"])


    # Transformo el df a una fila por ticker
    df_tickers = inputs_totales_por_rol.explode("Tickers Mapeados").copy()

    # Quito los ticker nulos
    df_tickers = df_tickers[df_tickers["Tickers Mapeados"].notna()]

    # Obtengo el maximo de eventos por noticias
    max_eventos = df_tickers.groupby("Fila Noticia")["Evento"].max()

    # Si dentro de un evento, hay 2 tickers, se da prioridad al ticker mapeado de la noticia empresarial
    # porque suele ser mas preciso
    def resolver_grupo(g):
        if g["Mencionado"].any():
            g = g[g["Mencionado"] == True]

        return pd.Series({
            "Eventos": max_eventos[g["Fila Noticia"].iloc[0]],
            "Agente": (g["Rol"] == "agente").any(),
            "Paciente": (g["Rol"] == "paciente").any(),
            "Afectado": (g["Rol"] == "afectado").any(),
            "Verbo": g["Verbo"].iloc[0],
            "Contexto": g["Contexto"].iloc[0],
            "Status": g["Status"].iloc[0],
            "Mencionado": g["Mencionado"].any()
        })

    # Aplico la eliminacion de tickers repetidos por evento
    inputs_gramatical = (
        df_tickers
        .groupby(["Fila Noticia", "Evento", "Tickers Mapeados"])
        .apply(resolver_grupo)
        .reset_index()
    )

    # Me cargo la columna Evento, porque ya tengo maximo eventos
    inputs_gramatical = inputs_gramatical.drop(columns=['Evento'])

    # Donde Mencionado igual False, pongo False en Agente, Paciente y Afectado. No hago analisi de roles para noticias sectoriales
    inputs_gramatical.loc[inputs_gramatical["Mencionado"] == False, ["Agente", "Paciente", "Afectado"]] = False

    # Elimino la columna Mencionado
    inputs_gramatical = inputs_gramatical.drop(columns=["Mencionado"])

    # De mi columna con toda la noticia unida, extraigo solo el titular y contenido
    def extraer_noticia(text):
        if pd.isna(text) or not isinstance(text, str):
            return None, None

        # Extraigo el titulo
        titulo_match = re.search(r"\[TITLE\]\s*(.*?)(?=\s*\[|$)", text, re.DOTALL)
        titulo = titulo_match.group(1).strip() if titulo_match else None

        # Extraigo el contenido
        contenido_match = re.search(r"\[CONTENT\]\s*(.*)", text, re.DOTALL)

        if contenido_match:
            contenido = contenido_match.group(1).strip()
        else:
            # Si no hay contenido, repito el titulo
            contenido = titulo

        return titulo, contenido

    # Aplico y desempaqueto la extraccion de titulo y contenido
    resultados = noticias_input_filtrado['Full Text'].apply(extraer_noticia)
    titulos, contenidos = zip(*resultados)

    # Las inserto en las posiciones 3 y 4 del df
    noticias_input_filtrado.insert(2, "Titulo Noticia", titulos)
    noticias_input_filtrado.insert(3, "Contenido Noticia", contenidos)

    # Obtengo y etiqueto en 5 posibles valores la hora de publicacion. Al final no lo uso
    def get_time_bucket(dt):
        h = dt.hour
        m = dt.minute

        total_minutes = h * 60 + m

        if (total_minutes >= 0 and total_minutes <= 9*60+29) or (total_minutes >= 16*60+1):
            return "extra oficial"
        elif 9*60+30 <= total_minutes <= 11*60+59:
            return "mañana"
        elif 12*60 <= total_minutes <= 13*60+59:
            return "medio dia"
        elif 14*60 <= total_minutes <= 16*60:
            return "tarde"
        else:
            return "extra oficial"  

    # Aseguro el formato datetime de Date
    noticias_input_filtrado["Date"] = pd.to_datetime(noticias_input_filtrado["Date"])

    # Extraigo el dia de la semana de la publicacion de la noticia
    day_col = noticias_input_filtrado["Date"].dt.day_name()
    # Extraigo la etiqueta de hora de la publicacion de la noticia
    time_bucket_col = noticias_input_filtrado["Date"].apply(get_time_bucket)

    # Elimino Week Date y Date Time si existen
    for col in ["Week Day", "Date Time"]:
        if col in noticias_input_filtrado.columns:
            noticias_input_filtrado.drop(columns=[col], inplace=True)

    # Inserto el dia de la semana y la hora etiquetada en df de noticias
    noticias_input_filtrado.insert(1, "Week Day", day_col)
    noticias_input_filtrado.insert(2, "Date Time", time_bucket_col)

    # Inserto ordenadamente las columnas de fecha, dia de semana, hora, titulo y contenido de notica en el df final
    date = noticias_input_filtrado["Date"]
    day_map = noticias_input_filtrado["Week Day"]
    time_map = noticias_input_filtrado["Date Time"]
    title_map = noticias_input_filtrado["Titulo Noticia"]
    content_map = noticias_input_filtrado["Contenido Noticia"]

    # Fecha
    inputs_gramatical.insert(
        1,
        "Date",
        inputs_gramatical["Fila Noticia"].map(date)
    )
    # Dia de semana
    inputs_gramatical.insert(
        2,
        "Week Day",
        inputs_gramatical["Fila Noticia"].map(day_map)
    )
    # Hora
    inputs_gramatical.insert(
        3,
        "Date Time",
        inputs_gramatical["Fila Noticia"].map(time_map)
    )
    # Titulo
    inputs_gramatical.insert(
        4,
        "Titulo Noticia",
        inputs_gramatical["Fila Noticia"].map(title_map)
    )
    # Contenido
    inputs_gramatical.insert(
        5,
        "Contenido Noticia",
        inputs_gramatical["Fila Noticia"].map(content_map)
    )


    # Colpaso las noticias con mas de un evento mediante: primer valor, maximo y lista
    inputs_gramatical = inputs_gramatical.groupby(["Fila Noticia", "Tickers Mapeados"]).agg({

        "Date": "first",
        "Week Day": "first",
        "Date Time": "first",
        "Titulo Noticia": "first",
        "Contenido Noticia": "first",
        "Eventos": "max",

        "Agente": "max",
        "Paciente": "max",
        "Afectado": "max",

        "Verbo": list,
        "Contexto": list,
        "Status": list,

    }).reset_index()

    # Ordeno por fecha y fila noticia
    inputs_gramatical['Date'] = pd.to_datetime(inputs_gramatical['Date'])
    inputs_gramatical = inputs_gramatical.sort_values(['Date', 'Fila Noticia']).reset_index(drop=True)


    # Codificacion de inputs
    # Codifico con booleano a los 3 roles
    boolean_cols = ["Agente", "Paciente", "Afectado"]
    boolean_data = inputs_gramatical[boolean_cols].astype(float)

    # Codifico con one hot encoder a dia de semana y hora
    categorical_cols = ["Week Day", "Date Time"]

    # Configuro el encoder para que devuelva un df
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.set_output(transform="pandas")

    encoded_data = encoder.fit_transform(inputs_gramatical[categorical_cols])

    # Codigo status con multi hot encoder, porque tengo a veces 2 status por los eventos colpasados
    status_map = {
        "speculative": 0,
        "in progress": 1,
        "confirmed neutral": 2,
        "confirmed positive": 3,
        "confirmed negative": 4
    }

    # Generador del multi hot enconding
    def multi_hot_status(values, n_classes=5):
        vec = np.zeros(n_classes)
        for s in values:
            vec[status_map[s]] = 1
        return vec

    status_array = np.stack(inputs_gramatical["Status"].apply(multi_hot_status))

    # Creo el df de status resultante
    status_cols = [f"Status_{k}" for k in status_map.keys()]
    status_data = pd.DataFrame(status_array, columns=status_cols, index=inputs_gramatical.index)

    # Codigo maximo eventos pasandolo a decimal, porque ya es un numero y si importa el orden
    numeric_cols = ["Eventos"]
    numeric_data = inputs_gramatical[numeric_cols].astype(float)


    # Detecto si hay GPU disponible, sino CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Codifico titulo, notiica, verbo y contexto con embedding
    # Uso Finbert para el embedding
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert").to(device)
    model.eval() # Poner en modo evaluación

    # Genero el embedding para titulo y noticia
    def get_embeddings(text_list, batch_size=64): 
        embeddings_list = []

        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device) 

            with torch.inference_mode():
                outputs = model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.append(cls_embeddings)

        return np.vstack(embeddings_list)

    # Aplico el embedding a titulos
    title_embeddings_matrix = get_embeddings(inputs_gramatical["Titulo Noticia"].tolist())
    title_embedding_data = pd.DataFrame(title_embeddings_matrix)
    # Cambio los nombres porque son numeros ahora y necesito reconocer a que input hace referencia
    title_embedding_data.columns = [f'Titulo_{i}' for i in range(title_embedding_data.shape[1])]

    # Aplico el embedding a contenidos
    content_embeddings_matrix = get_embeddings(inputs_gramatical["Contenido Noticia"].tolist())
    content_embedding_data = pd.DataFrame(content_embeddings_matrix)
    # Cambios los nombres son numeros ahora y necesito reconocer a que input hacer referencia
    content_embedding_data.columns = [f'Contenido_{i}' for i in range(content_embedding_data.shape[1])]

    # Generador de emebdding para verbo y contexto
    def get_embeddings_ndarray(verbo_column, batch_size=64):
        model.to(device)
        model.eval()

        all_verbs = []
        counts = []

        for val in verbo_column:
            if isinstance(val, (np.ndarray, list)):
                lista = [str(word).strip() for word in val if str(word).strip()]
            elif isinstance(val, str):
                lista = [w.strip() for w in val.replace('[','').replace(']','').split(',') if w.strip()]
            else:
                lista = []

            all_verbs.extend(lista if lista else [""]) 
            counts.append(len(lista) if lista else 1)

        embeddings_list = []
        for i in range(0, len(all_verbs), batch_size):

            batch = all_verbs[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(cls_emb)

        all_embeddings = np.vstack(embeddings_list)

        final_mean_embeddings = []
        current_idx = 0
        
        for count in counts:
            row_embs = all_embeddings[current_idx : current_idx + count]
            final_mean_embeddings.append(np.mean(row_embs, axis=0))
            current_idx += count

        return np.vstack(final_mean_embeddings)

    # Aplico el embedding a verbos
    verbo_embeddings_matrix = get_embeddings_ndarray(inputs_gramatical['Verbo'])
    verbo_embeddings = pd.DataFrame(verbo_embeddings_matrix)
    # Cambios los nombres son numeros ahora y necesito reconocer a que input hacer referencia
    verbo_embeddings.columns = [f'Verbo_{i}' for i in range(verbo_embeddings.shape[1])]

    # Aplico el embedding a contextos
    contexto_embeddings_matrix = get_embeddings_ndarray(inputs_gramatical['Contexto'])
    contexto_embeddings = pd.DataFrame(contexto_embeddings_matrix)
    # Cambios los nombres son numeros ahora y necesito reconocer a que input hacer referencia
    contexto_embeddings.columns = [f'Contexto_{i}' for i in range(contexto_embeddings.shape[1])]


    # Listado de todos lo inputs codificados
    dataframes = [
        title_embedding_data,
        content_embedding_data,
        verbo_embeddings,   
        contexto_embeddings, 
        boolean_data,        
        numeric_data,       
        encoded_data,       
        status_data,         
    ]

    # Df con los inputs codificados
    df_inputs = pd.concat(dataframes, axis=1)

# Salto todo el proceso de limpieza y devuelvo un df vacio
else:
    print("El df 'inputs_totales_por_rol' esta vacio desde el inicio. Generando df_inputs vacio.")
    df_inputs = pd.DataFrame() 


# Inferencia de los inputs
# Si no hay inputs codificados, devuelvo señales vacias
if df_inputs.empty:
    print("El df df_inputs esta vacio. Creo un df de señales vacio")
    seynales_modelo = pd.DataFrame(columns=["Fila Noticia", "Tickers Mapeados", "Date", "Prob_up", "Pred_label"])

else:
    # Divido los inputs
    X_title = df_inputs.iloc[:, 0:768].values
    X_content = df_inputs.iloc[:, 768:1536].values
    X_verb = df_inputs.iloc[:, 1536:2304].values
    X_context = df_inputs.iloc[:, 2304:3072].values
    columnas_meta = [
        'Agente', 
        'Paciente', 
        'Afectado', 
        'Eventos', 
        'Status_speculative', 
        'Status_in progress',      
        'Status_confirmed neutral', 
        'Status_confirmed positive', 
        'Status_confirmed negative'  
    ]
    X_meta = df_inputs[columnas_meta].values

    # Estandarizo
    scaler_title = StandardScaler()
    scaler_content = StandardScaler()
    scaler_verb = StandardScaler()
    scaler_context = StandardScaler()
    scaler_meta = StandardScaler()

    X_title = scaler_title.fit_transform(X_title)
    X_content = scaler_content.fit_transform(X_content)
    X_verb = scaler_verb.fit_transform(X_verb)
    X_context = scaler_context.fit_transform(X_context)
    X_meta = scaler_meta.fit_transform(X_meta)

    # Convierto a tensor de Pytorch para la red neuronal
    X_title = torch.tensor(X_title, dtype=torch.float32)
    X_content = torch.tensor(X_content, dtype=torch.float32)
    X_verb = torch.tensor(X_verb, dtype=torch.float32)
    X_context = torch.tensor(X_context, dtype=torch.float32)
    X_meta = torch.tensor(X_meta, dtype=torch.float32)

    batch_size = 512

    # Empaqueto los tensores de cada dataset
    test_loader = DataLoader(
        TensorDataset(X_title, X_content, X_verb, X_context, X_meta),
        batch_size=batch_size,
        shuffle=False  
    )
    
    # Arquitectura del modelo
    class MultiInputModel(nn.Module):

        def __init__(self):

            super().__init__()
            
            # Hiperparameto del modelo optimo: NN_41
            hidden_dim = 48
            dropout = 0.15

            small_dim = hidden_dim // 2  

            # Titulo
            self.title_branch = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            # Contenido
            self.content_branch = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            # Verbo
            self.verb_branch = nn.Sequential(
                nn.Linear(768, small_dim),
                nn.ReLU(),
                nn.Linear(small_dim, small_dim),
                nn.ReLU()
            )

            # Contexto
            self.context_branch = nn.Sequential(
                nn.Linear(768, small_dim),
                nn.ReLU(),
                nn.Linear(small_dim, small_dim),
                nn.ReLU()
            )

            # Metadatos
            self.meta_branch = nn.Sequential(
                nn.Linear(9, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )

            # Comnbinacion dimensiones
            combined_dim = (
                hidden_dim
                + hidden_dim
                + small_dim
                + small_dim
            )

            # Gate
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.Sigmoid()
            )

            # Cabeza
            total_dim = combined_dim * 2 + 32
            
            self.head = nn.Sequential(
                nn.Linear(total_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )

        # Forward pass
        def forward(
            self,
            x_title,
            x_content,
            x_verb,
            x_context,
            x_meta
        ):

            x_title = F.normalize(x_title, dim=1)
            x_content = F.normalize(x_content, dim=1)
            x_verb = F.normalize(x_verb, dim=1)
            x_context = F.normalize(x_context, dim=1)

            t = self.title_branch(x_title)
            c = self.content_branch(x_content)
            v = self.verb_branch(x_verb)
            cx = self.context_branch(x_context)
            m = self.meta_branch(x_meta)

            combined = torch.cat(
                [t, c, v, cx],
                dim=1
            )

            gate = self.gate(combined)

            gated = combined * gate

            x = torch.cat(
                [combined, gated, m],
                dim=1
            )

            return self.head(x)

    # Configur el cpu
    device = torch.device("cpu")

    # Intancio el modelo
    model = MultiInputModel()

    # Cargo los pesos del modelo optimo
    model.load_state_dict(torch.load(buffer, map_location=device))

    # Evaluo
    model.to(device)
    model.eval()

    model.eval()
    all_probs_test = []

    # Inferencia
    with torch.inference_mode():
        for x_title, x_content, x_verb, x_context, x_meta in test_loader:

            # Muevo a cpu
            x_title = x_title.to(device)
            x_content = x_content.to(device)
            x_verb = x_verb.to(device)
            x_context = x_context.to(device)
            x_meta = x_meta.to(device)

            # Predigo
            outputs_test = model(x_title, x_content, x_verb, x_context, x_meta)
            probs_test = torch.sigmoid(outputs_test)

            all_probs_test.extend(probs_test.cpu().numpy())

    # Convierto a array
    predicciones_finales = np.array(all_probs_test).flatten()

    # Convierto a numpy
    probs = np.array(all_probs_test).flatten()

    # Etiquta bajo un umbral de 0.5
    y_pred = (probs >= 0.5).astype(int)

    # Construyo el df de probabilidades y etiqueta
    df_results = pd.DataFrame({
        "Prob_up": probs,
        "Pred_label": y_pred
    })

    # Genero el df de señales para sistema de trading
    seynales_modelo = inputs_gramatical[["Fila Noticia", "Tickers Mapeados", "Date"]].merge(
        df_results,
        left_index=True,  
        right_index=True
    )


# Exporto la señales a la tabla de dynamodb model_signals
print("Iniciando actualización de la tabla 'model_signals'")
tabla_signal = dynamodb.Table('model_signals')

# Borro todo lo de la tabla
response = tabla_signal.scan(ProjectionExpression='ID')
items_a_borrar = response.get('Items', [])

while 'LastEvaluatedKey' in response:
    response = tabla_signal.scan(ExclusiveStartKey=response['LastEvaluatedKey'], ProjectionExpression='ID')
    items_a_borrar.extend(response.get('Items', []))

if items_a_borrar:
    print(f"Borrando {len(items_a_borrar)} registros antiguos...")
    with tabla_signal.batch_writer() as batch:
        for item in items_a_borrar:
            batch.delete_item(Key={'ID': item['ID']})
    print("Tabla vaciada por completo")
else:
    print("La tabla ya estaba vacia")

# Subo las señales, si es que hay o no
if not seynales_modelo.empty:
    print(f"Preparando {len(seynales_modelo)} nuevos registros para subir")
    
    # Creo que la columna ID porque necesita la tabla esa key partition
    df_upload = seynales_modelo.reset_index().rename(columns={'index': 'ID'})
    
    # Transformo ID como entero
    df_upload['ID'] = df_upload['ID'].astype(int)
    
    # Transformo fecha a string
    df_upload['Date'] = df_upload['Date'].astype(str)
    
    # Formateo de tipos para AWS
    for col in df_upload.select_dtypes(include=[float, 'float64']).columns:
        df_upload[col] = df_upload[col].apply(lambda x: Decimal(str(x)) if pd.notnull(x) else None)
        
    for col in df_upload.select_dtypes(include=[int, 'int64']).columns:
        df_upload[col] = df_upload[col].apply(lambda x: int(x) if pd.notnull(x) else None)
        
    registros = df_upload.to_dict(orient='records')
    with tabla_signal.batch_writer() as batch:
        for reg in registros:
            reg_limpio = {k: v for k, v in reg.items() if v is not None}
            batch.put_item(Item=reg_limpio)
            
    print("Nuevos datos subidos con éxito a 'model_signals'.")
else:
    print("El DataFrame de señales esta vacio. La tabla quedara vacia de forma intencional.")


