# !pip install spacy[transformers]
# !python -m spacy download en_core_web_trf
# !pip install ctransformers sentencepiece
# !pip install rapidfuzz
# !pip install groq

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
from datetime import datetime, timedelta
from datetime import time as dt_time
from collections import defaultdict, Counter

# Librerías específicas de NLP y ML
import spacy
from ctransformers import AutoModelForCausalLM
from rapidfuzz import process, fuzz
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset



# Configuración de AWS (Boto3 usará las variables de entorno de GitHub Actions)
dynamodb = boto3.resource('dynamodb', region_name='us-east-1') # Cambia a tu región
s3 = boto3.client('s3')

def get_table_df(table_name):
    table = dynamodb.Table(table_name)
    response = table.scan()
    data = response['Items']
    
    # Manejo de paginación si la tabla es grande
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    
    return pd.DataFrame(data)

# --- 1. LEER TABLAS DE DYNAMODB ---
composicion_sp500_actualizado = get_table_df('historic_composition_sp500')
wikipedia_actualizado = get_table_df('update_wikipedia_keys')
morningstar = get_table_df('morningstar_classification')
empresas_sectores_morningstar = get_table_df('companys_morningstar_sectors')
precios_cierre_sesion = get_table_df('sesion_close_prices')
volumenes_sesion = get_table_df('sesion_volumes')
pendientes_ayer = get_table_df('pending_trade')
# Descargamos el histórico para extraer el saldo de caja acumulado
historico_balance = get_table_df('daily_balance')

# --- 2. DESCARGAR PESOS DESDE S3 ---
bucket_name = 'trained-neuronal-model'
key = 'NN_41_weights.pth'

print(f"Descargando {key} desde S3...")
buffer = io.BytesIO()
s3.download_fileobj(bucket_name, key, buffer)
buffer.seek(0)


# INICIO DEL CODIGO

API_KEY = "1sO8gR66gfd7wAU8HaGYxqDMGhROZH0VsLMBY7WEXzjF6VgV"

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
                time.sleep(sleep_time) # Recomendado 6.0 para evitar bloqueos
                break

            elif r.status_code in [429, 500]:
                intento += 1
                wait = 12 * intento # Aumentamos el tiempo de espera en errores
                print(f"Error {r.status_code} (Rate Limit/Server) en pág {page}, reintento {intento}/{reintentos}...")
                time.sleep(wait)
            else:
                print(f"Error HTTP {r.status_code} en página {page}: {r.text}")
                return finalizar_dataframe(noticias)

    return finalizar_dataframe(noticias)

def finalizar_dataframe(noticias):
    df = pd.DataFrame(noticias)
    if not df.empty:
        # Ajuste de horario a Nueva York (Resuelve lo de las 00:58 Z)
        df["Date"] = (
            pd.to_datetime(df["Date"], utc=True, format='ISO8601')
            .dt.tz_convert(ZoneInfo("America/New_York"))
            .dt.tz_localize(None)
        )
        df = df.sort_values("Date").reset_index(drop=True)
    return df


hoy = datetime(2026, 5, 15)

# 2. Calculamos las fechas relativas
start = (hoy - timedelta(days=1)).strftime("%Y-%m-%d") # 2026-05-05
end = (hoy + timedelta(days=1)).strftime("%Y-%m-%d")   # 2026-05-07

# 3. Llamada a tu función con las variables programadas
df_nyt = descargar_nyt_periodo(start, end, query="news")


# 1. Convertir la columna Date a formato datetime si aún no lo está
df_nyt['Date'] = pd.to_datetime(df_nyt['Date'])


# Ahora cambia estas dos líneas:
start_period = datetime.combine(hoy.date() - pd.Timedelta(days=1), dt_time(18, 0, 0))
end_period = datetime.combine(hoy.date(), dt_time(18, 0, 0))

# 3. Filtrar el DataFrame
# Usamos >= y <= para incluir exactamente las 18:00
last_news = df_nyt[(df_nyt['Date'] >= start_period) & (df_nyt['Date'] <= end_period)].copy()

# Ordenar por fecha para verificar
last_news = last_news.sort_values(by='Date')


# Elimino filas con Na
last_news = last_news.dropna(subset=['Section', 'Title'])

# Reviso y elimino duplicados
duplicate_news = last_news[last_news.duplicated(subset=["Title", "Content"], keep=False)]

# 1. Eliminamos los duplicados manteniendo solo la primera fila encontrada
unique_news = last_news.drop_duplicates(subset=["Title", "Content"], keep='first').reset_index(drop=True)

black_list = [
    'Crosswords & Games', 'Gameplay', 'Movies', 'Arts', 'Theater', 'Books',
    'Book Review', 'Briefing', 'Today’s Paper', 'Times Insider', 'Corrections',
    'Admin', 'Reader Center', 'Homepage', 'Video', 'Multimedia/Photos',
    'The Learning Network', 'Education', 'Parenting', 'Well', 'At Home',
    'Smarter Living', 'Neediest Cases', 'Giving', 'Sports', 'Obituaries',
    'Weather', 'Travel', 'Podcasts', 'En español', 'en Español', 'New York',
    'International Home', 'Lens', 'Universal', 'Home & Garden'
]

relevant_last_news = last_news[~last_news['Section'].isin(black_list)]

# 1. Convertimos el objeto en una copia independiente y reseteamos el índice de una vez
relevant_last_news = relevant_last_news.reset_index(drop=True)

# 2. Ahora creamos la columna (ya no dará advertencia)
relevant_last_news["Full Text"] = (
    "[SECTION] " + relevant_last_news["Section"] + "\n" +
    "[TITLE] " + relevant_last_news["Title"] + "\n" +
    "[CONTENT] " + relevant_last_news["Content"]
)

nlp = spacy.load("en_core_web_trf")

def extract_entities(text):
    doc = nlp(text)

    entities = {"ORG": set(), "PERSON": set(), "PRODUCT": set()}

    for ent in doc.ents:
        if ent.label_ in ("ORG", "PERSON", "PRODUCT"):
            entities[ent.label_].add(ent.text)

    return entities

def run_ner_and_process(df, chunk_size=5):
    total = len(df)
    n_chunks = math.ceil(total / chunk_size)

    print(f"Total noticias: {total} → {n_chunks} bloques de ~{chunk_size}")

    # Lista para acumular los DataFrames de cada bloque
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

        # Creamos el DataFrame del bloque actual
        df_chunk = pd.DataFrame({
            "Date": chunk["Date"].values,
            "Full Text": chunk["Full Text"].values,
            "Organization": [", ".join(sorted(x)) if x else None for x in resultados["ORG"]],
            "Person": [", ".join(sorted(x)) if x else None for x in resultados["PERSON"]],
            "Product": [", ".join(sorted(x)) if x else None for x in resultados["PRODUCT"]],
            "All Names": [x if x else None for x in resultados["ALL_ENTITIES"]],
        })

        # Lo guardamos en la lista en lugar de a un archivo
        all_chunks.append(df_chunk)

    # Unimos todos los bloques en un solo DataFrame
    df_final = pd.concat(all_chunks, ignore_index=True)
    return df_final

# Uso
noticias_ner_mapeados = run_ner_and_process(relevant_last_news, chunk_size=5)


# 1. CARGA DEL MODELO (Todo en uno)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=0,  # Forzamos CPU
    context_length=2048
)


#PASO 1: Tu función build_prompt (tu prompt actual sin cambios)
def build_prompt(news_text):
    prompt = f"""
<s>[INST]
You are a financial and economic event extractor for quantitative investment analysis.

Your job is to extract ONLY explicit economic or financial information.

------------------------------------------------------------
1) FINANCIAL EVENT DETECTION (STRICT BUT COMPLETE)
------------------------------------------------------------
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

------------------------------------------------------------
2) ECONOMIC CHANNELS (INDUSTRY MAPPING REQUIRED)
------------------------------------------------------------
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

------------------------------------------------------------
3) COUNTRIES / LOCATIONS
------------------------------------------------------------
Extract locations only when clearly identifiable.
- Include continents, countries, regions, cities mentioned in the article.
- Map political leaders to their country (e.g., Trump → United States).
- Locations that appear in text but are not countries (e.g., Latin America, Europe) should also be included.
- Do NOT place any location in economic_channels.

If no location is explicitly mentioned, return an empty list.

------------------------------------------------------------
4) OUTPUT FORMAT (MANDATORY)
------------------------------------------------------------
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


# 2. FUNCIÓN DE EXTRACCIÓN (Adaptada a ctransformers)
def extract_events_batch(news_texts):
    """Procesa los textos uno a uno (lo más estable en CPU)."""
    all_results = []
    for text in news_texts:
        # Importante: Mistral requiere el formato [INST] [/INST]
        prompt = f"[INST] {build_prompt(text)} [/INST]"

        # El modelo genera el texto directamente
        respuesta = model(prompt, max_new_tokens=250, temperature=0)
        all_results.append(respuesta)
    return all_results

# 3. LIMPIEZA DE SALIDA
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

# 4. PROCESADOR PRINCIPAL (Todo en memoria)
def procesar_todo_en_variable(df_original):
    total = len(df_original)
    resultados_lista = []
    inicio = time.time()

    print(f"Procesando {total} noticias en CPU...")

    for i, (idx, row) in enumerate(df_original.iterrows()):
        # Procesamos de 1 en 1 para no saturar la RAM
        texto_noticia = [row["Full Text"]]
        respuesta_raw = extract_events_batch(texto_noticia)

        # Limpiamos el JSON de la respuesta
        datos_limpios = parse_llm_output_clean(respuesta_raw[0])

        # Guardamos en la lista
        resultados_lista.append({
            "Date": row["Date"],
            "Full Text": row["Full Text"],
            **datos_limpios
        })

        # if (i + 1) % 5 == 0:
        #     print(f"Progreso: {i+1}/{total} noticias finalizadas...")
        print(f"la noticia {i+1}/{total} ha terminado")

    df_final = pd.DataFrame(resultados_lista)
    print(f"\n¡Listo! Tiempo total: {time.time() - inicio:.2f}s")
    return df_final

# PARA USARLO:
noticias_sectores_mapeados = procesar_todo_en_variable(relevant_last_news)

noticias_keywords_mapeados = noticias_sectores_mapeados.copy()
noticias_keywords_mapeados['All Names'] = noticias_ner_mapeados['All Names']


COLUMNAS_KEYWORDS = ["Company Name", "Company Name Clean", "Predecessor", "Products",
                     "Services", "Brands", "Divisions", "Subsidiaries"]

# Construir índice invertido
keyword_index = {}  # keyword → lista de tickers

for _, row in wikipedia_actualizado.iterrows():
    ticker = row["Ticker"]

    keywords = []
    for col in COLUMNAS_KEYWORDS:
        if isinstance(row[col], str):
            # Separar por comas y limpiar cada keyword
            items = [k.strip() for k in row[col].split(",") if k.strip()]
            keywords.extend(items)

    for kw in keywords:
        kw_lower = kw.lower()
        # if len(kw_lower) < 4:  # descartar keywords demasiado cortas
        #     continue
        if kw_lower not in keyword_index:
            keyword_index[kw_lower] = []
        if ticker not in keyword_index[kw_lower]:
            keyword_index[kw_lower].append(ticker)


# Keywords repetidas solo de Products y Services
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

# Mostrar solo las repetidas
repetidas = {kw: tickers for kw, tickers in keyword_prod_serv.items() if len(tickers) > 1}


COLUMNAS_KEYWORDS = ["Predecessor",  "Products", "Services",
                    "Brands", "Divisions", "Subsidiaries"]
# Ver repetidas por columna de origen
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
    # if repetidas_col:
    #     for kw, tickers in sorted(repetidas_col.items(), key=lambda x: len(x[1]), reverse=True):
    #         print(f"  '{kw}' → {tickers}")
    #         # Keywords repetidas en Products/Services (ya las tienes calculadas)
keywords_a_eliminar = set(repetidas.keys())  # las de Products/Services

# Añadir keywords repetidas en otras columnas con más de 4 tickers
OTRAS_COLUMNAS = ["Predecessor", "Brands", "Divisions", "Subsidiaries"]

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

    # Eliminar solo las que aparecen en más de 4 tickers
    for kw, tickers in keyword_col.items():
        if len(tickers) > 4:
            keywords_a_eliminar.add(kw)

# Construir índice final limpio
keyword_index_clean = {kw: tickers for kw, tickers in keyword_index.items()
                       if kw not in keywords_a_eliminar}


noticias_con_NER = noticias_keywords_mapeados.copy()


def get_unique_sorted_by_freq(df, column):
    all_items = df[column].dropna().str.split(", ")
    flat_list = [item.strip() for sublist in all_items for item in sublist]

    counts = Counter(flat_list)

    # solo devolver las palabras ordenadas por frecuencia
    return [word for word, _ in counts.most_common()]

unique_names = get_unique_sorted_by_freq(noticias_con_NER, "All Names")
keywords = [x.lower() for x in unique_names]


NER_mapeado = pd.DataFrame({
    "original": keywords,
    "mapped": [keyword_index_clean.get(x) for x in keywords]
})

keys = list(keyword_index_clean.keys())

def get_match_info(x):
    match = process.extractOne(x, keys, scorer=fuzz.token_sort_ratio)
    if match:
        return match  # (match_string, score, index)
    return (None, 0, None)

NER_mapeado["match"], NER_mapeado["score"], _ = zip(*NER_mapeado["original"].apply(get_match_info))

THRESHOLD = 90

NER_mapeado["mapped_fuzzy"] = NER_mapeado.apply(
    lambda row: keyword_index_clean.get(row["match"]) if row["score"] >= THRESHOLD else None,
    axis=1
)

NER_bajo_score = NER_mapeado[NER_mapeado["score"] < THRESHOLD].copy()

def has_common_word(row):
    if not isinstance(row["original"], str) or not isinstance(row["match"], str):
        return False

    original_words = set(row["original"].lower().split())
    match_words = set(row["match"].lower().split())

    return len(original_words & match_words) > 0

NER_bajo_score["common_word"] = NER_bajo_score.apply(has_common_word, axis=1)

NER_candidatos = NER_bajo_score[NER_bajo_score["common_word"] == True].copy()

# ── Groq client ────────────────────────────────────────────────────────────
groc_client = Groq(api_key="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")

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

def check_relation_groc(original, match):
    prompt = build_groc_prompt(original, match)

    response = groc_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )

    # Parsear JSON
    import json
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"result": "Maybe", "reason": "Could not parse response"}

# Supongamos que df_candidates tiene 'original' y 'match'

def parse_llm_output(row):
    llm_result = check_relation_groc(row["original"], row["match"])

    # Convertir "Yes" a True, "No" a False, "Maybe" a False
    result_bool = llm_result.get("result", "Maybe") == "Yes"

    reason = llm_result.get("reason", "")

    return pd.Series([result_bool, reason])


NER_candidatos[["LLM_check_bool", "LLM_reason"]] = NER_candidatos.apply(parse_llm_output, axis=1)

NER_candidatos["mapped_llm"] = NER_candidatos.apply(
    lambda row: keyword_index_clean.get(row["match"]) if row["LLM_check_bool"] else None,
    axis=1
)

# Actualizar solo las filas donde mapped_LLM no es None
mask = NER_candidatos['mapped_llm'].notna()

NER_mapeado.loc[NER_candidatos[mask].index, 'mapped_fuzzy'] = NER_candidatos.loc[mask, 'mapped_llm']

# Crear lookup con clave en minúsculas
lookup = NER_mapeado.dropna(subset=['mapped_fuzzy']).set_index('original')['mapped_fuzzy'].to_dict()

def map_names_to_tickers(all_names):
    if pd.isna(all_names) or not all_names:
        return [] # Mejor devolver lista vacía que None para facilitar el cruce

    names = [n.strip() for n in all_names.split(',')]
    tickers = []

    for name in names:
        ticker = lookup.get(name.lower())
        if ticker:
            if isinstance(ticker, list):
                tickers.extend(ticker)
            else:
                tickers.append(ticker)

    # Eliminamos duplicados
    tickers = list(dict.fromkeys(tickers))

    # CAMBIO AQUÍ: Devuelve la lista directamente, NO el ".join"
    return tickers

noticias_con_NER['Tickers'] = noticias_con_NER['All Names'].apply(map_names_to_tickers)


def get_mapping_con_evidencia(tickers_noticia, all_names_noticia, df_keywords, df_ref):
    # Normalizamos All Names a una lista limpia en minúsculas
    if isinstance(all_names_noticia, str):
        all_names_list = [n.strip().lower() for n in all_names_noticia.split(',')]
    else:
        all_names_list = [str(n).strip().lower() for n in all_names_noticia]

    # Copia temporal para búsqueda insensible a mayúsculas
    df_temp = df_keywords.copy()
    df_temp['original_lower'] = df_temp['original'].str.lower().str.strip()

    mapping_entries = []

    for ticker in tickers_noticia:
        # Nombre oficial del SP500
        nombre_oficial = df_ref.loc[df_ref['Ticker'] == ticker, 'Company Name'].values
        nombre_oficial = nombre_oficial[0] if len(nombre_oficial) > 0 else "Unknown Entity"

        # Buscamos qué palabra de 'All Names' disparó este ticker en mapped_fuzzy
        mask = (df_temp['original_lower'].isin(all_names_list)) & \
               (df_temp['mapped_fuzzy'].apply(lambda x: ticker in x if isinstance(x, list) else ticker == str(x)))

        palabras_match = df_temp[mask]['original'].unique().tolist()
        evidencia = ", ".join(palabras_match) if palabras_match else "Contextual match"

        # Formato de auditoría: "TICKER (Empresa) -> Sugerido por: 'palabra'"
        mapping_entries.append(f"- {ticker} ({nombre_oficial}). Razón: keyword '{evidencia}'")

    return "\n".join(mapping_entries)

def audit_tickers_with_groq(full_text, mapping_evidencia):
    # Prompt de auditoría financiera
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
        model="llama-3.3-70b-versatile",  # <--- CAMBIA ESTO
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0
        )

        res = completion.choices[0].message.content.strip()

        # Limpieza de Markdown por si acaso
        if "```" in res:
            res = res.split("```")[-2].replace("python", "").strip()

        return ast.literal_eval(res)
    except Exception as e:
        print(f"Error en auditoría: {e}")
        return []


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
            break  # para en el primer error grave para no consumir más tokens

    procesadas = resultado['Cambios'].notna().sum()
    print(f"\n{'Completado' if not errores else 'Parcial'}. Filas procesadas: {procesadas}/{total}")
    if ultima_fila_ok:
        print(f"Última fila guardada: idx={ultima_fila_ok}")

    return resultado

noticias_muchos_tickers = noticias_con_NER[noticias_con_NER['Tickers'].apply(lambda x: len(x) > 1)]


# 1. Validación inicial: Si está vacío, creamos una lista vacía para 'resultados'
if noticias_muchos_tickers.empty:
    print("El DataFrame 'noticias_muchos_tickers' está vacío. Saltando procesamiento.")
    noticias_auditadas = pd.DataFrame() # Creamos un DF vacío final
else:
    # Dividir en 6 partes
    partes = np.array_split(noticias_muchos_tickers, 6)

    for i, parte in enumerate(partes):
        # Manejo de partes vacías dentro del split (por si hay menos de 6 filas)
        if not parte.empty:
            print(f"Parte {i+1}: {len(parte)} filas | índices {parte.index[0]} a {parte.index[-1]}")
        else:
            print(f"Parte {i+1}: 0 filas (Vacía)")

    api_keys = ["gsk_Sros...", "gsk_IEll...", "gsk_X1iU...", "gsk_hz9D...", "gsk_Zguy...", "gsk_2WvT..."]

    resultados = []
    for i, (parte, key) in enumerate(zip(partes, api_keys)):
        # Solo procesamos si la parte tiene contenido
        if parte.empty:
            continue

        print(f"\n Procesando parte {i+1}/6 con key {i+1}...")
        # Asumiendo que Groq ya está importado
        client = Groq(api_key=key)
        resultado = procesar_noticias_muchos_tickers(parte, NER_mapeado, wikipedia_actualizado)
        resultados.append(resultado)

    # 2. Unir todo (solo si hay resultados)
    if resultados:
        noticias_auditadas = pd.concat(resultados)
    else:
        noticias_auditadas = pd.DataFrame()


# Solo intentamos actualizar si noticias_auditadas tiene filas
if not noticias_auditadas.empty:
    noticias_con_NER.loc[noticias_auditadas.index, 'Tickers'] = noticias_auditadas['Tickers Filtrados']
    print(f"Se actualizaron {len(noticias_auditadas)} filas.")
else:
    print("No hay noticias auditadas para actualizar.")



# 1. Asegurar que las fechas estén en formato datetime y solo fecha (sin hora)
noticias_con_NER['Date_only'] = pd.to_datetime(noticias_con_NER['Date']).dt.date
composicion_sp500_actualizado['Date'] = pd.to_datetime(composicion_sp500_actualizado['Date']).dt.date

# 2. Convertir los tickers del universo (df2) en un diccionario de sets para búsqueda rápida
# Asumimos que los tickers en df2 están separados por ", "
universo_dict = composicion_sp500_actualizado.set_index('Date')['Ticker'].str.split(', ').apply(set).to_dict()

# 3. Función para filtrar los tickers
def filtrar_tickers(row):
    fecha = row['Date_only']
    tickers_fila = row['Tickers']

    # Si la lista está vacía o la fecha no está en el universo, devolvemos lista vacía
    if not tickers_fila or fecha not in universo_dict:
        return []

    # El universo de ese día
    universo_dia = universo_dict[fecha]

    # Retornamos solo los que coinciden
    return [t for t in tickers_fila if t in universo_dia]

# 4. Crear la nueva columna
noticias_con_NER['Tickers'] = noticias_con_NER.apply(filtrar_tickers, axis=1)

# Opcional: eliminar la columna auxiliar de fecha
noticias_con_NER = noticias_con_NER.drop(columns=['Date_only'])

noticias_con_sector = noticias_keywords_mapeados.copy()

def get_unique_sorted_by_freq(df, column):
    all_items = df[column].dropna().str.split(", ")
    flat_list = [item.strip() for sublist in all_items for item in sublist]

    counts = Counter(flat_list)

    # solo devolver las palabras ordenadas por frecuencia
    return [word for word, _ in counts.most_common()]

unique_channels = get_unique_sorted_by_freq(noticias_keywords_mapeados, "Economic Channels")


# ── Groq client ────────────────────────────────────────────────────────────
groq_client = Groq(api_key="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")


def parse_morningstar_df(df: pd.DataFrame) -> dict:
    # 1. Obtener sectores únicos (llaves con valores vacíos como el original)
    sectors = {sector: "" for sector in df["Sector"].unique()}

    # 2. Convertir el DataFrame a la lista de diccionarios de industrias
    # Renombramos columnas para que coincidan con la estructura que esperas
    industries = df.rename(columns={
        "Sector": "sector",
        "Industry Group": "industry_group",
        "Industry": "industry"
    }).to_dict(orient="records")

    # Si el DF no tiene descripción, añadimos el campo vacío por compatibilidad
    for item in industries:
        if "description" not in item:
            item["description"] = ""

    return {"sectors": sectors, "industries": industries}


# ── 2. Construir catálogo de 3 niveles ─────────────────────────────────────
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

    # Nivel Industry
    for ind in industries:
        catalog_inds.append({
            "level":          "industry",
            "sector":         ind["sector"],
            "industry_group": ind["industry_group"],
            "industry":       ind["industry"],
            "text":           ind["industry"]
        })

    # Nivel Industry Group
    for (sector, group), members in groups.items():
        catalog_groups.append({
            "level":          "industry_group",
            "sector":         sector,
            "industry_group": group,
            "industry":       None,
            "text":           group
        })

    # Nivel Sector
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


def build_embeddings(catalog: dict, model: SentenceTransformer):
    embeddings = {}
    for level, items in catalog.items():
        # Extraemos solo el texto de cada item en el catálogo
        texts = [item["text"] for item in items]

        print(f"Generando embeddings para {level} ({len(texts)} entradas)...")
        # El modelo convierte el texto en vectores numéricos
        embeddings[level] = model.encode(texts, show_progress_bar=True)

    return embeddings

# ── 4. Limpiar input ───────────────────────────────────────────────────────
def clean_nlp_input(text: str) -> str:
    noise   = r'\b(sector|industry|market)\b'
    cleaned = re.sub(noise, '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.lower()


# ── 5. Clasificador LLM ────────────────────────────────────────────────────
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

        # ── NUEVO: corrige inconsistencias entre level y campos rellenos ──
        if sector and industry_group and industry:
            level = "industry"
        elif sector and industry_group and not industry:
            level = "industry_group"
        elif sector and not industry_group and not industry:
            level = "sector"
        else:
            level = "unclassified"

        # Construye label según nivel
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

# ── 6. Función de normalización completa ──────────────────────────────────
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


def build_result_LLM(nlp_output, cleaned, llm_result, confidence,
                    best_per_level, top_n, llm_used, llm_reason, source):

  sector         = llm_result.get("sector")
  industry_group = llm_result.get("industry_group")
  industry       = llm_result.get("industry")  # ← NUEVO
  level          = llm_result.get("level", "unclassified")
  label          = llm_result.get("label")

  return {
      "input":          nlp_output,
      "cleaned":        cleaned,
      "label":          label,
      "sector":         sector,
      "industry_group": industry_group,
      "industry":       industry,  # ← NUEVO
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

    # ── NUEVO: extrae top 2 sectores por score ──
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

    # Score demasiado bajo → LLM clasifica directamente
    if best_score < threshold_min:
        llm_result = classify_with_llm(cleaned, catalog, probable_sectors)
        return build_result_LLM(nlp_output, cleaned, llm_result,
                                best_score, best_per_level, winner["top_n"],
                                llm_used=True, llm_reason=llm_result["reason"],
                                source="llm_direct")

    # Score alto → embedding directo sin LLM
    if best_score >= threshold_high:
        return build_result_embedding(nlp_output, cleaned, best_item,
                             level_map[winner_level], best_score,
                             best_per_level, winner["top_n"],
                             llm_used=False, llm_reason="high confidence",
                             source="embedding")

    # Score medio → LLM clasifica
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



# ── 7. Ejecutar ────────────────────────────────────────────────────────────
parsed     = parse_morningstar_df(morningstar)  #Debo adpatar a que sea una tabla
catalog    = build_catalog(parsed)
model      = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = build_embeddings(catalog, model)

levels = Counter(item["level"] for level_items in catalog.values()
                 for item in level_items)

# ── 8. Procesar inputs con checkpoint ─────────────────────────────────────
resultados = []

for i, s in enumerate(unique_channels):
    try:
        r = normalize_sector(s, catalog, embeddings, model)
        resultados.append(r)

        if (i + 1) % 100 == 0:
            print(f"Procesados: {i+1} keywords")

    except Exception as e:
        if "429" in str(e):
            print(f"Rate limit en keyword {i+1}: {s}")
            print(f"Guardando {len(resultados)} resultados...")
            break
        else:
            raise e

# ── 9. Mostrar resultados ──────────────────────────────────────────────────
clasificacion_sector = pd.DataFrame([{
    "input":      r["input"],
    "label":      r["label"],
    "level":      r["level"],
    "output":     r["industry"] or r["industry_group"] or r["sector"],
    "confidence": r["confidence"],
    "source":     r["source"],
    "llm_reason": r["llm_reason"]
} for r in resultados])

mapeo = {}
for _, row in clasificacion_sector.iterrows():
    if row["level"] not in ["unclassified", "error"]:
        mapeo[row["input"]] = row["output"]

def mapear_canales(economic_channels_str, mapeo):
    # Si no hay datos de entrada, devolvemos una lista vacía
    if pd.isna(economic_channels_str):
        return []

    canales = [c.strip().lower() for c in economic_channels_str.split(",")]

    # Obtenemos los resultados únicos
    resultados = list(dict.fromkeys(
        mapeo[canal] for canal in canales if canal in mapeo
    ))

    # CAMBIO: Si no hay resultados, devolvemos [] en lugar de None
    return resultados if resultados else []

noticias_con_sector["Morningstar"] = noticias_con_sector["Economic Channels"].apply(
    lambda x: mapear_canales(x, mapeo)
)

# Paso 1: construye índice
empresa_por_clasificador = {}

for _, row in empresas_sectores_morningstar.iterrows():
    for clasificador in [row["Sector"], row["Industry Group"], row["Industry"]]:
        if pd.notna(clasificador):
            if clasificador not in empresa_por_clasificador:
                empresa_por_clasificador[clasificador] = []
            empresa_por_clasificador[clasificador].append(row["Ticker"])


# Paso 2: mapea a tickers (siempre devolviendo listas)
def obtener_tickers(morningstar_list):
    # Si es None, NaN o una lista vacía [], devolvemos lista vacía
    if not isinstance(morningstar_list, list) or not morningstar_list:
        return []

    tickers = []
    for clasificador in morningstar_list:
        # Buscamos en el diccionario y extendemos la lista
        encontrados = empresa_por_clasificador.get(clasificador, [])
        tickers.extend(encontrados)

    # Eliminamos duplicados manteniendo el orden
    resultado = list(dict.fromkeys(tickers))

    # Devolvemos la lista (aunque esté vacía) para mantener consistencia
    return resultado

noticias_con_sector["Tickers"] = noticias_con_sector["Morningstar"].apply(obtener_tickers)


# 1. Asegurar que las fechas estén en formato datetime y solo fecha (sin hora)
noticias_con_sector['Date_only'] = pd.to_datetime(noticias_con_sector['Date']).dt.date
composicion_sp500_actualizado['Date'] = pd.to_datetime(composicion_sp500_actualizado['Date']).dt.date

# 2. Convertir los tickers del universo (df2) en un diccionario de sets para búsqueda rápida
# Asumimos que los tickers en df2 están separados por ", "
universo_dict = composicion_sp500_actualizado.set_index('Date')['Ticker'].str.split(', ').apply(set).to_dict()

# 3. Función para filtrar los tickers
def filtrar_tickers(row):
    fecha = row['Date_only']
    tickers_fila = row['Tickers']

    # Si la lista está vacía o la fecha no está en el universo, devolvemos lista vacía
    if not tickers_fila or fecha not in universo_dict:
        return []

    # El universo de ese día
    universo_dia = universo_dict[fecha]

    # Retornamos solo los que coinciden
    return [t for t in tickers_fila if t in universo_dia]

# 4. Crear la nueva columna
noticias_con_sector['Tickers'] = noticias_con_sector.apply(filtrar_tickers, axis=1)

# Opcional: eliminar la columna auxiliar de fecha
noticias_con_sector = noticias_con_sector.drop(columns=['Date_only'])

# 2. Construimos el DataFrame base (sin Morningstar ni el Tickers original)
noticias_input = noticias_con_sector.drop(columns=['Morningstar', 'Tickers'], errors='ignore').copy()

# 3. Insertamos las columnas en el orden solicitado
noticias_input['Tickers Sector'] = noticias_con_sector['Tickers']
noticias_input['Tickers Empresa'] = noticias_con_NER['Tickers']

# 4. Creamos la columna combinada
# Al ser ya listas limpias, la suma y el set funcionarán sin errores
noticias_input['Tickers Combinados'] = [
    sorted(list(set(s + e))) for s, e in zip(noticias_input['Tickers Sector'], noticias_input['Tickers Empresa'])
]



# Condición 1: (All Names y Financial Mentions NO son Na) Y (Tickers Empresa NO está vacío)
condicion_1 = (
    noticias_input['All Names'].notna() &
    noticias_input['Financial Mentions'].notna() &
    (noticias_input['Tickers Empresa'].str.len() > 0)
)

# Condición 2: (Economic Channels NO es Na) Y (Tickers Sector NO está vacío)
condicion_2 = (
    noticias_input['Economic Channels'].notna() &
    (noticias_input['Tickers Sector'].str.len() > 0)
)

# Filtro final: Se queda si cumple Condición 1 O Condición 2
noticias_input_filtrado = noticias_input[condicion_1 | condicion_2].copy()
noticias_input_filtrado = noticias_input_filtrado.reset_index(drop=True)

# Cambiamos el nombre de la variable a 'llm'
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=0,
    context_length=2048
)

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

    # Deduplicar manteniendo orden
    return list(dict.fromkeys(keywords_encontradas))


# input → label Morningstar
mapeo = {}
for _, row in clasificacion_sector.iterrows():
    if row["level"] not in ["unclassified", "error"]:
        mapeo[row["input"]] = row["output"]

def get_keywords_sector(economic_channels, mapeo):
    # Manejar NaN o None
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


def extract_event_empresa(full_text, tickers_empresa, tickers_sector, economic_channels, all_names, df_keywords, mapeo):

    keywords_empresa = get_keywords_empresa(tickers_empresa, all_names, df_keywords) if tickers_empresa and len(tickers_empresa) > 0 else []
    keywords_sector = get_keywords_sector(economic_channels, mapeo) if economic_channels and not isinstance(economic_channels, float) else []

    if not keywords_empresa:
        return []

    participantes = "BLOCK A - Entity keywords explicitly found in the news:\n"
    participantes += "\n".join([f'- "{kw}"' for kw in keywords_empresa])

    # Sector como contexto silencioso — ayuda al modelo a razonar pero no pide incluirlo en output
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
        # Con ctransformers NO usas tokenizer() ni torch.no_grad()
        # Simplemente llamas a 'llm' pasándole el prompt directamente
        response = llm(  #llm
            prompt,
            max_new_tokens=400,
            temperature=0,
            stop=["[/INST]", "</s>"]
        )

        response = response.strip()

        # Extraer JSON de la respuesta de texto
        matches = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
        if not matches:
            return []

        return [json.loads(m) for m in matches]

    except Exception as e:
        print(f"Error en la generación o parseo: {e}")
        return []


def normalize_to_list(x):

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, list):
        return [str(i).strip().lower() for i in x if i is not None and not pd.isna(i)]

    return [str(x).strip().lower()]

def clean_text(x):
    if x is None:
        return None
    return str(x).strip().lower()

def eventos_a_df_fast(resultado, indice_noticia, keywords_validas):

    filas = []
    append = filas.append

    # caso 1: resultado vacío
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

    # caso normal (guardar TODO)
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


def procesar_noticias_completo(noticias):
    """
    Procesa todas las noticias y devuelve un DataFrame con los eventos.
    No guarda archivos en disco.
    """
    inicio_total = time.time()
    all_dfs = []
    tiempos = []

    print(f"Iniciando procesamiento de {len(noticias)} noticias...")

    for i in noticias.index:
        fila = noticias.loc[i]
        t0 = time.time()

        # 1. Extracción con el modelo LLM (ya adaptado para CPU)
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

        # 2. Obtención de keywords para validación
        keywords_filtradas = get_keywords_empresa(
            tickers_noticia=fila['Tickers Empresa'],
            all_names_noticia=fila['All Names'],
            df_keywords=NER_mapeado
        )

        # 3. Conversión a DataFrame
        df_eventos = eventos_a_df_fast(
            resultado=resultado,
            indice_noticia=i,
            keywords_validas=keywords_filtradas
        )

        all_dfs.append(df_eventos)

        # Tracking de tiempo
        duracion = time.time() - t0
        tiempos.append(duracion)

        if (len(tiempos) % 5 == 0): # Feedback cada 5 noticias
            print(f"Procesadas {len(tiempos)}/{len(noticias)} - Última: {duracion:.2f}s")

    # Unificar todos los resultados en un solo DataFrame
    if all_dfs:
        df_final = pd.concat(all_dfs, ignore_index=True)
    else:
        df_final = pd.DataFrame()

    fin_total = time.time()

    print("\n=== PROCESO FINALIZADO ===")
    print(f"Noticias totales: {len(noticias)}")
    print(f"Tiempo total: {(fin_total - inicio_total)/60:.2f} min")
    print(f"Tiempo medio por noticia: {np.mean(tiempos):.2f} s")

    return df_final


mask = noticias_input_filtrado['Tickers Empresa'].apply(lambda x: len(x) > 0)
noticias_con_empresas = noticias_input_filtrado[mask].copy()

if not noticias_con_empresas.empty:
    print(f"Iniciando proceso para {len(noticias_con_empresas)} noticias...")
    # Reiniciamos índice solo si vamos a procesar
    # noticias_con_empresas = noticias_con_empresas.reset_index(drop=True)
    resultado_empresa = procesar_noticias_completo(noticias_con_empresas)
else:
    print("No se encontraron noticias con sectores. Devolviendo DataFrame vacío.")
    # Creamos un DataFrame vacío con las columnas que esperas de eventos_a_df_fast
    resultado_empresa = pd.DataFrame(columns=[
        "Fila Noticia", "Evento", "Keywords Validas", "Agente", "Paciente",
        "Afectado", "Verbo", "Contexto", "Status"
    ])
    
    
def extract_event_sector(full_text, tickers_sector, economic_channels, mapeo):
    # 1. Obtener keywords del sector (asumo que esta función ya la tienes definida)
    keywords_sector = get_keywords_sector(economic_channels, mapeo) if economic_channels and not isinstance(economic_channels, float) else []

    if not keywords_sector:
        return []

    # 2. Preparar los participantes para el prompt
    participantes = "SECTOR KEYWORDS (economic sectors identified as relevant to this news):\n"
    participantes += "\n".join([f'- "{kw}"' for kw in keywords_sector])

    # 3. Construir el Prompt (usando el formato de Mistral)
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

    # 4. INFERENCIA CON CTRANSFORMERS (CPU)
    # model.generate no se usa aquí, se llama al objeto directamente
    response = llm(
        prompt,
        max_new_tokens=300,
        temperature=0, # Equivalente a do_sample=False para extraer datos
        stop=["</s>"]
    )

    # 5. EXTRACCIÓN DE JSON
    try:
        # Buscamos el contenido entre corchetes o llaves
        matches = re.findall(r'\{[^{}]*\}', response, re.DOTALL)
        if not matches:
            return []

        # Limpiamos posibles errores de formato y cargamos
        result = [json.loads(m.replace("'", '"')) for m in matches]
        return result
    except Exception as e:
        print(f"Error parseando JSON: {e}")
        return []


def normalize_to_list(x):

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, list):
        return [str(i).strip().lower() for i in x if i is not None and not pd.isna(i)]

    return [str(x).strip().lower()]

def clean_text(x):
    if x is None:
        return None
    return str(x).strip().lower()

def eventos_a_df_fast(resultado, indice_noticia, keywords_validas):

    kw_set = set(kw.lower() for kw in keywords_validas)

    filas = []
    append = filas.append

    # caso 1: resultado vacío → guardar fila placeholder
    if not resultado:
        append((
            indice_noticia,
            1,
            keywords_validas,  # ← nueva columna
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

    # caso normal (guardar TODO, aunque no haya keywords)
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
            keywords_validas,  # ← nueva columna
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



def procesar_noticias(noticias, mapeo):
    inicio_total = time.time()
    todos_los_dfs = []
    tiempos = []

    print(f"Iniciando procesamiento de {len(noticias)} noticias...")

    for i in noticias.index:
        fila = noticias.loc[i]
        t0 = time.time()

        # 1. Extraer eventos con el modelo (CPU)
        resultado = extract_event_sector(
            full_text         = fila['Full Text'],# Apareci Text Clean
            tickers_sector    = fila['Tickers Sector'],
            economic_channels = fila['Economic Channels'],
            mapeo             = mapeo
        )

        # 2. Obtener keywords validadas
        keywords_filtradas = get_keywords_sector(
            economic_channels = fila['Economic Channels'],
            mapeo = mapeo
        )

        # 3. Convertir a DataFrame
        df_eventos = eventos_a_df_fast(
            resultado=resultado,
            indice_noticia=fila.name,
            keywords_validas=keywords_filtradas
        )

        todos_los_dfs.append(df_eventos)
        tiempos.append(time.time() - t0)

        # Feedback visual cada 10 noticias para monitorear el progreso en CPU
        if (len(todos_los_dfs) % 10) == 0:
            print(f"Procesadas {len(todos_los_dfs)}/{len(noticias)} noticias...")

    # Consolidar todo en un solo DataFrame
    df_final = pd.concat(todos_los_dfs, ignore_index=True)

    fin_total = time.time()

    print(f"Tiempo total: {(fin_total - inicio_total)/60:.2f} min")
    print(f"Tiempo medio por noticia: {np.mean(tiempos):.2f} s")

    return df_final

# 1. Aplicamos el filtro para encontrar noticias con Tickers Sector (Arrays)
mask = noticias_input_filtrado['Tickers Sector'].apply(lambda x: len(x) > 0)
noticias_con_sectores = noticias_input_filtrado[mask].copy()

# 2. Condición: Si hay filas, procesamos. Si no, devolvemos vacío.
if not noticias_input_filtrado.empty:
    print(f"Iniciando proceso para {len(noticias_input_filtrado)} noticias...")
    # Reiniciamos índice solo si vamos a procesar
    # noticias_input_filtrado = noticias_input_filtrado.reset_index(drop=True)
    resultado_sector = procesar_noticias(noticias_input_filtrado, mapeo)
else:
    print("No se encontraron noticias con sectores. Devolviendo DataFrame vacío.")
    # Creamos un DataFrame vacío con las columnas que esperas de eventos_a_df_fast
    resultado_sector = pd.DataFrame(columns=[
        "Fila Noticia", "Evento", "Keywords Validas", "Agente", "Paciente",
        "Afectado", "Verbo", "Contexto", "Status"
    ])
    
ner  = resultado_empresa.copy()
ner = ner.dropna()


def es_repetida(row):
    # Aseguramos que keywords sea una lista
    keywords = list(row['Keywords Validas']) if isinstance(row['Keywords Validas'], (list, np.ndarray)) else []

    # Combinamos las 3 columnas asegurando que cada una sea tratada como lista
    agente = list(row['Agente']) if isinstance(row['Agente'], (list, np.ndarray)) else []
    paciente = list(row['Paciente']) if isinstance(row['Paciente'], (list, np.ndarray)) else []
    afectado = list(row['Afectado']) if isinstance(row['Afectado'], (list, np.ndarray)) else []

    pool_palabras = agente + paciente + afectado

    # Contamos las ocurrencias
    for kw in keywords:
        if pool_palabras.count(kw) > 1:
            return True
    return False

# Verificación de seguridad antes de procesar
if not ner.empty:
    repetidos_empresa = ner[ner.apply(es_repetida, axis=1)]
else:
    # Si ner está vacío, creamos un df vacío con las mismas columnas
    repetidos_empresa = pd.DataFrame(columns=ner.columns)

# Solo ejecutamos la limpieza si el DataFrame 'repetidos' tiene contenido
if not repetidos_empresa.empty:
    print(f"Limpiando {len(repetidos_empresa)} filas con duplicados...")

    # Función para limpiar duplicados entre columnas
    def limpiar_duplicados(row):
        # Usamos sets para una búsqueda más rápida
        agente = set(row['Agente']) if isinstance(row['Agente'], (list, np.ndarray)) else set()

        # Filtramos paciente y afectado quitando lo que esté en agente
        paciente_limpio = [x for x in row['Paciente'] if x not in agente]
        afectado_limpio = [x for x in row['Afectado'] if x not in agente]

        return pd.Series([paciente_limpio, afectado_limpio])

    # Aplicamos la limpieza solo a las columnas necesarias
    ner[['Paciente', 'Afectado']] = ner.apply(limpiar_duplicados, axis=1)
else:
    print("No se detectaron repetidos. Saltando limpieza.")

# añadir nueva columna
ner["Mencionado"] = True

sector  = resultado_sector.copy()
sector = sector.dropna()

# Verificación de seguridad antes de procesar
if not ner.empty:
    repetidos_sector = sector[sector.apply(es_repetida, axis=1)]
else:
    # Si ner está vacío, creamos un df vacío con las mismas columnas
    repetidos_sector = pd.DataFrame(columns=sector.columns)


def limpiar_afectado(row):
    agente = set(row["Agente"])
    paciente = set(row["Paciente"])

    existentes = agente.union(paciente)

    row["Afectado"] = [
        x for x in row["Afectado"]
        if x not in existentes
    ]

    return row

# Aplicamos al DataFrame
sector = sector.apply(limpiar_afectado, axis=1)

def limpiar_agente_paciente(row):
    agente_set = set(row["Agente"])
    paciente_set = set(row["Paciente"])

    # solo duplicados entre agente y paciente
    overlap = agente_set.intersection(paciente_set)

    if overlap:
        row["Paciente"] = [x for x in row["Paciente"] if x not in overlap]

    return row
sector = sector.apply(limpiar_agente_paciente, axis=1)

# añadir nueva columna
sector["Mencionado"] = False

inputs_totales = pd.concat([ner, sector], ignore_index=True)

inputs_totales = inputs_totales.sort_values(by=["Fila Noticia", "Evento"]).reset_index(drop=True)

rows = []

for _, r in inputs_totales.iterrows():
    fila = r["Fila Noticia"]

    for rol in ["Agente", "Paciente", "Afectado"]:
        keywords = r[rol]

        # Verificamos si es lista o un array de numpy
        if isinstance(keywords, (list, np.ndarray)):
            for kw in keywords:
                rows.append({
                    "Fila Noticia": fila,
                    "Evento": r["Evento"],
                    "Rol": rol.lower(),
                    "Keyword": str(kw).lower(), # Aseguramos que sea string antes de lower()
                    "Verbo": r["Verbo"],
                    "Contexto": r["Contexto"],
                    "Status": r["Status"],
                    "Mencionado": r["Mencionado"]
                })

inputs_totales_por_rol = pd.DataFrame(rows)

def get_tickers(row):
    keyword = row["Keyword"]
    fila = row["Fila Noticia"]

    # =========================
    # CASO 1: MENCIONADO TRUE
    # =========================
    if row["Mencionado"]:
        ner_dict = NER_mapeado.set_index("original")["mapped_fuzzy"].to_dict()
        tickers_ner = ner_dict.get(keyword, [])
        tickers_news = noticias_input_filtrado["Tickers Empresa"].to_dict().get(fila, [])

        if not isinstance(tickers_ner, list):
            tickers_ner = []

        if not isinstance(tickers_news, list):
            tickers_news = []

        return list(set(tickers_ner) & set(tickers_news))

    # =========================
    # CASO 2: MENCIONADO FALSE
    # =========================
    else:
        sector = mapeo.get(keyword)

        if not sector:
            return []

        dict_sector = empresas_sectores_morningstar.groupby('Sector')['Ticker'].apply(list).to_dict()
        dict_group = empresas_sectores_morningstar.groupby('Industry Group')['Ticker'].apply(list).to_dict()
        dict_industry = empresas_sectores_morningstar.groupby('Industry')['Ticker'].apply(list).to_dict()

        # 2. Unirlos todos en un único diccionario de mapeo
        empresa_por_morningstar = {**dict_sector, **dict_group, **dict_industry}

        tickers_sector = empresa_por_morningstar.get(sector, [])
        tickers_news = noticias_input_filtrado["Tickers Sector"].to_dict().get(fila, [])

        if isinstance(tickers_sector, (list, np.ndarray)):
            tickers_sector = list(tickers_sector)
        else:
            tickers_sector = []

        # Lo mismo para tickers_news
        if isinstance(tickers_news, (list, np.ndarray)):
            tickers_news = list(tickers_news)
        else:
            tickers_news = []

        # Ahora el set() no fallará
        return list(set(tickers_sector) & set(tickers_news))

inputs_totales_por_rol["Tickers Mapeados"] = inputs_totales_por_rol.apply(get_tickers, axis=1)

# Verifico que no haya listas vacias y me quedo con las que si tienen elementos
inputs_totales_por_rol = inputs_totales_por_rol[
    inputs_totales_por_rol["Tickers Mapeados"].map(lambda x: len(x) > 0 if isinstance(x, list) else False)
]

inputs_totales_por_rol = inputs_totales_por_rol.reset_index(drop=True)

def normalize_text(text):
    if pd.isna(text):
        return text

    # minúsculas
    text = text.lower()

    # reemplazar underscores
    text = text.replace("_", " ")

    # quitar acentos / caracteres raros
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    # quitar espacios extra
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_status(text):
    text = normalize_text(text)

    if text is None:
        return text

    # reglas
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

inputs_totales_por_rol["Status"] = inputs_totales_por_rol["Status"].apply(clean_status)

mask_error = (
    ~inputs_totales_por_rol["Verbo"].apply(lambda x: isinstance(x, str))  # no es string
    |
    inputs_totales_por_rol["Verbo"].astype(str).str.contains(",")        # tiene coma
)

df_verbos_problematicos = inputs_totales_por_rol[mask_error]

inputs_totales_por_rol["Verbo"] = inputs_totales_por_rol["Verbo"].apply(
    lambda x: x.split(",")[0].strip() if isinstance(x, str) else x
)

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

inputs_totales_por_rol["Verbo"] = inputs_totales_por_rol.apply(
    lambda x: x["Verbo_true"] if pd.notna(x["Verbo_true"]) else x["Verbo"],
    axis=1
)

inputs_totales_por_rol["Contexto"] = inputs_totales_por_rol.apply(
    lambda x: x["Contexto_true"] if pd.notna(x["Contexto_true"]) else x["Contexto"],
    axis=1
)

inputs_totales_por_rol["Status"] = inputs_totales_por_rol.apply(
    lambda x: x["Status_true"] if pd.notna(x["Status_true"]) else x["Status"],
    axis=1
)

inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Verbo_true", "Contexto_true", "Status_true"])


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

api_keys = ["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", "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"]  # puedes añadir más
clients = [Groq(api_key=k) for k in api_keys]

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

verb_parts = split_list(verbs, len(clients))

def chunk_list(lst, size=50):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

all_results = []

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

verbos_procesados = pd.DataFrame(all_results)

verbos_procesados = verbos_procesados.rename(columns={
    "text": "Verbo",
    "verb": "Verbo_Limpio"
})

inputs_totales_por_rol = inputs_totales_por_rol.merge(
    verbos_procesados,
    on="Verbo",
    how="left"
)
inputs_totales_por_rol

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

missing_results = []
client = Groq(api_key="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")
# batches pequeños = más fiable
def chunk_list(lst, size=50):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

for chunk in chunk_list(verbs_missing, 20):
    try:
        results = extract_main_verbs_batch(chunk, client)

        # IMPORTANTE: mapear por texto
        mapping = {r["text"]: r["verb"] for r in results}

        for v in chunk:
            missing_results.append({
                "Verbo": v,
                "Verbo_Limpio": mapping.get(v, None)
            })

    except Exception as e:
        print(f"Error en batch: {e}")


verbos_no_procesados = pd.DataFrame(missing_results)

# Verificación de seguridad: ¿Existe el DataFrame y tiene filas?
if not verbos_no_procesados.empty:
    # 1. Identificar filas donde Verbo_Limpio es NaN
    mask = verbos_no_procesados['Verbo_Limpio'].isna()
    verbo_sucio = verbos_no_procesados.loc[mask, 'Verbo'].tolist()

    # Solo ejecutamos si realmente hay verbos pendientes (verbo_sucio no está vacía)
    if verbo_sucio:
        # 2. Obtener resultados del modelo
        resultados = extract_main_verbs_batch(verbo_sucio, client)

        # 3. Extraer solo los verbos limpios
        verbos_ordenados = [item['verb'] for item in resultados]

        # 4. Asignar directamente a las filas filtradas
        verbos_no_procesados.loc[mask, 'Verbo_Limpio'] = verbos_ordenados

        print(f"Actualización completada: {len(verbos_ordenados)} verbos procesados.")
    else:
        print("No hay filas con Verbo_Limpio pendiente (NaN).")
else:
    print("El DataFrame 'verbos_no_procesados' está vacío. Saltando paso.")


# Aseguramos que si está vacío, al menos tenga las columnas necesarias para el join
columnas_requeridas = ['Verbo', 'Verbo_Limpio']

if verbos_no_procesados.empty:
    # Si está vacío, nos aseguramos de que tenga las columnas para que set_index('Verbo') no falle
    for col in columnas_requeridas:
        if col not in verbos_no_procesados.columns:
            verbos_no_procesados[col] = None
else:
    # ... aquí va el bloque de código de procesamiento con el 'if verbo_sucio' que vimos antes ...
    mask = verbos_no_procesados['Verbo_Limpio'].isna()
    verbo_sucio = verbos_no_procesados.loc[mask, 'Verbo'].tolist()

    if verbo_sucio:
        resultados = extract_main_verbs_batch(verbo_sucio, client)
        verbos_ordenados = [item['verb'] for item in resultados]
        verbos_no_procesados.loc[mask, 'Verbo_Limpio'] = verbos_ordenados

# Ahora este bloque SIEMPRE funcionará sin errores de "KeyError: 'Verbo'"
df1 = verbos_procesados.set_index('Verbo')
df2 = verbos_no_procesados.set_index('Verbo')

verbos_procesados_finales = df1.combine_first(df2).reset_index()

inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Verbo_Limpio"])

inputs_totales_por_rol = inputs_totales_por_rol.merge(
    verbos_procesados_finales,
    on="Verbo",
    how="left"
)

inputs_totales_por_rol["Verbo"] = inputs_totales_por_rol["Verbo_Limpio"]
inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Verbo_Limpio"])

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

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

context_parts = split_list(contexts, len(clients))

def chunk_list(lst, size=50):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

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

contextos_procesados = pd.DataFrame(all_results)

contextos_procesados = contextos_procesados.rename(columns={
    "text": "Contexto",
    "context": "Contexto_Limpio"
})

inputs_totales_por_rol = inputs_totales_por_rol.merge(
    contextos_procesados,
    on="Contexto",
    how="left"
)

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

missing_results = []
client = Groq(api_key="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")
# batches pequeños = más fiable
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

contextos_no_procesados = pd.DataFrame(missing_results)

# Aseguramos que existan las columnas mínimas para que el flujo no se rompa
for col in ['Contexto', 'Contexto_Limpio']:
    if col not in contextos_no_procesados.columns:
        contextos_no_procesados[col] = None

if not contextos_no_procesados.empty:
    # 1. Identificar filas con Contexto_Limpio pendiente
    mask = contextos_no_procesados['Contexto_Limpio'].isna()
    contexto_sucio = contextos_no_procesados.loc[mask, 'Contexto'].tolist()

    if contexto_sucio:
        # 2. Obtener resultados (Llamada al batch de contexto)
        resultados = extract_main_context_batch(contexto_sucio, client)

        # 3. Extraer solo los contextos limpios manteniendo el orden
        contextos_ordenados = [item['context'] for item in resultados]

        # 4. Asignar directamente
        contextos_no_procesados.loc[mask, 'Contexto_Limpio'] = contextos_ordenados
        print(f"Actualización completada: {len(contextos_ordenados)} contextos procesados.")
    else:
        print("No hay contextos pendientes de limpiar.")
else:
    print("El DataFrame de contextos está vacío.")

# --- PARTE 2: Unión de DataFrames (Merge/Combine) ---

# Aseguramos que el DF de procesados también tenga la columna índice
if 'Contexto' not in contextos_procesados.columns:
    contextos_procesados['Contexto'] = None

# 1. Seteamos índices para la combinación
df1 = contextos_procesados.set_index('Contexto')
df2 = contextos_no_procesados.set_index('Contexto')

# 2. Combinamos: rellena los huecos de df1 con la info nueva de df2
contextos_procesados_finales = df1.combine_first(df2).reset_index()

inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Contexto_Limpio"])

inputs_totales_por_rol = inputs_totales_por_rol.merge(
    contextos_procesados_finales,
    on="Contexto",
    how="left"
)

inputs_totales_por_rol["Contexto"] = inputs_totales_por_rol["Contexto_Limpio"]
inputs_totales_por_rol = inputs_totales_por_rol.drop(columns=["Contexto_Limpio"])

df_tickers = inputs_totales_por_rol.explode("Tickers Mapeados").copy()

# quitar tickers nulos
df_tickers = df_tickers[df_tickers["Tickers Mapeados"].notna()]

max_eventos = df_tickers.groupby("Fila Noticia")["Evento"].max()

def resolver_grupo(g):
    #  regla clave: si hay algún True → quedarse solo con esos
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

# Esto tarda un poco
inputs_gramatical = (
    df_tickers
    .groupby(["Fila Noticia", "Evento", "Tickers Mapeados"])
    .apply(resolver_grupo)
    .reset_index()
)

# me cargo la columna Evento, porque ya tengo Eventos
inputs_gramatical = inputs_gramatical.drop(columns=['Evento'])

# 1. Donde Mencionado sea False, ponemos False en Agente, Paciente y Afectado
# Usamos .loc[filas, columnas] para modificar solo donde se cumple la condición
inputs_gramatical.loc[inputs_gramatical["Mencionado"] == False, ["Agente", "Paciente", "Afectado"]] = False

# 2. Eliminamos la columna Mencionado
inputs_gramatical = inputs_gramatical.drop(columns=["Mencionado"])


def extraer_noticia(text):
    if pd.isna(text) or not isinstance(text, str):
        return None, None

    # Extracción de Título
    titulo_match = re.search(r"\[TITLE\]\s*(.*?)(?=\s*\[|$)", text, re.DOTALL)
    titulo = titulo_match.group(1).strip() if titulo_match else None

    # Extracción de Contenido
    contenido_match = re.search(r"\[CONTENT\]\s*(.*)", text, re.DOTALL)

    if contenido_match:
        contenido = contenido_match.group(1).strip()
    else:
        # Si no hay contenido, repetimos el título
        contenido = titulo

    return titulo, contenido

# Aplicamos y desempaquetamos
resultados = noticias_input_filtrado['Full Text'].apply(extraer_noticia)
titulos, contenidos = zip(*resultados)

# Insertamos en las posiciones 3 y 4 (índices 2 y 3)
noticias_input_filtrado.insert(2, "Titulo Noticia", titulos)
noticias_input_filtrado.insert(3, "Contenido Noticia", contenidos)

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
        return "extra oficial"  # por seguridad

# Asegurar datetime
noticias_input_filtrado["Date"] = pd.to_datetime(noticias_input_filtrado["Date"])

day_col = noticias_input_filtrado["Date"].dt.day_name()
time_bucket_col = noticias_input_filtrado["Date"].apply(get_time_bucket)

# Eliminar si existen
for col in ["Week Day", "Date Time"]:
    if col in noticias_input_filtrado.columns:
        noticias_input_filtrado.drop(columns=[col], inplace=True)

# Insertar
noticias_input_filtrado.insert(1, "Week Day", day_col)
noticias_input_filtrado.insert(2, "Date Time", time_bucket_col)

date = noticias_input_filtrado["Date"]
day_map = noticias_input_filtrado["Week Day"]
time_map = noticias_input_filtrado["Date Time"]
title_map = noticias_input_filtrado["Titulo Noticia"]
content_map = noticias_input_filtrado["Contenido Noticia"]

inputs_gramatical.insert(
    1,
    "Date",
    inputs_gramatical["Fila Noticia"].map(date)
)


inputs_gramatical.insert(
    2,
    "Week Day",
    inputs_gramatical["Fila Noticia"].map(day_map)
)

inputs_gramatical.insert(
    3,
    "Date Time",
    inputs_gramatical["Fila Noticia"].map(time_map)
)

inputs_gramatical.insert(
    4,
    "Titulo Noticia",
    inputs_gramatical["Fila Noticia"].map(title_map)
)

inputs_gramatical.insert(
    5,
    "Contenido Noticia",
    inputs_gramatical["Fila Noticia"].map(content_map)
)



inputs_gramatical = inputs_gramatical.groupby(["Fila Noticia", "Tickers Mapeados"]).agg({

    "Date": "first",
    "Week Day": "first",
    "Date Time": "first",
    "Titulo Noticia": "first",
    "Contenido Noticia": "first",
    "Eventos": "max",

    # booleanos → OR lógico
    "Agente": "max",
    "Paciente": "max",
    "Afectado": "max",

    # Mencionado → SUMA (como definiste)
    #"Mencionado": "sum",

    # IMPORTANTE: mantener todos los valores (NO colapsar)
    "Verbo": list,
    "Contexto": list,
    "Status": list,

}).reset_index()

# ordeno por fecha
inputs_gramatical['Date'] = pd.to_datetime(inputs_gramatical['Date'])
inputs_gramatical = inputs_gramatical.sort_values(['Date', 'Fila Noticia']).reset_index(drop=True)


boolean_cols = ["Agente", "Paciente", "Afectado"]
boolean_data = inputs_gramatical[boolean_cols].astype(float)


categorical_cols = ["Week Day", "Date Time"]

# Configuramos el encoder para que devuelva un DataFrame de Pandas
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.set_output(transform="pandas")

# Ahora 'encoded_data' será un DataFrame con los nombres automáticos
encoded_data = encoder.fit_transform(inputs_gramatical[categorical_cols])


status_map = {
    "speculative": 0,
    "in progress": 1,
    "confirmed neutral": 2,
    "confirmed positive": 3,
    "confirmed negative": 4
}


def multi_hot_status(values, n_classes=5):
    vec = np.zeros(n_classes)
    for s in values:
        vec[status_map[s]] = 1
    return vec

status_array = np.stack(inputs_gramatical["Status"].apply(multi_hot_status))

# Crear el DataFrame resultante
status_cols = [f"Status_{k}" for k in status_map.keys()]
status_data = pd.DataFrame(status_array, columns=status_cols, index=inputs_gramatical.index)

numeric_cols = ["Eventos"]

numeric_data = inputs_gramatical[numeric_cols].astype(float)



# 1. Detectar el dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# 2. Mover el modelo a la GPU
model = AutoModel.from_pretrained("ProsusAI/finbert").to(device)
model.eval() # Poner en modo evaluación


def get_embeddings(text_list, batch_size=64): # En GPU puedes subir el batch a 32 o 64
    embeddings_list = []

    # Usamos tqdm para monitorear el tiempo real
    # for i in tqdm(range(0, len(text_list), batch_size)):
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device) # 3. Mover los inputs a la GPU

        # 4. Usar inference_mode es más rápido que no_grad
        with torch.inference_mode():
            outputs = model(**inputs)

        # 5. Extraer el CLS y mover de vuelta a CPU para liberar memoria GPU
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings_list.append(cls_embeddings)

        # Opcional: Liberar caché si notas lentitud
        # torch.cuda.empty_cache()

    return np.vstack(embeddings_list)

title_embeddings_matrix = get_embeddings(inputs_gramatical["Titulo Noticia"].tolist())

title_embedding_data = pd.DataFrame(title_embeddings_matrix)
title_embedding_data.columns = [f'Titulo_{i}' for i in range(title_embedding_data.shape[1])]

content_embeddings_matrix = get_embeddings(inputs_gramatical["Contenido Noticia"].tolist())

content_embedding_data = pd.DataFrame(content_embeddings_matrix)
content_embedding_data.columns = [f'Contenido_{i}' for i in range(content_embedding_data.shape[1])]


def get_embeddings_ndarray(verbo_column, batch_size=64):
    model.to(device)
    model.eval()

    # 1. Preparar datos: aplanamos todos los verbos y guardamos cuántos hay por fila
    all_verbs = []
    counts = []

    for val in verbo_column:
        if isinstance(val, (np.ndarray, list)):
            lista = [str(word).strip() for word in val if str(word).strip()]
        elif isinstance(val, str):
            lista = [w.strip() for w in val.replace('[','').replace(']','').split(',') if w.strip()]
        else:
            lista = []

        all_verbs.extend(lista if lista else [""]) # Evitamos listas vacías
        counts.append(len(lista) if lista else 1)

    # 2. Procesar todos los verbos en batches (usando tu lógica de GPU)
    embeddings_list = []
    # for i in tqdm(range(0, len(all_verbs), batch_size), desc=f"Procesando verbos en {device}"):
    for i in range(0, len(all_verbs), batch_size):

        batch = all_verbs[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            # Extraemos el token CLS
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.append(cls_emb)

    all_embeddings = np.vstack(embeddings_list)

    # 3. Re-agrupar y calcular la media por cada fila original
    final_mean_embeddings = []
    current_idx = 0
    for count in counts:
        # Tomamos los embeddings que pertenecen a esta fila y sacamos la media
        row_embs = all_embeddings[current_idx : current_idx + count]
        final_mean_embeddings.append(np.mean(row_embs, axis=0))
        current_idx += count

    return np.vstack(final_mean_embeddings)

verbo_embeddings_matrix = get_embeddings_ndarray(inputs_gramatical['Verbo'])

# Crear DataFrame expandido
verbo_embeddings = pd.DataFrame(verbo_embeddings_matrix)
verbo_embeddings.columns = [f'Verbo_{i}' for i in range(verbo_embeddings.shape[1])]

contexto_embeddings_matrix = get_embeddings_ndarray(inputs_gramatical['Contexto'])

# Crear DataFrame expandido
contexto_embeddings = pd.DataFrame(contexto_embeddings_matrix)
contexto_embeddings.columns = [f'Contexto_{i}' for i in range(contexto_embeddings.shape[1])]


# Lista de todos tus DataFrames
dataframes = [
    title_embedding_data,# Los embeddings de titulo
    content_embedding_data,# Embeddings de contenido
    verbo_embeddings,    # Embeddings de verbo
    contexto_embeddings, # Embeddings de contexto
    boolean_data,        # Agente, Paciente, Afectado
    numeric_data,        # Eventos y Mencionado ** quite mencionado
    encoded_data,        # Week Day, Sector, Datetime (One-Hot)
    status_data,         # Multi-hot Status
]

# Unirlos todos lateralmente
df_inputs = pd.concat(dataframes, axis=1)

# CONTROL DE SEGURIDAD ABSOLUTO: Si no hay datos, saltamos la red neuronal
if df_inputs.empty:
    print("⚠️ df_inputs está completamente vacío. Creando DataFrame de señales vacío...")
    seynales_modelo = pd.DataFrame(columns=["Fila Noticia", "Tickers Mapeados", "Date", "Prob_up", "Pred_label"])

else:
    # Divido los inputs
    X_title = df_inputs.iloc[:, 0:768].values
    X_content = df_inputs.iloc[:, 768:1536].values
    X_verb = df_inputs.iloc[:, 1536:2304].values
    X_context = df_inputs.iloc[:, 2304:3072].values
    # X_meta = df_inputs.iloc[:, 3072:3076].values #3076
    # Selección explícita por nombres para X_meta (Garantiza siempre 9 columnas)
    columnas_meta = [
        'Agente', 
        'Paciente', 
        'Afectado', 
        'Eventos', 
        'Status_speculative', 
        'Status_in progress',       # Ojo: con espacio según tu log
        'Status_confirmed neutral',  # Ojo: con espacio según tu log
        'Status_confirmed positive', # Ojo: con espacio según tu log
        'Status_confirmed negative'  # Ojo: con espacio según tu log
    ]

    X_meta = df_inputs[columnas_meta].values

    # =========================================================
    # ESCALADO INDEPENDIENTE
    # =========================================================

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

    X_title = torch.tensor(X_title, dtype=torch.float32)
    X_content = torch.tensor(X_content, dtype=torch.float32)
    X_verb = torch.tensor(X_verb, dtype=torch.float32)
    X_context = torch.tensor(X_context, dtype=torch.float32)
    X_meta = torch.tensor(X_meta, dtype=torch.float32)

    batch_size = 512

    test_loader = DataLoader(
        TensorDataset(X_title, X_content, X_verb, X_context, X_meta),
        batch_size=batch_size,
        shuffle=False  # importante: mantener orden temporal
    )

    class MultiInputModel(nn.Module):

        def __init__(self):

            super().__init__()

            # =================================================
            # HIPERPARAMETROS NN_41
            # =================================================

            hidden_dim = 48
            dropout = 0.15

            small_dim = hidden_dim // 2  # 24

            # =================================================
            # TITLE
            # =================================================

            self.title_branch = nn.Sequential(

                nn.Linear(768, hidden_dim),

                nn.ReLU(),

                nn.BatchNorm1d(hidden_dim),

                nn.Dropout(dropout),

                nn.Linear(hidden_dim, hidden_dim),

                nn.ReLU()
            )

            # =================================================
            # CONTENT
            # =================================================

            self.content_branch = nn.Sequential(

                nn.Linear(768, hidden_dim),

                nn.ReLU(),

                nn.BatchNorm1d(hidden_dim),

                nn.Dropout(dropout),

                nn.Linear(hidden_dim, hidden_dim),

                nn.ReLU()
            )

            # =================================================
            # VERB
            # =================================================

            self.verb_branch = nn.Sequential(

                nn.Linear(768, small_dim),

                nn.ReLU(),

                nn.Linear(small_dim, small_dim),

                nn.ReLU()
            )

            # =================================================
            # CONTEXT
            # =================================================

            self.context_branch = nn.Sequential(

                nn.Linear(768, small_dim),

                nn.ReLU(),

                nn.Linear(small_dim, small_dim),

                nn.ReLU()
            )

            # =================================================
            # META
            # =================================================

            self.meta_branch = nn.Sequential(

                nn.Linear(9, 32),

                nn.ReLU(),

                nn.Linear(32, 32),

                nn.ReLU()
            )

            # =================================================
            # COMBINED DIM
            # =================================================

            combined_dim = (
                hidden_dim
                + hidden_dim
                + small_dim
                + small_dim
            )

            # =================================================
            # GATE
            # =================================================

            self.gate = nn.Sequential(

                nn.Linear(combined_dim, combined_dim),

                nn.Sigmoid()
            )

            # =================================================
            # HEAD
            # =================================================

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

    # 2. CONFIGURA EL DISPOSITIVO
    device = torch.device("cpu")

    # 3. INSTANCIA EL MODELO
    model = MultiInputModel()

    # 4. CARGA LOS PESOS DESDE TU RUTA
    # model.load_state_dict(torch.load("mi_modelo_entrenado.pth", map_location=device))
    model.load_state_dict(torch.load(buffer, map_location=device))

    # 5. MODO EVALUACIÓN
    model.to(device)
    model.eval()

    model.eval()
    all_probs_test = []

    with torch.inference_mode():
        # El loader ahora solo devuelve los inputs, sin y_batch
        for x_title, x_content, x_verb, x_context, x_meta in test_loader:

            # Mover a CPU
            x_title = x_title.to(device)
            x_content = x_content.to(device)
            x_verb = x_verb.to(device)
            x_context = x_context.to(device)
            x_meta = x_meta.to(device)

            # Predicción
            outputs_test = model(x_title, x_content, x_verb, x_context, x_meta)
            probs_test = torch.sigmoid(outputs_test)

            all_probs_test.extend(probs_test.cpu().numpy())

    # Convertir a un array plano para fácil manejo
    predicciones_finales = np.array(all_probs_test).flatten()


    # convertir a numpy si no lo son
    probs = np.array(all_probs_test).flatten()

    # etiqueta predicha con threshold 0.5
    y_pred = (probs >= 0.5).astype(int)

    # construir dataframe
    df_results = pd.DataFrame({
        "Prob_up": probs,
        "Pred_label": y_pred
    })

    # Unimos seleccionando solo las 3 columnas de inputs_gramatical
    seynales_modelo = inputs_gramatical[["Fila Noticia", "Tickers Mapeados", "Date"]].merge(
        df_results,
        left_index=True,  # Usamos el índice si los resultados mantienen el orden original
        right_index=True
    )

# seynales_modelo = seynales_modelo.reset_index().rename(columns={'index': 'ID'})


def obtener_metricas(
    fila,
    df_vol,
    df_px,
    ventana=120,
    trailing=False,
    trailing_trigger_pct=0.02
):

    ticker      = fila['Tickers Mapeados']
    fecha_señal = fila['Date']
    label       = fila['Pred_label']

    # ==========================================================
    # VALIDACIONES INICIALES
    # ==========================================================

    if ticker not in df_vol.columns:
        return pd.Series([np.nan] * 20)

    idx_pos = df_vol.index.searchsorted(fecha_señal)

    if idx_pos >= len(df_vol):
        return pd.Series([np.nan] * 20)

    hora_dt         = fecha_señal.time()
    limite_apertura = pd.Timestamp("09:29:59").time()

    is_morning_today = (
        df_vol.index[idx_pos].date() == fecha_señal.date()
        and hora_dt <= pd.Timestamp("09:20:00").time()
    )

    is_overnight = (
        df_vol.index[idx_pos].date() > fecha_señal.date()
    )

    # ==========================================================
    # BUSQUEDA ENTRADA
    # ==========================================================

    if is_morning_today or is_overnight:

        proxima_ap_idx = idx_pos

        while (
            proxima_ap_idx < len(df_vol)
            and df_vol.index[proxima_ap_idx].time()
            != pd.Timestamp("09:30:00").time()
        ):
            proxima_ap_idx += 1

        if proxima_ap_idx < len(df_vol):

            idx_entrada    = df_vol.index[proxima_ap_idx]
            tipo_ejec_ent  = "APERTURA"
            minuto_entrada = 1

        else:

            return pd.Series([
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, 'PENDIENTE_ENTRAR',
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                pd.NaT, pd.NaT
            ])

    else:

        busqueda_misma = df_vol[ticker].iloc[idx_pos + 4: idx_pos + 400]

        busqueda_misma = busqueda_misma[
            busqueda_misma.index.date == fecha_señal.date()
        ].dropna()

        busqueda_misma = busqueda_misma[
            busqueda_misma.index.time < pd.Timestamp("16:00:00").time()
        ]

        if not busqueda_misma.empty:

            idx_entrada    = busqueda_misma.index[0]
            tipo_ejec_ent  = "SESION"

            minuto_entrada = (
                df_vol.index.get_loc(idx_entrada) - idx_pos
            ) + 1

        else:

            proxima_ap_idx = idx_pos

            while (
                proxima_ap_idx < len(df_vol)
                and df_vol.index[proxima_ap_idx].time()
                != pd.Timestamp("09:30:00").time()
            ):
                proxima_ap_idx += 1

            if proxima_ap_idx < len(df_vol):

                idx_entrada    = df_vol.index[proxima_ap_idx]
                tipo_ejec_ent  = "APERTURA"
                minuto_entrada = 1

            else:

                return pd.Series([
                    np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, 'PENDIENTE_ENTRAR',
                    np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan,
                    pd.NaT, pd.NaT
                ])

    # ==========================================================
    # VALIDAR IDX_ENTRADA EN PRECIOS
    # ==========================================================

    if idx_entrada not in df_px.index:

        return pd.Series([
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, 'PENDIENTE_ENTRAR',
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            pd.NaT, pd.NaT
        ])

    # ==========================================================
    # LIMITE PRODUCCION
    # ==========================================================

    ultimo_ts_disponible = df_px.index[-1]

    limite_sesion = pd.Timestamp(
        f"{ultimo_ts_disponible.date()} 15:59:00"
    )

    # ==========================================================
    # FUNCION AUXILIAR HISTORICO
    # ==========================================================

    def calcular_historico():

        dias_hist      = 10
        min_cobertura  = 0.70

        fechas_disponibles = pd.Index(
            sorted(
                df_vol.index[
                    df_vol.index < idx_entrada
                ].normalize().unique()
            )
        )

        ultimos_dias = fechas_disponibles[-dias_hist:]

        mask_hist = (
            df_vol.index.normalize().isin(ultimos_dias)
        ) & (
            df_vol.index < idx_entrada
        )

        # ======================================================
        # HORARIOS REFERENCIA
        # ======================================================

        if hora_dt <= limite_apertura:

            horarios_ref = [
                f"09:{m:02d}" for m in range(31, 37)
            ]

        else:

            horarios_ref = (
                df_vol.index[idx_pos: idx_pos + 5]
                .strftime('%H:%M')
                .tolist()
            )

        # ======================================================
        # HISTORICOS
        # ======================================================

        v_hist = df_vol.loc[mask_hist, ticker]

        v_hist = v_hist[
            v_hist.index.strftime('%H:%M').isin(horarios_ref)
        ].dropna()

        p_hist = df_px.loc[mask_hist, ticker]

        p_hist = p_hist[
            p_hist.index.strftime('%H:%M').isin(horarios_ref)
        ].dropna()

        # ======================================================
        # COBERTURA REAL
        # ======================================================

        obs_esperadas_vol = (
            df_vol.loc[mask_hist, ticker]
            .index.strftime('%H:%M')
            .isin(horarios_ref)
            .sum()
        )

        obs_esperadas_px = (
            df_px.loc[mask_hist, ticker]
            .index.strftime('%H:%M')
            .isin(horarios_ref)
            .sum()
        )

        obs_validas_vol = len(v_hist)
        obs_validas_px  = len(p_hist)

        cobertura_vol = (
            obs_validas_vol / obs_esperadas_vol
            if obs_esperadas_vol > 0 else 0
        )

        cobertura_px = (
            obs_validas_px / obs_esperadas_px
            if obs_esperadas_px > 0 else 0
        )

        # ======================================================
        # METRICAS
        # ======================================================

        if cobertura_vol < min_cobertura:

            vol_media = np.nan

        else:

            vol_media = round(v_hist.mean(), 2)

        if pd.notna(vol_media):

            vol_1_p = int(np.floor(vol_media * 0.01))

        else:

            vol_1_p = np.nan

        if cobertura_px < min_cobertura:

            px_media = np.nan
            px_std   = np.nan

        else:

            px_media = round(p_hist.mean(), 2)
            px_std   = round(p_hist.std(), 2)

        # ======================================================
        # UMBRALES
        # ======================================================

        if pd.isna(px_media) or pd.isna(px_std):

            u_gen_sup = np.nan
            u_gen_inf = np.nan
            u_ent_sup = np.nan
            u_ent_inf = np.nan

        else:

            u_gen_sup = round(px_media + 1.5 * px_std, 2)
            u_gen_inf = round(px_media - 1.5 * px_std, 2)

            u_ent_sup = round(px_media + 1.4 * px_std, 2)
            u_ent_inf = round(px_media - 1.4 * px_std, 2)

        return (
            vol_media,
            vol_1_p,
            px_media,
            px_std,
            u_gen_inf,
            u_gen_sup,
            u_ent_inf,
            u_ent_sup
        )

    # ==========================================================
    # PENDIENTE_ENTRAR (SIN DATOS FUTUROS)
    # ==========================================================

    if (
        idx_entrada > ultimo_ts_disponible
        or idx_entrada > limite_sesion
    ):

        (
            vol_media,
            vol_1_p,
            px_media,
            px_std,
            u_gen_inf,
            u_gen_sup,
            u_ent_inf,
            u_ent_sup
        ) = calcular_historico()

        return pd.Series([
            vol_media, vol_1_p, px_media, px_std,
            u_gen_inf, u_gen_sup, u_ent_inf, u_ent_sup,
            np.nan, np.nan, 'PENDIENTE_ENTRAR',
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            pd.NaT, pd.NaT
        ])

    # ==========================================================
    # DATOS ENTRADA
    # ==========================================================

    pos_entrada = df_vol.index.get_loc(idx_entrada)

    px_entrada = df_px.at[idx_entrada, ticker]

    if pd.isna(px_entrada):

        return pd.Series([
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, 'PENDIENTE_ENTRAR',
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            pd.NaT, pd.NaT
        ])

    px_entrada = round(px_entrada, 2)

    # ==========================================================
    # HISTORICO
    # ==========================================================

    (
        vol_media,
        vol_1_p,
        px_media,
        px_std,
        u_gen_inf,
        u_gen_sup,
        u_ent_inf,
        u_ent_sup
    ) = calcular_historico()

    # ==========================================================
    # SIN HISTORICO SUFICIENTE
    # ==========================================================

    if pd.isna(px_media) or pd.isna(px_std):

        return pd.Series([
            vol_media, vol_1_p, px_media, px_std,
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, 'NO_HISTORICO',
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            pd.NaT, pd.NaT
        ])

    # ==========================================================
    # VALIDACION ENTRADA
    # ==========================================================

    if not (u_ent_inf <= px_entrada <= u_ent_sup):

        return pd.Series([
            vol_media, vol_1_p, px_media, px_std,
            u_gen_inf, u_gen_sup, u_ent_inf, u_ent_sup,
            px_entrada, minuto_entrada, tipo_ejec_ent,
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            pd.NaT, pd.NaT
        ])

    # ==========================================================
    # BUSQUEDA SALIDA
    # ==========================================================

    ventana_px = df_px[ticker].iloc[
        pos_entrada + 1:
        pos_entrada + ventana + 1
    ]

    idx_salida_final = None
    min_disparo      = np.nan
    motivo           = "TIEMPO"

    if trailing:

        u_inf_actual    = u_gen_inf
        u_sup_actual    = u_gen_sup

        dist_stop_long  = px_entrada - u_gen_inf
        dist_stop_short = u_gen_sup - px_entrada

        mejor_precio = px_entrada

        trigger = trailing_trigger_pct * px_std

    for i, (fecha_min, precio) in enumerate(ventana_px.items()):

        if pd.isna(precio):
            continue

        if trailing:

            if label == 1:

                if precio > mejor_precio:

                    avance = precio - px_entrada

                    if avance >= trigger:
                        u_inf_actual = precio - dist_stop_long

                    mejor_precio = precio

            else:

                if precio < mejor_precio:

                    avance = px_entrada - precio

                    if avance >= trigger:
                        u_sup_actual = precio + dist_stop_short

                    mejor_precio = precio

            u_chk_sup = u_sup_actual
            u_chk_inf = u_inf_actual

        else:

            u_chk_sup = u_gen_sup
            u_chk_inf = u_gen_inf

        if precio >= u_chk_sup or precio <= u_chk_inf:

            min_disparo = i + 1

            pos_disparo = pos_entrada + 1 + i

            busqueda_lat = df_px[ticker].iloc[
                pos_disparo + 4:
                pos_disparo + 250
            ].dropna()

            if not busqueda_lat.empty:

                idx_salida_final = busqueda_lat.index[0]
                motivo = "UMBRAL"

                break

    # ==========================================================
    # SALIDA POR TIEMPO
    # ==========================================================

    if idx_salida_final is None:

        bus_t = df_px[ticker].iloc[
            pos_entrada + ventana + 4:
            pos_entrada + ventana + 250
        ].dropna()

        if not bus_t.empty:

            idx_salida_final = bus_t.index[0]
            min_disparo      = ventana

        else:

            return pd.Series([
                vol_media, vol_1_p, px_media, px_std,
                u_gen_inf, u_gen_sup, u_ent_inf, u_ent_sup,
                px_entrada, minuto_entrada, tipo_ejec_ent,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                idx_entrada, pd.NaT
            ])

    # ==========================================================
    # RESULTADOS
    # ==========================================================

    px_salida = round(
        df_px.at[idx_salida_final, ticker],
        2
    )

    min_ejec_sal = (
        df_vol.index.get_loc(idx_salida_final) - pos_entrada
    )

    tipo_ejec_sal = (
        "APERTURA"
        if idx_salida_final.date() > idx_entrada.date()
        else "SESION"
    )

    pnl = round(
        px_salida - px_entrada
        if label == 1
        else px_entrada - px_salida,
        2
    )

    res = (
        "GANA"
        if pnl > 0
        else "PIERDE"
        if pnl < 0
        else "EMPATE"
    )

    return pd.Series([
        vol_media, vol_1_p, px_media, px_std,
        u_gen_inf, u_gen_sup, u_ent_inf, u_ent_sup,
        px_entrada, minuto_entrada, tipo_ejec_ent,
        px_salida, min_disparo, min_ejec_sal, tipo_ejec_sal,
        pnl, res, motivo,
        idx_entrada, idx_salida_final
    ])

class TradingSimulator:
    def __init__(self,capital_inicial=20000,ventana=30,trailing=True,trailing_trigger_pct=0.02,
                 usar_limite_exposicion=True,limite_exposicion_pct=0.05,perc_riesgo=0.005):

        self.capital_inicial       = capital_inicial
        self.capital               = capital_inicial
        self.ventana               = ventana
        self.trailing              = trailing
        self.trailing_trigger_pct  = trailing_trigger_pct
        self.usar_limite_exposicion = usar_limite_exposicion  # NUEVO
        self.limite_exposicion_pct  = limite_exposicion_pct   # NUEVO
        self.perc_riesgo            = perc_riesgo             # NUEVO
        self.posiciones            = {}
        self.cola_salidas          = {}
        self.movimientos_caja      = []
        self.reporte_diario        = []

    def _liquidar_salidas_hasta(self, hasta_ts, df_px):
        for ticker in list(self.cola_salidas.keys()):
            ts_vencidos = sorted([
                ts for ts in self.cola_salidas[ticker]
                if ts <= hasta_ts
            ])
            for ts in ts_vencidos:
                for pos in self.cola_salidas[ticker][ts]:
                    if ts in df_px.index and ticker in df_px.columns:
                        px_sal = round(df_px.at[ts, ticker], 2)
                    else:
                        continue
                    cantidad = pos['cant']
                    comision = round(0.005 * cantidad, 2)
                    caja = (
                        (px_sal * cantidad) - comision if pos['pred'] == 1
                        else -(px_sal * cantidad) - comision
                    )
                    self.capital += caja
                    self.movimientos_caja.append((ts, caja, f"Salida {ticker}"))

                    if ticker in self.posiciones:
                        for i, p in enumerate(self.posiciones[ticker]):
                            if p['pred'] == pos['pred'] and p['cant'] == cantidad:
                                self.posiciones[ticker].pop(i)
                                break
                        if not self.posiciones[ticker]:
                            del self.posiciones[ticker]

                del self.cola_salidas[ticker][ts]

            if not self.cola_salidas[ticker]:
                del self.cola_salidas[ticker]

    def _abrir_posicion(self, ticker, volumen, px_ent, pred,
                        ts_entrada, ts_salida):
        comision = round(0.005 * volumen, 2)
        caja = (
            -(px_ent * volumen) - comision if pred == 1
            else (px_ent * volumen) - comision
        )
        self.capital += caja
        self.movimientos_caja.append((ts_entrada, caja, f"Entrada {ticker}"))

        if ticker not in self.posiciones:
            self.posiciones[ticker] = []
        self.posiciones[ticker].append({'cant': volumen, 'px_ent': px_ent, 'pred': pred})

        if ticker not in self.cola_salidas:
            self.cola_salidas[ticker] = {}
        if ts_salida not in self.cola_salidas[ticker]:
            self.cola_salidas[ticker][ts_salida] = []
        self.cola_salidas[ticker][ts_salida].append({'cant': volumen, 'pred': pred})

    def ejecutar_dia(
        self,
        df_señales_hoy,
        df_vol,
        df_px,
        df_pendientes_ayer=None,
    ):
        """
        Procesa en un único flujo cronológico con capital compartido:
          - PENDIENTE_SALIR de ayer  → se liquidan PRIMERO (devuelven capital)
          - señales nuevas de hoy    → loop por ts_entrada_real
          - PENDIENTE_ENTRAR de ayer → compiten por capital en el mismo ts
     
        Devuelve: (df_resultado_hoy, df_pendientes_resueltos)
        """
     
        tiene_nuevas     = df_señales_hoy is not None and not df_señales_hoy.empty
        tiene_pendientes = df_pendientes_ayer is not None and not df_pendientes_ayer.empty
     
        nombres = [
            'vol_media_10d', 'vol_1_porc', 'px_media_10d', 'px_std_10d',
            'u_gen_inf', 'u_gen_sup', 'u_ent_inf', 'u_ent_sup',
            'px_entrada', 'minuto_entrada', 'tipo_ejec_entrada',
            'px_salida', 'min_disparo_umbral', 'min_ejecucion_salida',
            'tipo_ejec_salida', 'pnl_unitario', 'res_tag',
            'motivo_salida', 'ts_entrada_real', 'ts_salida_real',
        ]
     
        # ------------------------------------------------------------------ #
        # 1. Métricas señales nuevas                                          #
        # ------------------------------------------------------------------ #
     
        if tiene_nuevas:
            df_nuevas = df_señales_hoy.copy()
            df_nuevas['_origen'] = 'HOY'
            df_nuevas[nombres] = df_nuevas.apply(
                lambda x: obtener_metricas(
                    x, df_vol, df_px,
                    ventana=self.ventana,
                    trailing=self.trailing,
                    trailing_trigger_pct=self.trailing_trigger_pct,
                ),
                axis=1,
            )
        else:
            df_nuevas = pd.DataFrame()
     
        # ------------------------------------------------------------------ #
        # 2. Métricas pendientes — recalcular TODAS para obtener              #
        #    ts_salida_real y px_salida actualizados con datos de hoy         #
        # ------------------------------------------------------------------ #
     
        if tiene_pendientes:
            df_pend = df_pendientes_ayer.copy()
            df_pend['_origen'] = 'PENDIENTE'
            df_pend[nombres] = df_pend.apply(
                lambda x: obtener_metricas(
                    x, df_vol, df_px,
                    ventana=self.ventana,
                    trailing=self.trailing,
                    trailing_trigger_pct=self.trailing_trigger_pct,
                ),
                axis=1,
            )
        else:
            df_pend = pd.DataFrame()
     
        # ------------------------------------------------------------------ #
        # 3. audit_data — inicializar entradas                                #
        # ------------------------------------------------------------------ #
     
        audit_data = {}
     
        if tiene_nuevas:
            for idx in df_nuevas.index:
                audit_data[('HOY', idx)] = _audit_entry_vacio()
            for idx, row in df_nuevas.iterrows():
                es_pendiente = (
                    row['tipo_ejec_entrada'] == 'PENDIENTE_ENTRAR'
                    or (pd.isna(row['px_entrada']) and pd.isna(row['u_gen_inf']))
                )
                if es_pendiente:
                    audit_data[('HOY', idx)]['estado']          = 'PENDIENTE_ENTRAR'
                    audit_data[('HOY', idx)]['tipo_ejec_final'] = 'PENDIENTE_ENTRAR'
     
        if tiene_pendientes:
            for idx, row in df_pend.iterrows():
                audit_data[('PEND', idx)] = _audit_entry_vacio()
                estado_orig = df_pendientes_ayer.at[idx, 'estado']
     
                if estado_orig == 'PENDIENTE_SALIR':
                    # Conservar cant_negociada original; el resto se actualiza al liquidar
                    audit_data[('PEND', idx)]['estado']          = 'PENDIENTE_SALIR'
                    audit_data[('PEND', idx)]['tipo_ejec_final'] = df_pendientes_ayer.at[idx, 'tipo_ejec_entrada'] \
                                                                    if 'tipo_ejec_entrada' in df_pendientes_ayer.columns else np.nan
                    audit_data[('PEND', idx)]['cant_negociada']  = df_pendientes_ayer.at[idx, 'cant_negociada']
                else:
                    # PENDIENTE_ENTRAR: detectar nuevo estado con métricas recalculadas
                    row_c = df_pend.loc[idx]
                    if pd.isna(row_c['px_entrada']) and pd.isna(row_c['u_gen_inf']):
                        audit_data[('PEND', idx)]['estado']          = 'PENDIENTE_ENTRAR'
                        audit_data[('PEND', idx)]['tipo_ejec_final'] = 'PENDIENTE_ENTRAR'
                    elif pd.isna(row_c.get('ts_entrada_real')):
                        audit_data[('PEND', idx)]['estado']          = 'NO NEGOCIADO'
                        audit_data[('PEND', idx)]['tipo_ejec_final'] = 'FUERA_UMBRAL'
                    else:
                        # Tiene ts_entrada_real (con o sin ts_salida_real)
                        # → debe entrar al loop para abrir posición
                        audit_data[('PEND', idx)]['estado']          = 'PENDIENTE_ENTRAR'
                        audit_data[('PEND', idx)]['tipo_ejec_final'] = row_c['tipo_ejec_entrada']
     
        # ------------------------------------------------------------------ #
        # 4. REGISTRAR PENDIENTE_SALIR DE AYER EN cola_salidas               #
        #    No los liquidamos aquí — los metemos en la cola para que         #
        #    _liquidar_salidas_hasta los ejecute en su ts_salida_real         #
        #    exacto dentro del loop. Así respetamos el orden cronológico:     #
        #    si CTAS entra a las 09:30 y TRGP sale a las 10:06, el capital   #
        #    de TRGP NO está disponible cuando entra CTAS.                    #
        # ------------------------------------------------------------------ #
     
        if tiene_pendientes:
            mask_salir_ayer = df_pendientes_ayer['estado'] == 'PENDIENTE_SALIR'
            for idx in df_pend[mask_salir_ayer].index:
                ak   = ('PEND', idx)
                fila = df_pend.loc[idx]       # métricas recalculadas
                orig = df_pendientes_ayer.loc[idx]
     
                ts_sal = fila.get('ts_salida_real')
                px_sal = fila.get('px_salida')
                cant   = orig.get('cant_negociada', 0)
                pred   = orig['Pred_label']
                ticker = orig['Tickers Mapeados']
                px_ent = fila.get('px_entrada') if pd.notna(fila.get('px_entrada')) \
                         else orig.get('px_entrada')
     
                audit_data[ak]['cant_negociada'] = cant
     
                if pd.notna(ts_sal) and pd.notna(px_sal) and cant > 0:
                    # Registrar en cola para que el loop lo liquide en orden
                    if ticker not in self.cola_salidas:
                        self.cola_salidas[ticker] = {}
                    if ts_sal not in self.cola_salidas[ticker]:
                        self.cola_salidas[ticker][ts_sal] = []
                    self.cola_salidas[ticker][ts_sal].append({'cant': cant, 'pred': pred})
     
                    # Guardar datos necesarios para el audit post-liquidación
                    # Los almacenamos en un dict auxiliar que consultamos al final
                    if not hasattr(self, '_pend_salir_audit'):
                        self._pend_salir_audit = {}
                    self._pend_salir_audit[ticker] = self._pend_salir_audit.get(ticker, [])
                    self._pend_salir_audit[ticker].append({
                        'ak': ak, 'cant': cant, 'pred': pred,
                        'px_ent': px_ent, 'px_sal': px_sal,
                        'ts_sal': ts_sal,
                    })
                    audit_data[ak]['caja_ent'] = round(px_ent * cant, 2) if pd.notna(px_ent) else np.nan
                    audit_data[ak]['estado']   = 'PENDIENTE_SALIR'  # se actualizará a NEGOCIADO tras el loop
                else:
                    # Sin salida disponible → sigue PENDIENTE_SALIR
                    audit_data[ak]['estado'] = 'PENDIENTE_SALIR'
     
        # ------------------------------------------------------------------ #
        # 5. Loop cronológico unificado por ts_entrada_real                   #
        # ------------------------------------------------------------------ #
     
        filas_activas_hoy  = df_nuevas if tiene_nuevas else pd.DataFrame()
     
        # Para pendientes: solo PENDIENTE_ENTRAR que ahora tienen entrada disponible
        if tiene_pendientes:
            filas_activas_pend = df_pend[
                df_pend.index.map(
                    lambda i: audit_data.get(('PEND', i), {}).get('estado') == 'PENDIENTE_ENTRAR'
                              and pd.notna(df_pend.at[i, 'ts_entrada_real'])
                              and df_pend.at[i, 'tipo_ejec_entrada'] != 'PENDIENTE_ENTRAR'
                              and pd.notna(df_pend.at[i, 'px_entrada'])
                )
            ]
        else:
            filas_activas_pend = pd.DataFrame()
     
        ts_nuevas = (
            set(filas_activas_hoy['ts_entrada_real'].dropna().unique())
            if not filas_activas_hoy.empty else set()
        )
        ts_pend = (
            set(filas_activas_pend['ts_entrada_real'].dropna().unique())
            if not filas_activas_pend.empty else set()
        )
        todos_ts = sorted(ts_nuevas | ts_pend)
     
        for f_entrada in todos_ts:
     
            self._liquidar_salidas_hasta(f_entrada, df_px)
     
            capital_disponible = self.capital
            ordenes_propuestas = []
     
            # Señales nuevas en este ts
            if not filas_activas_hoy.empty:
                grupo_hoy = filas_activas_hoy[
                    (filas_activas_hoy['ts_entrada_real'] == f_entrada) &
                    filas_activas_hoy['px_entrada'].notna() &
                    (filas_activas_hoy['tipo_ejec_entrada'] != 'PENDIENTE_ENTRAR')
                ]
                _proponer_ordenes(
                    grupo        = grupo_hoy,
                    origen       = 'HOY',
                    capital_disp = capital_disponible,
                    sim          = self,
                    audit_data   = audit_data,
                    ordenes      = ordenes_propuestas,
                )
     
            # PENDIENTE_ENTRAR resueltos en este ts
            if not filas_activas_pend.empty:
                grupo_pend = filas_activas_pend[
                    filas_activas_pend['ts_entrada_real'] == f_entrada
                ]
                _proponer_ordenes(
                    grupo        = grupo_pend,
                    origen       = 'PEND',
                    capital_disp = capital_disponible,
                    sim          = self,
                    audit_data   = audit_data,
                    ordenes      = ordenes_propuestas,
                )
     
            if not ordenes_propuestas:
                continue
     
            propuestas_df = pd.DataFrame(ordenes_propuestas)
     
            for ticker, ordenes_tk in propuestas_df.groupby('ticker'):
                compras = ordenes_tk[ordenes_tk['pred'] == 1].sort_values('prob', ascending=False)
                ventas  = ordenes_tk[ordenes_tk['pred'] == 0].sort_values('prob', ascending=True)
                neto    = compras['cant'].sum() - ventas['cant'].sum()
     
                if neto == 0:
                    continue
     
                pred_final = 1 if neto > 0 else 0
                orden      = compras if neto > 0 else ventas
                f_valida   = orden.iloc[0]
                vol_final  = abs(neto)
                idx_v      = f_valida['original_row_idx']
                origen_v   = f_valida['origen']
                ts_sal     = f_valida['ts_salida_real']
                ak         = (origen_v, idx_v)
     
                self._abrir_posicion(
                    ticker     = ticker,
                    volumen    = vol_final,
                    px_ent     = round(df_px.at[f_entrada, ticker], 2),
                    pred       = pred_final,
                    ts_entrada = f_entrada,
                    ts_salida  = ts_sal if pd.notna(ts_sal) else df_px.index[-1],
                )
     
                comision = round(0.005 * vol_final, 2)
                caja_ent = round(f_valida['px_ent'] * vol_final, 2)
     
                if pd.notna(ts_sal):
                    res_neto = (
                        (f_valida['px_salida'] - f_valida['px_ent']) * vol_final
                        if pred_final == 1
                        else (f_valida['px_ent'] - f_valida['px_salida']) * vol_final
                    ) - comision * 2
     
                    audit_data[ak]['cant_negociada'] = vol_final
                    audit_data[ak]['caja_ent']       = caja_ent
                    audit_data[ak]['caja_sal']       = round(f_valida['px_salida'] * vol_final, 2)
                    audit_data[ak]['res_neto']       = round(res_neto, 2)
                    audit_data[ak]['estado']         = 'NEGOCIADO'
                else:
                    audit_data[ak]['cant_negociada'] = vol_final
                    audit_data[ak]['caja_ent']       = caja_ent
                    audit_data[ak]['caja_sal']       = np.nan
                    audit_data[ak]['res_neto']       = np.nan
                    audit_data[ak]['estado']         = 'PENDIENTE_SALIR'
     
        # ------------------------------------------------------------------ #
        # 6. Liquidar salidas restantes en cola                               #
        # ------------------------------------------------------------------ #
     
        for ts in sorted([ts for c in self.cola_salidas.values() for ts in c.keys()]):
            self._liquidar_salidas_hasta(ts, df_px)
     
        # ------------------------------------------------------------------ #
        # 6b. Actualizar audit de PENDIENTE_SALIR de ayer                     #
        #     _liquidar_salidas_hasta ya actualizó self.capital y              #
        #     movimientos_caja. Ahora actualizamos audit_data con res_neto     #
        #     y estado final.                                                  #
        # ------------------------------------------------------------------ #
     
        if hasattr(self, '_pend_salir_audit'):
            for ticker, entradas in self._pend_salir_audit.items():
                for e in entradas:
                    ak     = e['ak']
                    cant   = e['cant']
                    pred   = e['pred']
                    px_ent = e['px_ent']
                    px_sal = e['px_sal']
     
                    if pd.notna(px_sal) and pd.notna(px_ent) and cant > 0:
                        comision = round(0.005 * cant, 2)
                        res_neto = (
                            (px_sal - px_ent) * cant if pred == 1
                            else (px_ent - px_sal) * cant
                        ) - comision * 2
     
                        audit_data[ak]['caja_sal'] = round(px_sal * cant, 2)
                        audit_data[ak]['res_neto'] = round(res_neto, 2)
                        audit_data[ak]['estado']   = 'NEGOCIADO'
     
            del self._pend_salir_audit  # limpiar para no contaminar próximas ejecuciones
     
        # ------------------------------------------------------------------ #
        # 7. Construir outputs                                                 #
        # ------------------------------------------------------------------ #
     
        df_resultado = pd.DataFrame()
        if tiene_nuevas:
            df_resultado = _construir_df_resultado(
                df=df_nuevas,
                audit_data=audit_data,
                origen='HOY',
                cols_base=list(df_señales_hoy.columns),
            )
     
        df_pend_resueltos = pd.DataFrame()
        if tiene_pendientes:
            df_pend_resueltos = _construir_df_resultado(
                df=df_pend,
                audit_data=audit_data,
                origen='PEND',
                cols_base=list(df_pendientes_ayer.columns),
            )
     
        return df_resultado, df_pend_resueltos
    
    def construir_balance(self, df_todo, df_px, fecha_inicio=None):
 
        self.reporte_diario = []
        comision = 0.005
     
        eventos = []
     
        for _, op in df_todo.iterrows():
     
            cant = op.get('cant_negociada', 0)
            if cant == 0 or pd.isna(cant):
                continue
     
            pred = op['Pred_label']
     
            # ─────────────────────────────
            # ENTRADA
            # ─────────────────────────────
            ts_ent = op.get('ts_entrada_real')
            if pd.notna(op.get('caja_ent')) and pd.notna(ts_ent):
     
                # Ignorar entradas anteriores a fecha_inicio:
                # ya están incorporadas en capital_inicial
                if fecha_inicio is None or pd.Timestamp(ts_ent).date() >= fecha_inicio:
     
                    if pred == 1:
                        flujo_ent = -op['caja_ent']
                    else:
                        flujo_ent = +op['caja_ent']
     
                    flujo_ent -= cant * comision
                    eventos.append((ts_ent, flujo_ent))
     
            # ─────────────────────────────
            # SALIDA
            # ─────────────────────────────
            ts_sal = op.get('ts_salida_real')
            if pd.notna(op.get('caja_sal')) and pd.notna(ts_sal):
     
                # Las salidas siempre se incluyen si ocurren en fecha_inicio o después
                if fecha_inicio is None or pd.Timestamp(ts_sal).date() >= fecha_inicio:
     
                    if pred == 1:
                        flujo_sal = +op['caja_sal']
                    else:
                        flujo_sal = -op['caja_sal']
     
                    flujo_sal -= cant * comision
                    eventos.append((ts_sal, flujo_sal))
     
        # ─────────────────────────────
        # ORDENAR Y ACUMULAR CASH
        # ─────────────────────────────
        df_eventos = pd.DataFrame(eventos, columns=['ts', 'flujo']).sort_values('ts')
     
        if not df_eventos.empty:
            df_eventos['capital'] = self.capital_inicial + df_eventos['flujo'].cumsum()
        else:
            df_eventos = pd.DataFrame(columns=['ts', 'flujo', 'capital'])
     
        # ─────────────────────────────
        # POR DÍAS
        # ─────────────────────────────
        dias = sorted(set(df_px.index.date))
     
        if fecha_inicio:
            dias = [d for d in dias if d >= fecha_inicio]
     
        for dia in dias:
     
            cierre_dia = pd.Timestamp(f"{dia} 16:00:00")
     
            df_filtrado = df_eventos[df_eventos['ts'] <= cierre_dia]
     
            capital_cash = (
                df_filtrado.iloc[-1]['capital']
                if not df_filtrado.empty
                else self.capital_inicial
            )
     
            # Valor posiciones abiertas a cierre
            dia_data  = df_px.loc[df_px.index.date == dia]
            cierre_16 = dia_data[dia_data.index.time == pd.Timestamp("16:00:00").time()]
     
            if cierre_16.empty and not dia_data.empty:
                cierre_16 = dia_data.iloc[[-1]]
     
            valor_pos = 0.0
     
            for _, op in df_todo.iterrows():
     
                ts_ent = op['ts_entrada_real']
                ts_sal = op['ts_salida_real']
     
                if pd.notna(ts_ent) and ts_ent <= cierre_dia:
                    if pd.isna(ts_sal) or ts_sal > cierre_dia:
     
                        ticker = op['Tickers Mapeados']
     
                        if ticker not in cierre_16.columns:
                            continue
     
                        px_c = cierre_16[ticker].dropna()
                        if px_c.empty:
                            continue
     
                        px_c = px_c.iloc[-1]
                        cant = op.get('cant_negociada', 0)
     
                        if op['Pred_label'] == 1:
                            valor_pos += cant * px_c
                        else:
                            valor_pos -= cant * px_c
     
            self.reporte_diario.append({
                'fecha':             dia,
                'capital_cash':      round(capital_cash, 2),
                'valor_posiciones':  round(valor_pos, 2),
                'equity_total':      round(capital_cash + valor_pos, 2),
            })
     
        return pd.DataFrame(self.reporte_diario)

# ======================================================================= #
# Helpers — pegar FUERA de la clase, en la misma celda o antes            #
# ======================================================================= #
 
def _audit_entry_vacio():
    return {
        'cap_disponible':         np.nan,
        'cap_riesgo':             np.nan,
        'distancia_stop':         np.nan,
        'cant_teorica_riesgo':    np.nan,
        'cant_limite_exposicion': np.nan,
        'cant_negociada':         np.nan,
        'caja_ent':               np.nan,
        'caja_sal':               np.nan,
        'res_neto':               np.nan,
        'tipo_ejec_final':        np.nan,
        'estado':                 'NO NEGOCIADO',
    }
 
 
def _proponer_ordenes(grupo, origen, capital_disp, sim, audit_data, ordenes):
    """
    Evalúa un grupo de filas y añade órdenes propuestas a la lista `ordenes`.
    Comparte el mismo capital_disp (snapshot de self.capital en ese momento).
    """
    for noti, sub_grupo in grupo.groupby('Fila Noticia'):
        for pred_val in [0, 1]:
            filas = sub_grupo[sub_grupo['Pred_label'] == pred_val]
            n     = len(filas)
            if n == 0:
                continue
 
            prob        = filas['Prob_up'].iloc[0]
            perc_riesgo = sim.perc_riesgo if 0.3 <= prob <= 0.7 else 0.01
            cap_riesgo  = np.floor((capital_disp * perc_riesgo) / n)
            limite_cash = (capital_disp * sim.limite_exposicion_pct) / n
 
            for _, f in filas.iterrows():
                ak     = (origen, f.name)
                px_ent = f['px_entrada']
 
                dist_sl = round(
                    px_ent - f['u_gen_inf'] if pred_val == 1
                    else f['u_gen_sup'] - px_ent,
                    2,
                )
                dist_sl     = max(dist_sl, 0.01)
                cant_riesgo = int(np.floor(cap_riesgo / dist_sl))
                cant_exp    = int(np.floor(limite_cash / px_ent))
 
                if pd.isna(f['vol_1_porc']) or f['vol_1_porc'] <= 0:
                    cantidad = 0
                    audit_data[ak]['cant_negociada']  = 0
                    audit_data[ak]['tipo_ejec_final'] = 'NO_NEGOCIADO_SIN_HISTORICO'
                else:
                    vol_disp = int(f['vol_1_porc'])
                    if sim.usar_limite_exposicion:
                        cantidad = min(cant_riesgo, vol_disp, cant_exp)
                    else:
                        cantidad = min(cant_riesgo, vol_disp)
 
                audit_data[ak]['cap_disponible']         = round(capital_disp, 2)
                audit_data[ak]['cap_riesgo']             = round(cap_riesgo, 2)
                audit_data[ak]['distancia_stop']         = dist_sl
                audit_data[ak]['cant_teorica_riesgo']    = cant_riesgo
                audit_data[ak]['cant_limite_exposicion'] = cant_exp
 
                if cantidad > 0:
                    audit_data[ak]['cant_negociada']  = cantidad
                    audit_data[ak]['tipo_ejec_final'] = f['tipo_ejec_entrada']
                    ordenes.append({
                        'ticker':           f['Tickers Mapeados'],
                        'pred':             pred_val,
                        'prob':             prob,
                        'cant':             cantidad,
                        'px_ent':           px_ent,
                        'original_row_idx': f.name,
                        'origen':           origen,
                        'px_salida':        f['px_salida'],
                        'ts_salida_real':   f['ts_salida_real'],
                        'dist_sl':          dist_sl,
                        'cant_riesgo':      cant_riesgo,
                        'cant_exposicion':  cant_exp,
                        'cap_riesgo':       cap_riesgo,
                    })
                else:
                    if pd.isna(audit_data[ak].get('tipo_ejec_final')):
                        audit_data[ak]['cant_negociada']  = 0
                        audit_data[ak]['tipo_ejec_final'] = 'SIN_LIQUIDEZ'
 
 
def _construir_df_resultado(df, audit_data, origen, cols_base):
    """Aplica audit_data al df y devuelve el dataframe con columnas ordenadas."""
 
    audit_local = {
        idx: audit_data[(origen, idx)]
        for idx in df.index
        if (origen, idx) in audit_data
    }
    audit_df = pd.DataFrame.from_dict(audit_local, orient='index')
 
    df = df.copy()
    df['tipo_ejec_entrada'] = audit_df['tipo_ejec_final'].reindex(df.index)
 
    cols_limpiar = [
        'px_salida', 'min_disparo_umbral', 'min_ejecucion_salida',
        'tipo_ejec_salida', 'pnl_unitario', 'res_tag',
        'ts_entrada_real', 'ts_salida_real',
    ]
    for idx in df.index:
        estado = audit_local.get(idx, {}).get('estado', '')
        if estado in ('NO NEGOCIADO', 'PENDIENTE_ENTRAR'):
            for col in cols_limpiar:
                if col in df.columns:
                    df.at[idx, col] = np.nan
 
    df = df.join(
        audit_df[[
            'cap_disponible', 'cap_riesgo', 'distancia_stop',
            'cant_teorica_riesgo', 'cant_limite_exposicion',
            'cant_negociada', 'caja_ent', 'caja_sal', 'res_neto', 'estado',
        ]],
        rsuffix='_new',
    )
 
    for col in ['cap_disponible', 'cap_riesgo', 'distancia_stop',
                'cant_teorica_riesgo', 'cant_limite_exposicion',
                'cant_negociada', 'caja_ent', 'caja_sal', 'res_neto', 'estado']:
        if f'{col}_new' in df.columns:
            df[col] = df[f'{col}_new']
            df.drop(columns=[f'{col}_new'], inplace=True)
 
    cols_nuevas = [
        'vol_media_10d', 'vol_1_porc', 'px_media_10d', 'px_std_10d',
        'cap_disponible', 'cap_riesgo', 'distancia_stop',
        'cant_teorica_riesgo', 'cant_limite_exposicion', 'cant_negociada',
        'u_gen_inf', 'u_gen_sup', 'u_ent_inf', 'u_ent_sup',
        'px_entrada', 'minuto_entrada', 'tipo_ejec_entrada',
        'px_salida', 'min_disparo_umbral', 'min_ejecucion_salida',
        'tipo_ejec_salida', 'motivo_salida', 'pnl_unitario', 'res_tag',
        'ts_entrada_real', 'ts_salida_real',
        'caja_ent', 'caja_sal', 'res_neto', 'estado',
    ]
    cols_finales = cols_base + [c for c in cols_nuevas if c not in cols_base]
    df = df.drop(columns=['_origen'], errors='ignore')
    return df[[c for c in cols_finales if c in df.columns]]


# Reparar tabla de precios de cierre
if not precios_cierre_sesion.empty and 'Date' in precios_cierre_sesion.columns:
    precios_cierre_sesion['Date'] = pd.to_datetime(precios_cierre_sesion['Date'])
    precios_cierre_sesion.set_index('Date', inplace=True)
    precios_cierre_sesion.sort_index(inplace=True)
    # TRUCO CLAVE: Forzar conversión de todas las columnas (tickers) a números
    precios_cierre_sesion = precios_cierre_sesion.apply(pd.to_numeric, errors='coerce')

# Reparar tabla de volúmenes de sesión
if not volumenes_sesion.empty and 'Date' in volumenes_sesion.columns:
    volumenes_sesion['Date'] = pd.to_datetime(volumenes_sesion['Date'])
    volumenes_sesion.set_index('Date', inplace=True)
    volumenes_sesion.sort_index(inplace=True)
    # TRUCO CLAVE: Forzar conversión de todas las columnas (tickers) a números
    volumenes_sesion = volumenes_sesion.apply(pd.to_numeric, errors='coerce')
    

CAPITAL_POR_DEFECTO = 20000

if historico_balance.empty:
    print(f"La tabla 'daily_balance' está vacía. Usando capital inicial por defecto: ${CAPITAL_POR_DEFECTO}")
    capital_hoy = CAPITAL_POR_DEFECTO
else:
    print("¡Histórico contable recuperado! Extrayendo el último capital disponible...")
    # 1. Aseguramos que los valores sean numéricos
    historico_balance['capital_cash'] = pd.to_numeric(historico_balance['capital_cash'])
    
    # 2. Ordenamos cronológicamente por fecha para garantizar que el último registro sea el de ayer
    historico_balance = historico_balance.sort_values('fecha').reset_index(drop=True)
    
    # 3. Extraemos el último capital_cash registrado en la base de datos
    capital_hoy = float(historico_balance['capital_cash'].iloc[-1])
    print(f"Capital recuperado para la sesión de hoy: ${capital_hoy:.2f}")

# ==============================================================================
# INSTANCIA Y EJECUCIÓN DEL SIMULADOR
# ==============================================================================
# Pasamos la variable dinámica 'capital_hoy' en lugar del número fijo 20000
sim = TradingSimulator(capital_inicial=capital_hoy, ventana=30)

# sim = TradingSimulator(capital_inicial=20000, ventana=30)

df_resultado, df_pendientes_resueltos = sim.ejecutar_dia(
    df_señales_hoy     = seynales_modelo,
    # Si está vacío pasamos None, si tiene filas pasamos el DataFrame
    df_pendientes_ayer = None if pendientes_ayer.empty else pendientes_ayer,
    df_vol             = volumenes_sesion,
    df_px              = precios_cierre_sesion,
)


df_todo   = pd.concat([df_resultado, df_pendientes_resueltos], ignore_index=True)
df_balance = sim.construir_balance(
    df_todo,
    precios_cierre_sesion,
    fecha_inicio=datetime.date(2026, 5, 15)
)
# Filtrar solo pendientes
df_pendientes = df_todo[
    df_todo['estado'].isin(['PENDIENTE_SALIR', 'PENDIENTE_ENTRAR'])
].copy()


# ==============================================================================
# PROCESO 1: GUARDAR DF_PENDIENTES EN "pending_trade" (REEMPLAZAR TABLA ENTERA)
# ==============================================================================
NOMBRE_TABLA_PENDIENTES = "pending_trade"

# 1. Borrar la tabla antigua para asegurar que no queden registros pasados
try:
    print(f"Eliminando tabla antigua '{NOMBRE_TABLA_PENDIENTES}'...")
    dynamodb_client.delete_table(TableName=NOMBRE_TABLA_PENDIENTES)
    waiter = dynamodb_client.get_waiter("table_not_exists")
    waiter.wait(TableName=NOMBRE_TABLA_PENDIENTES)
except dynamodb_client.exceptions.ResourceNotFoundException:
    pass

# 2. Recrear la tabla completamente limpia
print(f"Creando nueva tabla limpia '{NOMBRE_TABLA_PENDIENTES}'...")
dynamodb_client.create_table(
    TableName=NOMBRE_TABLA_PENDIENTES,
    KeySchema=[
        {"AttributeName": "Tickers Mapeados", "KeyType": "HASH"},  # String
        {"AttributeName": "Fila Noticia", "KeyType": "RANGE"},  # Number
    ],
    AttributeDefinitions=[
        {"AttributeName": "Tickers Mapeados", "AttributeType": "S"},
        {"AttributeName": "Fila Noticia", "AttributeType": "N"},
    ],
    BillingMode="PAY_PER_REQUEST",
)
waiter = dynamodb_client.get_waiter("table_exists")
waiter.wait(TableName=NOMBRE_TABLA_PENDIENTES)
table_pendientes = dynamodb_resource.Table(NOMBRE_TABLA_PENDIENTES)

# 3. Preparar y subir datos solo si el DataFrame NO está vacío
if df_pendientes.empty:
    print(
        f"El dataframe 'df_pendientes' está vacío. La tabla '{NOMBRE_TABLA_PENDIENTES}' quedará activa pero sin registros."
    )
else:
    df_p_prep = df_pendientes.copy()

    # Formatear columnas problemáticas (Fechas, NaT y NaN)
    for col in df_p_prep.columns:
        # Detectar columnas de tipo fecha o marcas de tiempo
        if pd.api.types.is_datetime64_any_dtype(df_p_prep[col]):
            df_p_prep[col] = (
                df_p_prep[col].astype(str).replace(["NaT", "NaN", "nat", "nan"], None)
            )

    # Convertir floats a tipo Decimal compatible con DynamoDB via JSON
    df_p_json = df_p_prep.to_json(orient="records")
    items_pendientes = json.loads(df_p_json, parse_float=Decimal)

    print(f"Subiendo {len(items_pendientes)} registros a {NOMBRE_TABLA_PENDIENTES}...")
    with table_pendientes.batch_writer() as batch:
        for item in items_pendientes:
            # Forzar tipos estrictos en las llaves primarias requeridas por AWS
            item["Tickers Mapeados"] = str(item["Tickers Mapeados"])
            item["Fila Noticia"] = int(item["Fila Noticia"])

            # Filtro extremo: Elimina nulos, textos 'None', strings vacías o floats NaN remanentes
            item_limpio = {
                k: v
                for k, v in item.items()
                if v is not None
                and str(v) not in ["None", "NaN", "NaT", "nan", "nat"]
                and v != ""
            }

            batch.put_item(Item=item_limpio)
    print("¡Tabla de pendientes actualizada con éxito!")


# ==============================================================================
# PROCESO 2: GUARDAR DF_BALANCE EN "daily_balance" (ACUMULAR REGISTROS)
# ==============================================================================
NOMBRE_TABLA_BALANCE = "daily_balance"
table_balance = dynamodb_resource.Table(NOMBRE_TABLA_BALANCE)

if df_balance.empty:
    print("⚠️ Advertencia: 'df_balance' está vacío, omitiendo guardado contable.")
else:
    df_b_prep = df_balance.copy()

    # Asegurar que la fecha contable sea texto plano
    df_b_prep["fecha"] = df_b_prep["fecha"].astype(str)

    # Convertir la fila única a tipos Decimal
    df_b_json = df_b_prep.to_json(orient="records")
    items_balance = json.loads(df_b_json, parse_float=Decimal)

    print(f"Acumulando registro de balance diario en '{NOMBRE_TABLA_BALANCE}'...")
    for item in items_balance:
        item["fecha"] = str(item["fecha"])

        # Limpieza estándar de nulos
        item_limpio = {
            k: v
            for k, v in item.items()
            if v is not None and str(v) not in ["None", "NaN", "nan"]
        }

        # Inserta o sobreescribe el día actual sin alterar el historial contable previo
        table_balance.put_item(Item=item_limpio)

    print(f"¡Balance del día {df_b_prep['fecha'].iloc[0]} guardado correctamente!")



