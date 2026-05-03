import os
import boto3
import requests
import re
import time
import json
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from groq import Groq
from datetime import datetime
from boto3.dynamodb.conditions import Attr

def handler(event, context):
    # --- BLOQUE INICIAL: LEER DYNAMODB ---
    dynamodb = boto3.resource('dynamodb')
    # Tabla de origen de cambios
    table_changes = dynamodb.Table('clean_changes_sp500')
    # Tabla de Wikipedia (tu 'update_wikipedia_keys')
    table_wiki = dynamodb.Table('update_wikipedia_keys')

    # Lectura de las tablas para crear tus DataFrames iniciales
    res_changes = table_changes.scan(FilterExpression=Attr('Action').eq('Addition'))
    clean_changes_sp500 = pd.DataFrame(res_changes.get('Items', []))
    
    res_wiki = table_wiki.scan()
    update_wikipedia_keys = pd.DataFrame(res_wiki.get('Items', []))

    # --- TU CÓDIGO TAL CUAL (SIN TOCAR) ---
    
    # Reviso los tickers nuevos
    new_tickers = clean_changes_sp500[clean_changes_sp500["Action"] == "Addition"].reset_index(drop=True)
    new_tickers['Effective Date'] = pd.to_datetime(new_tickers['Effective Date'])
    today = pd.Timestamp.now().normalize()
    new_tickers = new_tickers[new_tickers['Effective Date'] <= today]

    # Aquí usamos el df 'update_wikipedia_keys' que leímos de DynamoDB arriba
    real_new_tickers = new_tickers[~new_tickers['Ticker'].isin(update_wikipedia_keys['Ticker'])].reset_index(drop=True)

    HEADERS = {"User-Agent": "Mozilla/5.0"}

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


    rows = []
    total = len(real_new_tickers)
    for _, row in real_new_tickers.iterrows():
        data = scrape_wikipedia_infobox(row["Company Name"])
        rows.append({"Ticker": row["Ticker"], "Company Name": row["Company Name"], **data})
        time.sleep(0.3)

    new_wikipedia_keys = pd.DataFrame(rows, columns=[
        "Ticker", "Company Name", "Predecessor",
        "Products", "Services", "Brands", "Divisions", "Subsidiaries"
    ])

    if not new_wikipedia_keys.empty:
        def clean_names(text):
            if not isinstance(text, str):
                return None
            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"\(.*?\)", "", text)
            # Permite mayúsculas en medio: McCarthy, McGregor, DeNiro, O'Brien
            names = re.findall(r"[A-Z][a-zA-Z']+(?:\s[A-Z][a-zA-Z',.]+)+", text)
            if not names:
                return None
            seen = set()
            unique_names = [n for n in names if not (n in seen or seen.add(n))]
            return ", ".join(unique_names)
    
    
        def clean_garbage(text):
            if not isinstance(text, str):
                return None
            text = re.sub(r"\(.*?\)", "", text)
            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r",\s*,", ",", text)
            return text.strip().strip(",").strip()
        
        # df_wikipedia["Founders"]   = df_wikipedia["Founders"].apply(clean_names)
        new_wikipedia_keys["Predecessor"]   = new_wikipedia_keys["Predecessor"].apply(clean_garbage)
        new_wikipedia_keys["Products"]   = new_wikipedia_keys["Products"].apply(clean_garbage)
        new_wikipedia_keys["Services"]   = new_wikipedia_keys["Services"].apply(clean_garbage)
        new_wikipedia_keys["Brands"]   = new_wikipedia_keys["Brands"].apply(clean_garbage)
        new_wikipedia_keys["Divisions"]   = new_wikipedia_keys["Divisions"].apply(clean_garbage)
        new_wikipedia_keys["Subsidiaries"]   = new_wikipedia_keys["Subsidiaries"].apply(clean_garbage)
        
        def get_unique_sorted_by_freq_multi(df, columns):
            # 1. Unificamos todas las columnas seleccionadas en una sola serie
            # .stack() convierte las columnas en una sola fila vertical ignorando NaNs
            combined_series = df[columns].stack().astype(str)
        
            # 2. Separamos por coma y aplanamos la lista
            # split(", ") crea listas, luego el list comprehension las "aplana"
            all_items = [item.strip() for sublist in combined_series.str.split(",") for item in sublist]
        
            # 3. Contamos frecuencias
            counts = Counter(all_items)
        
            # 4. Devolvemos solo las palabras (llaves) ordenadas por su conteo
            return [word for word, _ in counts.most_common()]
        
        # Llamo a la funcion
        problem_columns= ["Products", "Services"]
        keywords = get_unique_sorted_by_freq_multi(new_wikipedia_keys, problem_columns)
        
        
        client = Groq(api_key="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")
        
        
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
        
        # Procesar en batches de 20 por si tienes muchas keywords
        def classify_all(terms, batch_size=20):
            all_results = []
            for i in range(0, len(terms), batch_size):
                batch = terms[i:i+batch_size]
                results = classify_batch(batch)
                all_results.extend(results)
            return all_results
        
        
        results = classify_all(keywords)
        common_words = pd.DataFrame(results)
        
        # Crear un dict de lookup rápido
        proper_noun_set = set(
            common_words[common_words['is_proper_noun'] == True]['term'].tolist()
        )
        
        def filter_proper_nouns(cell_value):
            """Filtra una celda conservando solo los nombres propios"""
            if pd.isna(cell_value) or cell_value == 'None':
                return cell_value
        
            # Separar por coma (ajusta el separador si usas otro)
            terms = [t.strip() for t in str(cell_value).split(',')]
        
            # Conservar solo los que son nombres propios
            filtered = [t for t in terms if t in proper_noun_set]
        
            # Devolver None si no queda nada, o los términos unidos
            return ', '.join(filtered) if filtered else None
        
        df_review = new_wikipedia_keys[['Ticker', 'Company Name', 'Products', 'Services']].copy()
        df_review['Products_clean'] = new_wikipedia_keys['Products'].apply(filter_proper_nouns)
        df_review['Services_clean'] = new_wikipedia_keys['Services'].apply(filter_proper_nouns)
        
        # Columnas de cambios
        df_review['Products_changed'] = df_review['Products'] != df_review['Products_clean']
        df_review['Services_changed'] = df_review['Services'] != df_review['Services_clean']
        
        # Reordenar
        df_review = df_review[['Ticker', 'Company Name',
                                'Products', 'Products_clean', 'Products_changed',
                                'Services', 'Services_clean', 'Services_changed']]
        
        df_review_changed = df_review[
            df_review['Products_changed'] | df_review['Services_changed']
        ]
        
        new_wikipedia_keys['Products'] = df_review['Products_clean']
        new_wikipedia_keys['Services'] = df_review['Services_clean']
        
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
        
        # Solo cálculo temporal, no modifica df_wikipedia
        df_review_names = new_wikipedia_keys[['Company Name']].copy()
        
        df_review_names[['Company Name Clean', 'Removed']] = df_review_names['Company Name'].apply(
            lambda x: pd.Series(clean_company_name(x))
        )
        
        df_review_names['changed'] = df_review_names['Company Name'] != df_review_names['Company Name Clean']
        
        # 1. Creamos la serie con los datos que quieres traer
        clean_names = df_review_names['Company Name Clean']
        
        # 2. Insertamos en la posición 2 (que es la tercera columna)
        # Sintaxis: df.insert(posicion, nombre_columna, valores)
        new_wikipedia_keys.insert(2, 'Company Name Clean', clean_names)
        
        # Tus columnas de interés
        columns = [
            'Predecessor', 'Products', 'Services',
            'Brands', 'Divisions', 'Subsidiaries'
        ]
        
        def is_suspicious(text):
            if pd.isna(text):
                return False
        
            texto = str(text).strip()
            # Condición: Más de 2 palabras (split genera una lista)
            # Y que NO contenga ninguna coma
            palabras = text.split()
            if len(palabras) > 3 and ',' not in text:
                return True
            return False
        
        # Aplicamos el filtro y creamos una nueva columna indicadora
        new_wikipedia_keys['Need Revision'] = new_wikipedia_keys[columns].apply(lambda row: any(is_suspicious(val) for val in row), axis=1)
        
        # Ver solo las filas que cumplen tu criterio
        # rows_to_correct = new_wikipedia_keys[new_wikipedia_keys['Need_Revision']]
        
        columnas = [
            'Predecessor', 'Products', 'Services',
            'Brands', 'Divisions', 'Subsidiaries'
        ]
        
        def get_incorrect_columns(row):
            cols_sospechosas = []
            for col in columnas:
                texto = str(row[col]).strip()
                if pd.isna(row[col]) or texto == "" or texto.lower() == 'nan':
                    continue
        
                # Tu lógica: Más de 4 palabras y sin comas
                palabras = texto.split()
                if len(palabras) > 3 and ',' not in texto:
                    cols_sospechosas.append(col)
        
            # Devolvemos la lista de columnas (o None si está limpia)
            return cols_sospechosas if cols_sospechosas else None
        
        # Creamos la nueva columna con la lista de culpables
        new_wikipedia_keys['Incorrect Columns'] = new_wikipedia_keys.apply(get_incorrect_columns, axis=1)
        
        # Filtramos las que no son None
        rows_to_correct = new_wikipedia_keys[new_wikipedia_keys['Incorrect Columns'].notna()]
        
        
        client = Groq(api_key="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")
        
        def fix_merged_keywords(row, suspicious_columns):
            """Para una fila sospechosa, pide al LLM que corrija las columnas problemáticas"""
        
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
                    model="llama-3.1-8b-instant",  #llama-3.3-70b-versatile llama-3.1-8b-instant
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0
                )
        
                text = response.choices[0].message.content.strip()
                result = json.loads(text)
                corrections[col] = result
        
            return corrections
        
        results = []
        
        for idx, row in rows_to_correct.iterrows():
            # Parsear las columnas sospechosas de tu columna 'Columnas_Errores'
            suspicious_cols = row['Incorrect Columns']  # ya tienes esto como lista
        
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
        
        df_corrections = pd.DataFrame(results)
        
        # Aplicar correcciones directamente sobre df_wikipedia
        for _, fix in df_corrections.iterrows():
            new_wikipedia_keys.loc[fix['index'], fix['column']] = fix['corrected']
        
        new_wikipedia_keys = new_wikipedia_keys.drop(columns=[ "Need Revision", "Incorrect Columns"])
        

        # --- BLOQUE FINAL: ACTUALIZAR DYNAMODB ---
        # Solo se ejecuta si new_wikipedia_keys no está vacío
        for _, row in new_wikipedia_keys.iterrows():
            # Convertimos fila a dict y filtramos nulos para que DynamoDB no proteste
            item = {k: v for k, v in row.to_dict().items() if pd.notna(v) and v != ""}
            table_wiki.put_item(Item=item)
        
        return {"status": "success", "added": len(new_wikipedia_keys)}
    
    else:
        return {"status": "success", "message": "new_wikipedia_keys estaba vacío, nada que actualizar"}
