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

# Manejo de actualizacion de la tabla en dynamodb llamada uptade_wikipedia_keys
def handler(event, context):
    # Confirguro el dynamodb
    dynamodb = boto3.resource('dynamodb')
    
    # Creo la referencias a las tablas de interes
    table_changes = dynamodb.Table('clean_changes_sp500')
    table_wiki = dynamodb.Table('update_wikipedia_keys')

    # Descargo solo los Addition de cambios en el SP500 y los paso a df
    res_changes = table_changes.scan(FilterExpression=Attr('Action').eq('Addition'))
    clean_changes_sp500 = pd.DataFrame(res_changes.get('Items', []))
    
    # Descago todo y los paso a df
    res_wiki = table_wiki.scan()
    update_wikipedia_keys = pd.DataFrame(res_wiki.get('Items', []))
    
    # Reviso y me guardo los nuevos tickers
    new_tickers = clean_changes_sp500[clean_changes_sp500["Action"] == "Addition"].reset_index(drop=True)
    new_tickers['Effective Date'] = pd.to_datetime(new_tickers['Effective Date'])
    today = pd.Timestamp.now().normalize()
    new_tickers = new_tickers[new_tickers['Effective Date'] <= today]

    # Confirmo que nuevos ticker no esten en la tabla, porque puede ser que antes el ticker hubiese estado en el SP500
    real_new_tickers = new_tickers[~new_tickers['Ticker'].isin(update_wikipedia_keys['Ticker'])].reset_index(drop=True)

    # Webscrapping a wikipedia
    HEADERS = {"User-Agent": "Mozilla/5.0"}

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
    total = len(real_new_tickers)
    for _, row in real_new_tickers.iterrows():
        data = scrape_wikipedia_infobox(row["Company Name"])
        rows.append({"Ticker": row["Ticker"], "Company Name": row["Company Name"], **data})
        time.sleep(0.3)

    # Creo que df de nuevos tickers con sus categorias
    new_wikipedia_keys = pd.DataFrame(rows, columns=[
        "Ticker", "Company Name", "Predecessor",
        "Products", "Services", "Brands", "Divisions", "Subsidiaries"
    ])

    # Si existe ese df para nuevos tickers, ejecuto la limpieza correspondiente y subida de datos
    if not new_wikipedia_keys.empty:
        
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
        

        # Actualizo la tabla del dynamodb
        for _, row in new_wikipedia_keys.iterrows():
            item = {k: v for k, v in row.to_dict().items() if pd.notna(v) and v != ""}
            table_wiki.put_item(Item=item)
        
        return {"status": "success", "added": len(new_wikipedia_keys)}
    
    else:
        # Si no hay nuevos tickers, no actualizo la tabla
        return {"status": "success", "message": "new_wikipedia_keys estaba vacío, nada que actualizar"}
