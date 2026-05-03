import boto3
import pandas as pd
from datetime import datetime
from boto3.dynamodb.conditions import Attr

def handler(event, context):
    try:
        dynamodb = boto3.resource('dynamodb')
        
        # Tablas
        table_historic = dynamodb.Table('historic_composition_sp500')
        table_changes = dynamodb.Table('clean_changes_sp500')

        # 1. Definir hoy
        today_str = datetime.now().date().isoformat()

        # 2. LEER ÚLTIMO REGISTRO HISTÓRICO
        # Escaneamos para obtener la composición más reciente
        res_hist = table_historic.scan()
        items_hist = res_hist.get('Items', [])
        
        if not items_hist:
            return {"status": "error", "message": "Tabla histórica vacía"}

        df_hist = pd.DataFrame(items_hist)
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        last_row = df_hist.sort_values('Date').iloc[-1]
        
        # Tickers actuales (Set para evitar duplicados)
        tickers_actuals = set(t.strip() for t in last_row['Ticker'].split(',') if t.strip())

        # 3. BUSCAR CAMBIOS PARA HOY EN LA OTRA TABLA
        # Filtramos en DynamoDB donde 'Effective Date' sea igual a hoy
        res_changes = table_changes.scan(
            FilterExpression=Attr('Effective Date').eq(today_str)
        )
        changes_today = res_changes.get('Items', [])

        if changes_today:
            print(f"Se encontraron {len(changes_today)} cambios para hoy {today_str}")
            for change in changes_today:
                ticker = str(change['Ticker']).strip()
                action = change['Action']
                
                if action == 'Addition':
                    tickers_actuals.add(ticker)
                elif action == 'Deletion':
                    tickers_actuals.discard(ticker)
        else:
            print(f"Sin cambios programados para hoy {today_str}. Se repite composición.")

        # 4. PREPARAR NUEVA FILA
        sorted_list = sorted(list(tickers_actuals))
        tickers_string = ",".join(sorted_list)
        total_companies = len(sorted_list)

        # 5. GUARDAR EN DYNAMODB
        table_historic.put_item(
            Item={
                'Date': today_str,
                'Ticker': tickers_string,
                'Total Companies': total_companies
            }
        )

        return {
            "status": "success", 
            "date": today_str, 
            "total": total_companies,
            "changes_applied": len(changes_today)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}

