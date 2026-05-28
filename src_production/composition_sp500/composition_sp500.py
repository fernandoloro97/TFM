import boto3
import pandas as pd
from datetime import datetime
from boto3.dynamodb.conditions import Attr

def handler(event, context):
    # Fijo la fecha del momento de la ejecucion
    target_date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        # Confirguro el dynamodb
        dynamodb = boto3.resource('dynamodb')
        
        # Inicializo las tablas del dynamodb
        table_historic = dynamodb.Table('historic_composition_sp500')
        table_changes = dynamodb.Table('clean_changes_sp500')

        # Leo y filtro el registro historico
        res_hist = table_historic.scan()
        items_hist = res_hist.get('Items', [])
        
        if not items_hist:
            return {"status": "error", "message": "La tabla histórica 'historic_composition_sp500' esta vacia"}

        df_hist = pd.DataFrame(items_hist)
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        
        # Miro el historico
        target_datetime = pd.to_datetime(target_date_str)
        df_hist_filtered = df_hist[df_hist['Date'] < target_datetime]
        
        if df_hist_filtered.empty:
            return {
                "status": "error", 
                "message": f"Error de consistencia: No existen registros historicos anteriores al {target_date_str} para usar como base."
            }

        # Obtengo el registro anterior a hoy
        last_row = df_hist_filtered.sort_values('Date').iloc[-1]
        base_date = last_row['Date'].strftime('%Y-%m-%d')
        print(f"Composicion base encontrada del dia anterior valido: {base_date}")
        
        # Extraigo y limpio los tickers base
        tickers_actuals = set(t.strip() for t in last_row['Ticker'].split(',') if t.strip())

        # Busco y aplico los cambios de composicion del SP500, si es que hay
        res_changes = table_changes.scan(
            FilterExpression=Attr('Effective Date').eq(target_date_str)
        )
        changes_today = res_changes.get('Items', [])

        # Actulizo tabla si hay cambios
        if changes_today:
            print(f"Encontre {len(changes_today)} cambios para aplicar")
            for change in changes_today:
                ticker = str(change['Ticker']).strip()
                action = change['Action'].strip()
                
                if action == 'Addition':
                    if ticker not in tickers_actuals:
                        tickers_actuals.add(ticker)
                        print(f"Añadido: {ticker}")
                    else:
                        print(f"Ojo: Intente añadir {ticker} pero ya existia")
                        
                elif action == 'Deletion':
                    if ticker in tickers_actuals:
                        tickers_actuals.discard(ticker)
                        print(f"Eliminado: {ticker}")
                    else:
                        print(f"Ojo: Intente eliminar {ticker} pero no existia en la base")
                else:
                    print(f"Accion desconocida ignorada: {action} para el ticker {ticker}")
                    
        # No hay cambios y repito la composicion de ayer
        else:
            print(f"Sin cambios programados para el {target_date_str}. Se mantiene la estructura del {base_date}.")

        # Ordeno los tickers y guardo
        sorted_list = sorted(list(tickers_actuals))
        tickers_string = ",".join(sorted_list)
        total_companies = len(sorted_list)

        print(f"Guardando nueva composición. Total compañias: {total_companies}")
        table_historic.put_item(
            Item={
                'Date': target_date_str,
                'Ticker': tickers_string,
                'Total Companies': total_companies
            }
        )

        print("Tabla actualizada con exito")
        return {
            "status": "success", 
            "date_processed": target_date_str, 
            "base_date_used": base_date,
            "total_companies": total_companies,
            "changes_applied": len(changes_today)
        }

    except Exception as e:
        error_msg = f"Fallo crítico en la ejecución: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {"status": "error", "message": error_msg}
