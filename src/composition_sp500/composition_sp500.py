# import boto3
# import pandas as pd
# from datetime import datetime
# from boto3.dynamodb.conditions import Attr

# def handler(event, context):
#     try:
#         dynamodb = boto3.resource('dynamodb')
        
#         # Tablas
#         table_historic = dynamodb.Table('historic_composition_sp500')
#         table_changes = dynamodb.Table('clean_changes_sp500')

#         # 1. Definir hoy
#         today_str = datetime.now().date().isoformat()

#         # 2. LEER ÚLTIMO REGISTRO HISTÓRICO
#         # Escaneamos para obtener la composición más reciente
#         res_hist = table_historic.scan()
#         items_hist = res_hist.get('Items', [])
        
#         if not items_hist:
#             return {"status": "error", "message": "Tabla histórica vacía"}

#         df_hist = pd.DataFrame(items_hist)
#         df_hist['Date'] = pd.to_datetime(df_hist['Date'])
#         last_row = df_hist.sort_values('Date').iloc[-1]
        
#         # Tickers actuales (Set para evitar duplicados)
#         tickers_actuals = set(t.strip() for t in last_row['Ticker'].split(',') if t.strip())

#         # 3. BUSCAR CAMBIOS PARA HOY EN LA OTRA TABLA
#         # Filtramos en DynamoDB donde 'Effective Date' sea igual a hoy
#         res_changes = table_changes.scan(
#             FilterExpression=Attr('Effective Date').eq(today_str)
#         )
#         changes_today = res_changes.get('Items', [])

#         if changes_today:
#             print(f"Se encontraron {len(changes_today)} cambios para hoy {today_str}")
#             for change in changes_today:
#                 ticker = str(change['Ticker']).strip()
#                 action = change['Action']
                
#                 if action == 'Addition':
#                     tickers_actuals.add(ticker)
#                 elif action == 'Deletion':
#                     tickers_actuals.discard(ticker)
#         else:
#             print(f"Sin cambios programados para hoy {today_str}. Se repite composición.")

#         # 4. PREPARAR NUEVA FILA
#         sorted_list = sorted(list(tickers_actuals))
#         tickers_string = ",".join(sorted_list)
#         total_companies = len(sorted_list)

#         # 5. GUARDAR EN DYNAMODB
#         table_historic.put_item(
#             Item={
#                 'Date': today_str,
#                 'Ticker': tickers_string,
#                 'Total Companies': total_companies
#             }
#         )

#         return {
#             "status": "success", 
#             "date": today_str, 
#             "total": total_companies,
#             "changes_applied": len(changes_today)
#         }

#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return {"status": "error", "message": str(e)}

import boto3
import pandas as pd
from datetime import datetime
from boto3.dynamodb.conditions import Attr

def handler(event, context):
    # =========================================================================
    # CONFIGURACIÓN DE LA FECHA OBJETIVO
    # =========================================================================
    # Si quieres dejarlo fijo para el 15/05/2026, usa esta línea:
    target_date_str = "2026-05-15"
    
    # NOTA OPCIONAL: Si en el futuro prefieres que sea dinámico vía evento, 
    # puedes descomentar las siguientes líneas:
    # if event and 'date' in event:
    #     target_date_str = event['date']
    # else:
    #     target_date_str = datetime.now().date().isoformat()

    print(f"--- INICIANDO EJECUCIÓN PARA LA FECHA: {target_date_str} ---")

    try:
        dynamodb = boto3.resource('dynamodb')
        
        # Inicialización de tablas
        table_historic = dynamodb.Table('historic_composition_sp500')
        table_changes = dynamodb.Table('clean_changes_sp500')

        # =========================================================================
        # 1. LEER Y FILTRAR REGISTRO HISTÓRICO ANTERIOR
        # =========================================================================
        print("Leyendo tabla histórica...")
        res_hist = table_historic.scan()
        items_hist = res_hist.get('Items', [])
        
        if not items_hist:
            return {"status": "error", "message": "La tabla histórica 'historic_composition_sp500' está completamente vacía."}

        df_hist = pd.DataFrame(items_hist)
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        
        # ROBUSTEZ: Convertimos la fecha objetivo y filtramos el histórico.
        # Solo nos interesan los días ESTRICTAMENTE ANTERIORES a la fecha objetivo.
        target_datetime = pd.to_datetime(target_date_str)
        df_hist_filtered = df_hist[df_hist['Date'] < target_datetime]
        
        if df_hist_filtered.empty:
            return {
                "status": "error", 
                "message": f"Error de consistencia: No existen registros históricos anteriores al {target_date_str} para usar como base."
            }

        # Obtenemos el registro más cercano en el pasado
        last_row = df_hist_filtered.sort_values('Date').iloc[-1]
        base_date = last_row['Date'].strftime('%Y-%m-%d')
        print(f"Composición base encontrada del día anterior válido: {base_date}")
        
        # Extraer y limpiar los tickers base
        tickers_actuals = set(t.strip() for t in last_row['Ticker'].split(',') if t.strip())

        # =========================================================================
        # 2. BUSCAR Y APLICAR CAMBIOS PROGRAMADOS
        # =========================================================================
        print(f"Buscando cambios programados en 'clean_changes_sp500' para el {target_date_str}...")
        res_changes = table_changes.scan(
            FilterExpression=Attr('Effective Date').eq(target_date_str)
        )
        changes_today = res_changes.get('Items', [])

        if changes_today:
            print(f"Se encontraron {len(changes_today)} cambios para aplicar.")
            for change in changes_today:
                ticker = str(change['Ticker']).strip()
                action = change['Action'].strip()
                
                if action == 'Addition':
                    if ticker not in tickers_actuals:
                        tickers_actuals.add(ticker)
                        print(f" [+] Añadido: {ticker}")
                    else:
                        print(f" [!] Advertencia: Se intentó añadir {ticker} pero ya existía.")
                        
                elif action == 'Deletion':
                    if ticker in tickers_actuals:
                        tickers_actuals.discard(ticker)
                        print(f" [-] Eliminado: {ticker}")
                    else:
                        print(f" [!] Advertencia: Se intentó eliminar {ticker} pero no existía en la base.")
                else:
                    print(f" [!] Acción desconocida ignorada: {action} para el ticker {ticker}")
        else:
            print(f"Sin cambios programados para el {target_date_str}. Se mantiene la estructura del {base_date}.")

        # =========================================================================
        # 3. CONSOLIDAR Y GUARDAR
        # =========================================================================
        sorted_list = sorted(list(tickers_actuals))
        tickers_string = ",".join(sorted_list)
        total_companies = len(sorted_list)

        print(f"Guardando nueva composición. Total compañías: {total_companies}")
        table_historic.put_item(
            Item={
                'Date': target_date_str,
                'Ticker': tickers_string,
                'Total Companies': total_companies
            }
        )

        print("--- EJECUCIÓN FINALIZADA CON ÉXITO ---")
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
