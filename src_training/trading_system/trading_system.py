import pandas as pd
import numpy as np
import pandas as pd
import itertools
import re
import boto3
import pickle
import multiprocessing

from concurrent.futures import ThreadPoolExecutor


# Congifur el s3 y el dynamodb
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
s3 = boto3.client('s3')

# Descargo las tablas en partes y corrijo los tipos de datos para ejecutarlo en el sistema trading
def descargar_y_limpiar_tabla(nombre_tabla):

    print(f"Iniciando descarga de la tabla: {nombre_tabla}", flush=True)
    table = dynamodb.Table(nombre_tabla)
    
    response = table.scan()
    chunks = [pd.DataFrame(response['Items'])]
    total_filas = len(response['Items'])
    
    # Manejo de la paginación de 1MB de AWS
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        if response['Items']:
            chunks.append(pd.DataFrame(response['Items']))
            total_filas += len(response['Items'])
            if total_filas % 100000 == 0:
                print(f"  -> {nombre_tabla}: {total_filas} filas leídas...", flush=True)
                
    if not chunks or chunks[0].empty:
        print(f"[Advertencia] La tabla {nombre_tabla} está vacía.")
        return pd.DataFrame()

    # Concato todos los resultado en bloques
    df = pd.concat(chunks, ignore_index=True)
    print(f"[Procesando] Estructurando DataFrame para {nombre_tabla}...", flush=True)

    if 'Date' not in df.columns:
        raise KeyError(f"La columna 'Date' no se encontró en la tabla {nombre_tabla}")

    # Convierto Date a formato datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Establezco Date como indice, porque lo tenia como columna en la tabla
    df = df.set_index('Date')
    
    # Convierto tanto los precios y volumenes a float para operar con ellos, ya que esta en Decimal
    df = df.apply(pd.to_numeric, errors='coerce').astype(float)
    
    # Ordeno las fechas de las mas antiguo al reciente
    df = df.sort_index().sort_index(axis=1)
    
    print(f"[OK] {nombre_tabla} lista. Dimensiones: {df.shape}", flush=True)
    
    return df


# Calculo las metricas necesarias para entrar y salir en el mercado
def obtener_metricas(fila, df_vol, df_px, ventana=120, trailing=False, trailing_trigger_pct=0.02):

    ticker      = fila['Tickers Mapeados']
    fecha_señal = fila['Date']
    label       = fila['Pred_label']

    # Compruebo si puedo entrar en apertura
    
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

    # Busco la entrada al mercado

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
            return pd.Series([np.nan] * 20)

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
                return pd.Series([np.nan] * 20)

    # Calculo el precio de entrada

    pos_entrada = df_vol.index.get_loc(idx_entrada)

    if idx_entrada not in df_px.index:
        return pd.Series([np.nan] * 20)

    px_entrada = df_px.at[idx_entrada, ticker]

    if pd.isna(px_entrada):
        return pd.Series([np.nan] * 20)

    px_entrada = round(px_entrada, 2)

    # Me aseguro que haya datos para calcular el historico para el volumen

    dias_hist = 10
    min_cobertura = 0.70

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

    # Tengo en cuenta el horario para negociar en apertura, dado mi latencia

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

    # Calculo los historicos para precio cierre y volumen

    v_hist = df_vol.loc[mask_hist, ticker]

    v_hist = v_hist[
        v_hist.index.strftime('%H:%M').isin(horarios_ref)
    ].dropna()

    p_hist = df_px.loc[mask_hist, ticker]

    p_hist = p_hist[
        p_hist.index.strftime('%H:%M').isin(horarios_ref)
    ].dropna()

    # Dato la diferencia de liquidez, tengo en cuenta el historico real

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

    # Ahora si calculo la metricas con el historico

    # Volumen
    if cobertura_vol < min_cobertura:

        vol_media = np.nan

    else:

        vol_media = round(v_hist.mean(), 2)

    # 1% del volumen medio del historico
    if pd.notna(vol_media):

        vol_1_p = int(np.floor(vol_media * 0.01))

    else:

        vol_1_p = np.nan

    # Precio
    if cobertura_px < min_cobertura:

        px_media = np.nan
        px_std   = np.nan

    else:

        px_media = round(p_hist.mean(), 2)
        px_std   = round(p_hist.std(), 2)

    # No hago negociacion si no alcanza un limite de datos historicos

    if pd.isna(px_media) or pd.isna(px_std):

        return pd.Series([
            vol_media,
            vol_1_p,
            px_media,
            px_std,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            'NO_HISTORICO',
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            pd.NaT,
            pd.NaT
        ])

    # Calculo los umbrales de entrada y de salida del mercado 

    u_gen_sup = round(px_media + 1.5 * px_std, 2)
    u_gen_inf = round(px_media - 1.5 * px_std, 2)

    u_ent_sup = round(px_media + 1.4 * px_std, 2)
    u_ent_inf = round(px_media - 1.4 * px_std, 2)

    # Si se entra, pues se devuelve las metricas calculadas 

    if not (u_ent_inf <= px_entrada <= u_ent_sup):

        return pd.Series([
            vol_media,
            vol_1_p,
            px_media,
            px_std,
            u_gen_inf,
            u_gen_sup,
            u_ent_inf,
            u_ent_sup,
            px_entrada,
            minuto_entrada,
            tipo_ejec_ent,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            pd.NaT,
            pd.NaT
        ])

    # Busco salir del mercado mediante umbrales

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

    # Busco salir del mercado mediante el tiempo maximo que dura la ventana de analisis

    if idx_salida_final is None:

        bus_t = df_px[ticker].iloc[
            pos_entrada + ventana + 4:
            pos_entrada + ventana + 250
        ].dropna()

        if not bus_t.empty:

            idx_salida_final = bus_t.index[0]
            min_disparo      = ventana

        else:
            return pd.Series([np.nan] * 20)

    # Guardo los datos de salida y resultado final

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
        vol_media,
        vol_1_p,
        px_media,
        px_std,
        u_gen_inf,
        u_gen_sup,
        u_ent_inf,
        u_ent_sup,
        px_entrada,
        minuto_entrada,
        tipo_ejec_ent,
        px_salida,
        min_disparo,
        min_ejec_sal,
        tipo_ejec_sal,
        pnl,
        res,
        motivo,
        idx_entrada,
        idx_salida_final
    ])
    

# Clase que simula un sistema de trading diario, manteniendo el estado del capital, posiciones abiertas y movimientos de caja entre ejecuciones
class TradingSimulator2:
    # Distintos parametros para ejecutar la configuracion basica, robusta y combinacion de ambas, con distintos capitas y ventanas
    def __init__(self, capital_inicial=10000, ventana=120, trailing=False, 
                 trailing_trigger_pct=0.02, usar_limite_exposicion=False, 
                 limite_exposicion_pct=0.05, perc_riesgo=0.005):
       
        # Defino variables de parametros seleccionados
        self.capital_inicial = capital_inicial
        self.capital = capital_inicial
        self.ventana = ventana
        self.trailing = trailing
        self.trailing_trigger_pct = trailing_trigger_pct
        self.usar_limite_exposicion = usar_limite_exposicion    
        self.limite_exposicion_pct = limite_exposicion_pct     
        self.perc_riesgo = perc_riesgo   
        
        # Contenedores para el registro de datos de negociacion
        self.posiciones            = {}
        self.cola_salidas          = {}
        self.movimientos_caja      = []
        self.reporte_diario        = []

    # Proceso todas las salidas de posiciones cuyo timestamp es anterior o igual al momento dado, actualizando el capital
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

    # Registro la entrada a una posicion descontando el efectivo correspondiente y programando su salida en la cola
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

    # Ejecuto la negociacion de entrada y salida al mercado con vision de todo los precios del perido de analisis
    def ejecutar_simulacion(self, df_señales, df_vol, df_px):
        df = df_señales.copy()

        nombres = [
            'vol_media_10d', 'vol_1_porc', 'px_media_10d', 'px_std_10d',
            'u_gen_inf', 'u_gen_sup', 'u_ent_inf', 'u_ent_sup',
            'px_entrada', 'minuto_entrada', 'tipo_ejec_entrada',
            'px_salida', 'min_disparo_umbral', 'min_ejecucion_salida', 'tipo_ejec_salida',
            'pnl_unitario', 'res_tag', 'motivo_salida',
            'ts_entrada_real', 'ts_salida_real'
        ]
        df[nombres] = df.apply(
            lambda x: obtener_metricas(
                x, df_vol, df_px,
                ventana              = self.ventana,
                trailing             = self.trailing,
                trailing_trigger_pct = self.trailing_trigger_pct
            ), axis=1
        )

        df = df.sort_values(by=['ts_entrada_real', 'Fila Noticia'])

        audit_data = {
            idx: {
                'cap_disponible':        np.nan,
                'cap_riesgo':            np.nan,
                'distancia_stop':        np.nan,
                'cant_teorica_riesgo':   np.nan,
                'cant_limite_exposicion': np.nan,  
                'cant_negociada':        np.nan,
                'caja_ent':              np.nan,
                'caja_sal':              np.nan,
                'res_neto':              np.nan,
                'tipo_ejec_final':       np.nan,
                'estado':                'NO NEGOCIADO'
            }
            for idx in df.index
        }
        df_validas = df[df['ts_entrada_real'].notna()].copy()

        for f_entrada, grupo in df_validas.groupby('ts_entrada_real'):

            self._liquidar_salidas_hasta(f_entrada, df_px)

            capital_disponible = self.capital
            ordenes_propuestas = []

            grupo_en_umbral = grupo[
                grupo['px_entrada'].notna() &
                grupo['px_salida'].notna() &
                (grupo['tipo_ejec_entrada'] != 'NO_HISTORICO')
            ]

            for noti, sub_grupo in grupo_en_umbral.groupby('Fila Noticia'):
                for pred_val in [0, 1]:
                    filas = sub_grupo[sub_grupo['Pred_label'] == pred_val]
                    n     = len(filas)
                    if n == 0:
                        continue

                    prob        = filas['Prob_up'].iloc[0]
                    perc_riesgo = self.perc_riesgo if 0.3 <= prob <= 0.7 else 0.01
                    cap_riesgo  = np.floor((capital_disponible * perc_riesgo) / n)

                    limite_cash_tk = (capital_disponible * self.limite_exposicion_pct) / n

                    for _, f in filas.iterrows():
                        px_ent  = f['px_entrada']
                        dist_sl = round(
                            px_ent - f['u_gen_inf'] if pred_val == 1
                            else f['u_gen_sup'] - px_ent, 2
                        )
                        dist_sl     = max(dist_sl, 0.01)
                        cant_riesgo = int(np.floor(cap_riesgo / dist_sl))

                        cant_exposicion = int(np.floor(limite_cash_tk / px_ent))

                        if pd.isna(f['vol_1_porc']) or f['vol_1_porc'] <= 0:

                            cantidad = 0
                        
                            audit_data[f.name]['cant_negociada']  = 0
                            audit_data[f.name]['tipo_ejec_final'] = 'NO_NEGOCIADO_SIN_HISTORICO'
                        
                        else:
                        
                            vol_disponible = int(f['vol_1_porc'])
                        
                            if self.usar_limite_exposicion:
                        
                                cantidad = min(
                                    cant_riesgo,
                                    vol_disponible,
                                    cant_exposicion
                                )
                        
                            else:
                        
                                cantidad = min(
                                    cant_riesgo,
                                    vol_disponible
                                )

                        audit_data[f.name]['cap_disponible']         = round(capital_disponible, 2)
                        audit_data[f.name]['cap_riesgo']             = round(cap_riesgo, 2)
                        audit_data[f.name]['distancia_stop']         = dist_sl
                        audit_data[f.name]['cant_teorica_riesgo']    = cant_riesgo
                        audit_data[f.name]['cant_limite_exposicion'] = cant_exposicion  

                        if cantidad > 0:
                            audit_data[f.name]['cant_negociada']  = cantidad
                            audit_data[f.name]['tipo_ejec_final'] = f['tipo_ejec_entrada']
                            ordenes_propuestas.append({
                                'ticker':            f['Tickers Mapeados'],
                                'pred':              pred_val,
                                'prob':              prob,
                                'cant':              cantidad,
                                'px_ent':            px_ent,
                                'original_row_idx':  f.name,
                                'px_salida':         f['px_salida'],
                                'ts_salida_real':    f['ts_salida_real'],
                                'dist_sl':           dist_sl,
                                'cant_riesgo':       cant_riesgo,
                                'cant_exposicion':   cant_exposicion,  
                                'cap_riesgo':        cap_riesgo,
                            })
                            
                        else:
                            if pd.isna(audit_data[f.name]['tipo_ejec_final']):
                        
                                audit_data[f.name]['cant_negociada']  = 0
                                audit_data[f.name]['tipo_ejec_final'] = 'SIN_LIQUIDEZ'

            grupo_fuera = grupo[
                grupo['px_entrada'].notna() &
                grupo['px_salida'].isna()
            ]
            for idx in grupo_fuera.index:
                if pd.isna(audit_data[idx]['cant_negociada']):
                    audit_data[idx]['tipo_ejec_final'] = 'FUERA_UMBRAL'

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

                self._abrir_posicion(
                    ticker     = ticker,
                    volumen    = vol_final,
                    px_ent     = round(df_px.at[f_entrada, ticker], 2),
                    pred       = pred_final,
                    ts_entrada = f_entrada,
                    ts_salida  = f_valida['ts_salida_real'],
                )

                comision = round(0.005 * vol_final, 2)
                res_neto = (
                    (f_valida['px_salida'] - f_valida['px_ent']) * vol_final
                    if pred_final == 1
                    else (f_valida['px_ent'] - f_valida['px_salida']) * vol_final
                ) - comision * 2

                audit_data[idx_v]['cant_negociada'] = vol_final
                audit_data[idx_v]['caja_ent']       = round(f_valida['px_ent'] * vol_final, 2)
                audit_data[idx_v]['caja_sal']       = round(f_valida['px_salida'] * vol_final, 2)
                audit_data[idx_v]['res_neto']       = round(res_neto, 2)
                audit_data[idx_v]['estado']         = 'NEGOCIADO'

        # Termino de liquidar las salidas restantes
        for ts in sorted([
            ts for c in self.cola_salidas.values() for ts in c.keys()
        ]):
            self._liquidar_salidas_hasta(ts, df_px)

        # Calculo el balancerio diario de caja, posiciones y patrimonio total al final de sesion
        movimientos = sorted(self.movimientos_caja, key=lambda x: x[0])

        mask_neg  = pd.Series({idx: audit_data[idx]['estado'] == 'NEGOCIADO' for idx in df.index})
        filas_neg = df[mask_neg].copy()
        filas_neg['_cant'] = filas_neg.index.map(lambda idx: audit_data[idx]['cant_negociada'])
        filas_neg = filas_neg[filas_neg['_cant'] > 0].dropna(subset=['ts_entrada_real', 'ts_salida_real'])

        for dia in sorted(set(df_px.index.date)):
            cierre_dia  = pd.Timestamp(f"{dia} 16:00:00")
            capital_dia = self.capital_inicial + sum(c for ts, c, _ in movimientos if ts <= cierre_dia)
            dia_data    = df_px.loc[df_px.index.date == dia]
            cierre_16   = dia_data[dia_data.index.time == pd.Timestamp("16:00:00").time()]
            if cierre_16.empty and not dia_data.empty:
                cierre_16 = dia_data.iloc[[-1]]

            valor_pos = 0.0
            for _, op in filas_neg.iterrows():
                ts_ent = op['ts_entrada_real']
                ts_sal = op['ts_salida_real']
                if pd.notna(ts_ent) and pd.notna(ts_sal):
                    if ts_ent.date() <= dia and ts_sal > cierre_dia:
                        ticker = op['Tickers Mapeados']
                        if ticker not in cierre_16.columns:
                            continue
                        serie = cierre_16[ticker].dropna()
                        if serie.empty:
                            continue
                        px_c = serie.iloc[-1]
                        valor_pos += (
                            op['_cant'] * px_c if op['Pred_label'] == 1
                            else -(op['_cant'] * px_c)
                        )

            self.reporte_diario.append({
                'fecha':            dia,
                'capital_cash':     round(capital_dia, 2),
                'valor_posiciones': round(valor_pos, 2),
                'equity_total':     round(capital_dia + valor_pos, 2)
            })

        self.reporte_diario = sorted(self.reporte_diario, key=lambda x: x['fecha'])

        # Construyo la tabla de resultados por cada señal: limites de entrada, si se negocio, cuanto?, cuando? de que forma?, resultados, etc
        audit_df = pd.DataFrame.from_dict(audit_data, orient='index')
        df['tipo_ejec_entrada'] = audit_df['tipo_ejec_final'].reindex(df.index)

        cols_limpiar = [
            'px_salida', 'min_disparo_umbral', 'min_ejecucion_salida',
            'tipo_ejec_salida', 'pnl_unitario', 'res_tag',
            'ts_entrada_real', 'ts_salida_real'
        ]
        for idx in df.index:
            if audit_data[idx]['estado'] == 'NO NEGOCIADO':
                for col in cols_limpiar:
                    if col in df.columns:
                        df.at[idx, col] = np.nan

        df = df.join(audit_df[[
            'cap_disponible', 'cap_riesgo', 'distancia_stop',
            'cant_teorica_riesgo', 'cant_limite_exposicion',  
            'cant_negociada', 'caja_ent', 'caja_sal', 'res_neto', 'estado'
        ]])

        cols_base   = [c for c in df_señales.columns]
        cols_nuevas = [
            'vol_media_10d', 'vol_1_porc', 'px_media_10d', 'px_std_10d',
            'cap_disponible', 'cap_riesgo', 'distancia_stop',
            'cant_teorica_riesgo', 'cant_limite_exposicion',  
            'cant_negociada',
            'u_gen_inf', 'u_gen_sup', 'u_ent_inf', 'u_ent_sup',
            'px_entrada', 'minuto_entrada', 'tipo_ejec_entrada',
            'px_salida', 'min_disparo_umbral', 'min_ejecucion_salida', 'tipo_ejec_salida',
            'motivo_salida', 'pnl_unitario', 'res_tag',
            'ts_entrada_real', 'ts_salida_real',
            'caja_ent', 'caja_sal', 'res_neto', 'estado'
        ]
        cols_finales = cols_base + [c for c in cols_nuevas if c not in cols_base]
        df = df[[c for c in cols_finales if c in df.columns]]

        return df

    
# Ejecucion del sistema trading para el top20 de mejores modelos de prediccion por las 8 posibles configuracion de dicho sistema
def extraer_ventana_minutos(ventana_str):

    ultimo = ventana_str.split("_")[-1].lower()
    match = re.match(r"(\d+)([hm])", ultimo)

    if not match:
        raise ValueError(f"No se pudo interpretar ventana: {ventana_str}")

    valor  = int(match.group(1))
    unidad = match.group(2)
    
    if unidad == "h":
        return valor * 60

    elif unidad == "m":
        return valor

    else:
        raise ValueError(f"Unidad desconocida: {unidad}")
    
    

resultados_sistema_trading = None

# Funcion para entrenar distintas configuraciones por modelo y hacer sus inferencias
def sistemas_trading():
    global resultados_sistema_trading

    print("Inicio de la descargar de precios, volumnes e inferencias")
    # --- EJECUCIÓN MULTIHILO PARALELO ---
    tablas_mercado = ['period_sesion_close_prices', 'period_sesion_volumes']
    resultados_tablas = {}

    print("Lanzando procesos de descarga en paralelo...", flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(descargar_y_limpiar_tabla, name): name for name in tablas_mercado}
        for future in futures:
            nombre = futures[future]
            resultados_tablas[nombre] = future.result()

    # Asignación final a tus variables individuales
    precios_cierre_sesion = resultados_tablas['period_sesion_close_prices']
    volumen_sesion = resultados_tablas['period_sesion_volumes']


    print(f"\nEl df de precios de cierre tiene las siguientes dimensiones: {precios_cierre_sesion.shape}")
    print(f"\nEl df de volumnes tiene las siguientes dimensiones: {volumen_sesion.shape}")

    # Fijo los nombres del bucket y pickle donde estan las inferencias
    NOMBRE_BUCKET = 'top50_models_inferences'
    NOMBRE_ARCHIVO = 'top50_inferencias.pkl'

    response = s3.get_object(Bucket=NOMBRE_BUCKET, Key=NOMBRE_ARCHIVO)
    archivo_bytes = response['Body'].read()

    # Reconstruye tu diccionario manteniendo el 100% de los tipos de datos originales
    inferencia_modelos = pickle.loads(archivo_bytes)
    print(f"Cargue el pickle exitosamentes, donde tengo cargadas en el diccionario: {len(inferencia_modelos)} elementos", flush=True)

    print("Inicio del proceso del sistema trading")
    # Fijo el capital el 10000 para todos los modelos
    capitales = [10000]

    # Combinaciones de configuracion de sistemas trading: 8 posibles, porque hay 2 opciones por parametro y tengo solo 3 parametros
    combinaciones = list(itertools.product(
        [False, True],   # trailing
        [False, True],   # usar_limite_exposicion
        [0, 0.005]       # perc_riesgo
    ))

    resultados_backtest = {}

    # Inicio la ejecucion del sistema trading para los 20 modelos
    for idx_modelo in range(0, 20):

        info_modelo = inferencia_modelos[idx_modelo]

        model_id = info_modelo["model_id"]
        modelo = info_modelo["modelo"]
        ventana_str = info_modelo["ventana"]
        df_senales = info_modelo["inference_df"]
        ventana_min = extraer_ventana_minutos(ventana_str)

        print(f"\nModelo idx: {idx_modelo}")
        print(f"Model ID: {model_id}")
        print(f"Ventana: {ventana_min} min")


        # Recipiente diccionario para guardar los resultados, principalmente: tabla de resultados y balance diario
        resultados_backtest[idx_modelo] = {}

        # contador de tests
        idx_test = 0

        # Iteracion por capitales, pero solo hay uno para este caso
        for capital in capitales:

            # Itero por las combinacion de sistemas trading
            for trailing, usar_limite_exposicion, perc_riesgo in combinaciones:

                print(
                    f"Test={idx_test} | "
                    f"Capital={capital} | "
                    f"trailing={trailing} | "
                    f"limite_exp={usar_limite_exposicion} | "
                    f"perc_riesgo={perc_riesgo}"
                )

                # Ejecuto el sistema trading
                sim = TradingSimulator2(
                    capital_inicial=capital,
                    ventana=ventana_min,
                    trailing=trailing,
                    trailing_trigger_pct=0.02,
                    usar_limite_exposicion=usar_limite_exposicion,
                    limite_exposicion_pct=0.05,
                    perc_riesgo=perc_riesgo
                )

                # Ejecuto el calculo de la tabla de resultados
                df_resultado = sim.ejecutar_simulacion(
                    df_senales,
                    volumen_sesion,
                    precios_cierre_sesion
                )

                # Ejecuto el calculo del balance contable diario
                df_balance = pd.DataFrame(sim.reporte_diario)

                # Guardo los resultado en el diccionario mencionado
                resultados_backtest[idx_modelo][idx_test] = {

                    # Modelo y ventanas
                    "model_id": model_id,
                    "modelo": modelo,
                    "ventana_original": ventana_str,
                    "ventana_minutos": ventana_min,

                    # Test
                    "test_id": idx_test,

                    # Parametros sidel sistema trading
                    "capital_inicial": capital,
                    "trailing": trailing,
                    "usar_limite_exposicion": usar_limite_exposicion,
                    "perc_riesgo": perc_riesgo,

                    # Tabla de resultados y balancia diario
                    "df_resultado": df_resultado,
                    "df_balance": df_balance
                }

                idx_test += 1
    
    resultados_sistema_trading = resultados_backtest


# Controlo la duracion de ejecucion
proceso1 = multiprocessing.Process(target=sistemas_trading)
proceso1.start()
# Limite de maximo 2 minutos
proceso1.join(timeout=120)

# Si pasa de 2 minutos, dejo de ejecutar
if proceso1.is_alive():
    # Cierro por completo la ejecucion
    proceso1.terminate()
    proceso1.join() 
    print("\nSolo la carga de precios y volumenes minuto a minuto tarda mas de 30 minutos y el sistema trading para 20 modelos me tarda alrededor de 20 horas")
    print("\nLos resultados de trading de los 20 modelos lo tengo guardado en mi buscket top20-trading-results ")
else:
    print("El proceso de sistema trading milagrosamente termino a tiempo")
    