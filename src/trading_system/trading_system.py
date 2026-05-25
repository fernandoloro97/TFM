import pandas as pd
import numpy as np

from datetime import datetime
from decimal import Decimal
import json
import boto3



# Configuración de AWS (Boto3 usará las variables de entorno de GitHub Actions)
dynamodb = boto3.resource('dynamodb', region_name='us-east-1') # Cambia a tu región
s3 = boto3.client('s3')
dynamodb_client = boto3.client('dynamodb', region_name='us-east-1')

def get_table_df(table_name):
    table = dynamodb.Table(table_name)
    response = table.scan()
    data = response['Items']
    
    # Manejo de paginación si la tabla es grande
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    
    return pd.DataFrame(data)


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


def handler(event, context):

    precios_cierre_sesion = get_table_df('sesion_close_prices')
    volumenes_sesion = get_table_df('sesion_volumes')
    pendientes_ayer = get_table_df('pending_trade')
    historico_balance = get_table_df('daily_balance')
    seynales_modelo = get_table_df('model_signals')

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
        
        
    # 2. Aplicar formateo estricto de tipos de datos (Corregido sin True_label)
    if not seynales_modelo.empty:
        # Columna de Texto (String)
        seynales_modelo['Tickers Mapeados'] = seynales_modelo['Tickers Mapeados'].astype(str)
        
        # Columnas de Números Enteros (Integers)
        seynales_modelo['ID'] = pd.to_numeric(seynales_modelo['ID'], errors='coerce').astype(int)
        seynales_modelo['Fila Noticia'] = pd.to_numeric(seynales_modelo['Fila Noticia'], errors='coerce').astype(int)
        seynales_modelo['Pred_label'] = pd.to_numeric(seynales_modelo['Pred_label'], errors='coerce').astype(int)
        
        # Columna de Números Decimales (Float)
        seynales_modelo['Prob_up'] = pd.to_numeric(seynales_modelo['Prob_up'], errors='coerce').astype(float)
        
        # Columna de Fecha y Hora (Datetime)
        seynales_modelo['Date'] = pd.to_datetime(seynales_modelo['Date'])
        
        # Limpieza de índices: Descartar desorden de filas y reiniciar desde 0
        seynales_modelo.reset_index(drop=True, inplace=True)
    else:
        # Estructura vacía con tipos definidos por si no hay registros
        columnas = ['ID', 'Tickers Mapeados', 'Fila Noticia', 'Date', 'Prob_up', 'Pred_label']
        seynales_modelo = pd.DataFrame(columns=columnas)


    # ==============================================================================
    # REPARAR Y CONFIGURAR TABLA DE PENDIENTES DE AYER (PENDING_TRADE)
    # ==============================================================================
    # Si la tabla tiene registros, aplicamos tipado estricto para no romper el simulador
    if not pendientes_ayer.empty:
        print(f"¡Se encontraron {len(pendientes_ayer)} operaciones pendientes de ayer! Formateando...")
        
        # 1. Forzar conversión de la columna 'Date' a datetime real de Pandas
        if 'Date' in pendientes_ayer.columns:
            pendientes_ayer['Date'] = pd.to_datetime(pendientes_ayer['Date'])
        
        # 2. Convertir columnas de números enteros (IDs y contadores)
        columnas_enteras = ['Fila Noticia', 'Pred_label', 'True_label', 'minuto_entrada']
        for col in columnas_enteras:
            if col in pendientes_ayer.columns:
                pendientes_ayer[col] = pd.to_numeric(pendientes_ayer[col], errors='coerce').fillna(0).astype(int)
                
        # 3. Convertir columnas monetarias, precios y volúmenes (Floats con NaN permitidos)
        columnas_decimales = [
            'Prob_up', 'vol_media_10d', 'vol_1_porc', 'px_media_10d', 'px_std_10d', 
            'cap_disponible', 'cap_riesgo', 'distancia_stop', 'cant_teorica_riesgo', 
            'cant_limite_exposicion', 'cant_negociada', 'px_entrada', 'px_salida', 'pnl_unitario'
        ]
        for col in columnas_decimales:
            if col in pendientes_ayer.columns:
                pendientes_ayer[col] = pd.to_numeric(pendientes_ayer[col], errors='coerce').astype(float)
                
        # 4. Asegurar formato de texto para tickers y estado
        if 'Tickers Mapeados' in pendientes_ayer.columns:
            pendientes_ayer['Tickers Mapeados'] = pendientes_ayer['Tickers Mapeados'].astype(str)
        if 'estado' in pendientes_ayer.columns:
            pendientes_ayer['estado'] = pendientes_ayer['estado'].astype(str)

        # 5. Reiniciar índice para eliminar residuos numéricos de DynamoDB
        pendientes_ayer.reset_index(drop=True, inplace=True)
    else:
        print("La tabla 'pending_trade' está vacía en DynamoDB. El simulador ignorará registros pasados.")
        # Al dejarlo vacío, la condición 'None if pendientes_ayer.empty' del simulador funcionará perfectamente

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
        fecha_inicio = datetime(2026, 5, 21).date()# <-- Usa solo date()
    )
    
    if not df_todo.empty and 'estado' in df_todo.columns:
        # Filtrar solo pendientes si la columna existe y hay datos
        df_pendientes = df_todo[df_todo['estado'].isin(['PENDIENTE_SALIR', 'PENDIENTE_ENTRAR'])].copy()
    else:
        print("⚠️ df_todo está vacío o no contiene la columna 'estado'. Generando df_pendientes vacío.")
        # Creamos un DataFrame vacío con las columnas estructurales que exige tu proceso de DynamoDB abajo
        df_pendientes = pd.DataFrame(columns=['Tickers Mapeados', 'Fila Noticia', 'estado'])
    

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
    table_pendientes = dynamodb.Table(NOMBRE_TABLA_PENDIENTES)

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
    table_balance = dynamodb.Table(NOMBRE_TABLA_BALANCE)

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
        
    # Retorno exitoso que requiere AWS Lambda
    return {
        'statusCode': 200,
        'body': json.dumps('Simulación y persistencia completadas exitosamente.')
    }