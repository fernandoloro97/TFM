# import streamlit as st
# import pandas as pd
# import numpy as np
# import boto3
# import plotly.express as px

# # Configuración de página de Streamlit
# st.set_page_config(page_title="Dashboard de Inversión", layout="wide")
# st.title("📊 Evolución de Patrimonio Neto y Métricas de Trading")

# # 1. Cargar Datos de DynamoDB con Caché para no saturar lecturas
# @st.cache_data(ttl=60)
# def cargar_datos_balance():
#     # En App Runner, boto3 tomará automáticamente el Rol de IAM asignado al servicio
#     dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
#     table = dynamodb.Table('daily_balance')
    
#     response = table.scan()
#     data = response.get('Items', [])

import streamlit as st
import pandas as pd
import numpy as np
import boto3
import plotly.express as px

@st.cache_data(ttl=60)
def cargar_datos_balance():
    # Leer las credenciales desde los secretos de la plataforma de Streamlit
    dynamodb = boto3.resource(
        'dynamodb',
        region_name='us-east-1',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
    table = dynamodb.Table('daily_balance')
    
    response = table.scan()
    data = response.get('Items', [])
    
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
        
    df = pd.DataFrame(data)
    
    if df.empty:
        # Fila base de emergencia si la tabla está totalmente vacía en tus pruebas iniciales
        df = pd.DataFrame([{
            'fecha': '2026-05-20',
            'capital_cash': 20000.0,
            'valor_posiciones': 0.0,
            'equity_total': 20000.0
        }])
    else:
        # Asegurar tipos correctos
        df['fecha'] = pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d')
        df['equity_total'] = pd.to_numeric(df['equity_total']).astype(float)
        df['capital_cash'] = pd.to_numeric(df['capital_cash']).astype(float)
        df['valor_posiciones'] = pd.to_numeric(df['valor_posiciones']).astype(float)
        
    # Ordenar por fecha cronológicamente
    df = df.sort_values('fecha').reset_index(drop=True)
    return df

# Obtener registro real de AWS
df_real = cargar_datos_balance()

# 2. Inyección de las 2 filas simuladas solicitadas (21/05 y 22/05)
# Extraemos el último equity disponible real
ultimo_equity = float(df_real['equity_total'].iloc[-1])

filas_simuladas = [
    {'fecha': '2026-05-21', 'capital_cash': ultimo_equity + 10.0, 'valor_posiciones': 0.0, 'equity_total': ultimo_equity + 10.0},
    {'fecha': '2026-05-22', 'capital_cash': ultimo_equity + 30.0, 'valor_posiciones': 0.0, 'equity_total': ultimo_equity + 30.0}
]
df_simulado = pd.DataFrame(filas_simuladas)

# Unir y consolidar histórico definitivo
df_historico = pd.concat([df_real, df_simulado], ignore_index=True).drop_duplicates(subset=['fecha'], keep='first')
df_historico = df_historico.sort_values('fecha').reset_index(drop=True)

# 3. Componente Selector de Fecha en la Barra Lateral
fechas_disponibles = df_historico['fecha'].tolist()
fecha_seleccionada = st.sidebar.selectbox("Seleccione la fecha de análisis:", fechas_disponibles, index=len(fechas_disponibles)-1)

# 4. Filtrar Datos Acumulados hasta la Fecha Seleccionada
df_filtrado = df_historico[df_historico['fecha'] <= fecha_seleccionada].copy()
fila_actual = df_historico[df_historico['fecha'] == fecha_seleccionada].iloc[0]

# 5. CÁLCULO DE MÉTRICAS FINANCIERAS AJUSTADAS (NO ANUALIZADAS)
# Calcular rendimientos simples aritméticos diarios (necesarios para volatilidades y drawdowns nativos)
df_filtrado['ret_diario'] = df_filtrado['equity_total'].pct_change().fillna(0)
# Rendimientos logarítmicos solicitados para sumatorias directas
df_filtrado['ret_diario_log'] = np.log(df_filtrado['equity_total'] / df_filtrado['equity_total'].shift(1)).fillna(0)

# Rentabilidad Total Acumulada No Anualizada
rent_total = (df_filtrado['equity_total'].iloc[-1] / df_filtrado['equity_total'].iloc[0]) - 1

# Volatilidad No Anualizada (Desviación estándar de los retornos logarítmicos del periodo)
vol_periodo = df_filtrado['ret_diario_log'].std()

# Métricas de Ratios con Tasa Libre de Riesgo = 0
sharpe_periodo = (df_filtrado['ret_diario_log'].mean() / vol_periodo) if vol_periodo != 0 else 0

# Drawdowns exactos
picos = df_filtrado['equity_total'].cummax()
drawdowns = (df_filtrado['equity_total'] - picos) / picos
max_dd = drawdowns.min()

# Sortino No Anualizado (Ajustado por volatilidad a la baja)
ret_negativos = df_filtrado.loc[df_filtrado['ret_diario_log'] < 0, 'ret_diario_log']
vol_downside = ret_negativos.std() if not ret_negativos.empty else 0
sortino_periodo = (df_filtrado['ret_diario_log'].mean() / vol_downside) if vol_downside != 0 else 0

# Calmar No Anualizado
calmar_periodo = (rent_total / abs(max_dd)) if max_dd != 0 else 0

# Duración Máxima de Drawdown
en_drawdown = df_filtrado['equity_total'] < picos
racha_drawdown = en_drawdown.groupby((~en_drawdown).cumsum()).cumsum()
max_duracion_dd = int(racha_drawdown.max()) if not racha_drawdown.empty else 0

# Win Rate y Profit/Loss Ratio Diario
dias_ganadores = df_filtrado['ret_diario'] > 0
dias_perdedores = df_filtrado['ret_diario'] < 0
total_dias_operados = len(df_filtrado) - 1 if len(df_filtrado) > 1 else 1

win_rate = dias_ganadores.sum() / total_dias_operados
avg_ganancia = df_filtrado.loc[dias_ganadores, 'ret_diario'].mean() if dias_ganadores.any() else 0
avg_perdida = df_filtrado.loc[dias_perdedores, 'ret_diario'].mean() if dias_perdedores.any() else 0
pl_ratio = abs(avg_ganancia / avg_perdida) if avg_perdida != 0 else 0

# ==============================================================================
# DISPOSICIÓN DE LA INTERFAZ GRÁFICA (LAYOUT)
# ==============================================================================
col_izq, col_der = st.columns([2, 1])

with col_izq:
    st.subheader(f"📈 Evolución del Patrimonio Neto (Hasta {fecha_seleccionada})")
    
    # Crear el gráfico con Plotly Express
    fig = px.line(df_filtrado, x='fecha', y='equity_total', markers=True,
                  labels={'fecha': 'Fecha', 'equity_total': 'Patrimonio Total ($)'},
                  template="plotly_dark")
    
    # CORRECCIÓN AQUÍ: Usamos un diccionario para pasar los estilos de línea de forma estricta
    fig.update_traces(line=dict(color='#00FFCC', width=3))
    
    # Renderizar en Streamlit
    st.plotly_chart(fig, use_container_width=True)

with col_der:
    st.subheader(f"💰 Desglose Contable al {fecha_seleccionada}")
    # Tabla 1: Componentes del patrimonio para la fecha seleccionada
    df_tabla_contable = pd.DataFrame({
        'Métrica': ['Efectivo', 'Valor de posiciones', 'Patrimonio Total'],
        'Valor ($)': [f"${fila_actual['capital_cash']:.2f}", 
                      f"${fila_actual['valor_posiciones']:.2f}", 
                      f"${fila_actual['equity_total']:.2f}"]
    })
    st.table(df_tabla_contable.set_index('Métrica'))

st.separator()
st.subheader(f"📋 Matriz de Métricas Avanzadas del Periodo (Acumulado hasta {fecha_seleccionada})")

# Tabla 2: Métricas de rendimiento y riesgo del periodo filtrado
df_tabla_metricas = pd.DataFrame({
    'Indicador Financiero': [
        'Rentabilidad Acumulada',
        'Volatilidad del Periodo',
        'Ratio Sharpe del Periodo',
        'Ratio Sortino del Periodo',
        'Ratio Calmar del Periodo',
        'Máximo Drawdown (Max DD)',
        'Duración Máxima del Drawdown',
        'Porcentaje de Días Ganados (Win Rate)',
        'Ratio de Ganancia / Pérdida Diario (P/L Ratio)'
    ],
    'Valor Obtenido': [
        f"{rent_total * 100:.2f}%",
        f"{vol_periodo * 100:.4f}%",
        f"{sharpe_periodo:.4f}",
        f"{sortino_periodo:.4f}",
        f"{calmar_periodo:.4f}",
        f"{max_dd * 100:.2f}%",
        f"{max_duracion_dd} días",
        f"{win_rate * 100:.2f}%",
        f"{pl_ratio:.2f}"
    ]
})
st.dataframe(df_tabla_metricas, use_container_width=True, hide_index=True)
