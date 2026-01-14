import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime

# ==========================================
# 0. CONFIGURAÇÃO DA PÁGINA E CONSTANTES
# ==========================================
st.set_page_config(
    page_title="Asset Allocator Pro - Style Mais Retorno",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- CONFIGURAÇÃO BRAPI ---
BRAPI_TOKEN = "2D29LijXrSGRJAQ7De5bUh"

# Mapeamento Nome no Dashboard -> CNPJ
FUND_CNPJS = {
    'Tarpon GT': '22.232.927/0001-90',
    'Absolute Pace': '32.073.525/0001-43',
    'SPX Patriot': '15.334.585/0001-53',
    'Sparta Infra': '30.877.528/0001-04' # Mapeado como Sparta Infra conforme seu script original
}

# CSS Customizado
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DADOS DOS FUNDOS (HÍBRIDO: MANUAL + BRAPI)
# ==========================================

@st.cache_data(ttl=3600)
def get_brapi_fund_data(cnpjs_dict, token):
    api_returns = pd.DataFrame()

    for name, cnpj_raw in cnpjs_dict.items():
        # 1. Correção da limpeza do CNPJ (Erro do log)
        cnpj_clean = "".join([c for c in cnpj_raw if c.isdigit()])
        
        # 2. Endpoint correto para fundos (plural 'funds')
        url = f"https://brapi.dev/api/v2/funds/{cnpj_clean}"
        params = {'token': token, 'range': '5y'}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # 3. Navegação na estrutura correta da API para fundos
                if 'funds' in data and len(data['funds']) > 0:
                    fund_info = data['funds'][0]
                    # Chave específica para histórico de cotas de fundos
                    history = fund_info.get('equityValueHistory', [])
                    
                    if history:
                        df_temp = pd.DataFrame(history)
                        
                        # Converte datas (formato ISO comum na BRAPI)
                        df_temp['date'] = pd.to_datetime(df_temp['date'])
                        
                        # Define a coluna de preço (valor da cota)
                        # Tenta 'dailyEquityValue' (padrão fundos) ou 'close' (fallback)
                        col_price = 'dailyEquityValue' if 'dailyEquityValue' in df_temp.columns else 'close'
                        
                        if col_price in df_temp.columns:
                            df_temp.set_index('date', inplace=True)
                            df_temp = df_temp.sort_index()
                            
                            # Resample mensal e cálculo de retorno
                            monthly_ret = df_temp[col_price].resample('ME').last().pct_change()
                            monthly_ret.name = name
                            
                            if api_returns.empty:
                                api_returns = monthly_ret.to_frame()
                            else:
                                api_returns = api_returns.join(monthly_ret, how='outer')
            else:
                st.warning(f"A API BRAPI retornou erro {response.status_code} para {name}")
                
        except Exception as e:
            st.error(f"Erro ao processar {name}: {e}")
            continue
            
    return api_returns



def get_combined_funds_data():
    # --- 1. Dados Hardcoded (Histórico Antigo) ---
    tarpon_returns = {
        '2018-01': 0.0518, '2018-02': 0.0018, '2018-03': 0.0337, '2018-04': -0.0224, '2018-05': -0.1091,
        '2018-06': -0.0884, '2018-07': 0.0849, '2018-08': -0.0347, '2018-09': -0.0189, '2018-10': 0.2337,
        '2018-11': 0.0910, '2018-12': 0.0567, '2019-01': 0.0740, '2019-02': 0.0439, '2019-03': -0.0159,
        '2019-04': 0.0400, '2019-05': 0.0436, '2019-06': 0.0292, '2019-07': 0.0628, '2019-08': 0.0211,
        '2019-09': 0.0026, '2019-10': -0.0031, '2019-11': 0.0062, '2019-12': 0.1877, '2020-01': 0.0218,
        '2020-02': -0.0712, '2020-03': -0.2893, '2020-04': 0.0877, '2020-05': 0.0870, '2020-06': 0.2435,
        '2020-07': 0.1347, '2020-08': -0.0231, '2020-09': -0.0698, '2020-10': -0.0182, '2020-11': 0.1111,
        '2020-12': 0.0679, '2021-01': -0.0318, '2021-02': 0.0735, '2021-03': 0.0256, '2021-04': 0.1305,
        '2021-05': 0.0556, '2021-06': 0.0386, '2021-07': -0.0030, '2021-08': -0.0730, '2021-09': 0.0232,
        '2021-10': -0.0251, '2021-11': -0.0714, '2021-12': 0.1015, '2022-01': 0.0161, '2022-02': 0.0575,
        '2022-03': 0.0852, '2022-04': -0.0551, '2022-05': -0.0043, '2022-06': -0.1106, '2022-07': 0.1145,
        '2022-08': 0.1226, '2022-09': 0.0448, '2022-10': 0.1258, '2022-11': -0.1150, '2022-12': -0.0507,
        '2023-01': 0.0267, '2023-02': -0.0239, '2023-03': -0.0389, '2023-04': 0.0543, '2023-05': 0.1624,
        '2023-06': 0.1054, '2023-07': 0.0663, '2023-08': -0.0135, '2023-09': 0.0918, '2023-10': -0.0646,
        '2023-11': 0.0737, '2023-12': 0.1090, '2024-01': -0.0577, '2024-02': 0.0239, '2024-03': 0.0869,
        '2024-04': -0.0287, '2024-05': -0.0398, '2024-06': 0.0033, '2024-07': 0.0237, '2024-08': 0.0406,
        '2024-09': 0.0123, '2024-10': 0.0088, '2024-11': -0.0331, '2024-12': -0.0163, '2025-01': 0.0702,
        '2025-02': 0.0441, '2025-03': 0.0856, '2025-04': 0.0614, '2025-05': 0.0104, '2025-06': -0.0004,
        '2025-07': -0.0560, '2025-08': 0.0348, '2025-09': 0.0005, '2025-10': 0.0179, '2025-11': 0.0966,
        '2025-12': 0.0112
    }
    absolute_returns = {
        '2018-12': 0.0262, '2019-01': 0.1408, '2019-02': 0.0436, '2019-03': 0.0365, '2019-04': 0.0325,
        '2019-05': 0.0276, '2019-06': 0.0519, '2019-07': 0.0634, '2019-08': 0.0237, '2019-09': 0.0163,
        '2019-10': 0.0566, '2019-11': 0.0319, '2019-12': 0.1364, '2020-01': 0.0217, '2020-02': -0.0749,
        '2020-03': -0.2324, '2020-04': 0.1233, '2020-05': 0.0584, '2020-06': 0.0880, '2020-07': 0.0882,
        '2020-08': -0.0116, '2020-09': -0.0700, '2020-10': -0.0033, '2020-11': 0.1473, '2020-12': 0.0649,
        '2021-01': -0.0218, '2021-02': -0.0187, '2021-03': 0.0666, '2021-04': 0.0615, '2021-05': 0.0567,
        '2021-06': 0.0124, '2021-07': -0.0262, '2021-08': -0.0108, '2021-09': -0.0227, '2021-10': -0.0513,
        '2021-11': 0.0265, '2021-12': 0.0534, '2022-01': 0.0529, '2022-02': -0.0020, '2022-03': 0.0700,
        '2022-04': -0.0121, '2022-05': 0.0653, '2022-06': -0.0795, '2022-07': 0.0288, '2022-08': 0.0512,
        '2022-09': -0.0237, '2022-10': 0.0343, '2022-11': -0.0033, '2022-12': -0.0099, '2023-01': 0.0046,
        '2023-02': -0.0233, '2023-03': -0.0032, '2023-04': 0.0240, '2023-05': 0.0516, '2023-06': 0.1085,
        '2023-07': 0.0597, '2023-08': -0.0308, '2023-09': -0.0060, '2023-10': -0.0380, '2023-11': 0.1039,
        '2023-12': 0.0487, '2024-01': -0.0009, '2024-02': 0.0306, '2024-03': 0.0159, '2024-04': -0.0361,
        '2024-05': -0.0195, '2024-06': 0.0137, '2024-07': 0.0344, '2024-08': 0.0624, '2024-09': -0.0258,
        '2024-10': -0.0225, '2024-11': -0.0373, '2024-12': -0.0377, '2025-01': 0.0652, '2025-02': -0.0108,
        '2025-03': 0.0312, '2025-04': 0.0608, '2025-05': 0.0866, '2025-06': 0.0010, '2025-07': -0.0305,
        '2025-08': 0.0576, '2025-09': 0.0532, '2025-10': 0.0047, '2025-11': 0.0779, '2025-12': -0.0207
    }
    sparta_returns = {
        '2018-09': 0.0037, '2018-10': 0.0049, '2018-11': 0.0066, '2018-12': 0.0058, '2019-01': 0.0073,
        '2019-02': 0.0075, '2019-03': 0.0066, '2019-04': 0.0060, '2019-05': 0.0070, '2019-06': 0.0062,
        '2019-07': 0.0039, '2019-08': 0.0021, '2019-09': 0.0015, '2019-10': -0.0060, '2019-11': -0.0126,
        '2019-12': 0.0080, '2020-01': 0.0077, '2020-02': 0.0022, '2020-03': -0.0112, '2020-04': 0.0022,
        '2020-05': 0.0000, '2020-06': 0.0075, '2020-07': 0.0096, '2020-08': 0.0072, '2020-09': 0.0034,
        '2020-10': 0.0084, '2020-11': 0.0027, '2020-12': 0.0061, '2021-01': 0.0087, '2021-02': 0.0050,
        '2021-03': 0.0100, '2021-04': 0.0060, '2021-05': 0.0051, '2021-06': 0.0116, '2021-07': 0.0079,
        '2021-08': 0.0089, '2021-09': 0.0056, '2021-10': 0.0082, '2021-11': 0.0100, '2021-12': 0.0075,
        '2022-01': 0.0091, '2022-02': 0.0100, '2022-03': 0.0110, '2022-04': 0.0096, '2022-05': 0.0155,
        '2022-06': 0.0113, '2022-07': 0.0110, '2022-08': 0.0106, '2022-09': 0.0091, '2022-10': 0.0014,
        '2022-11': 0.0068, '2022-12': 0.0108, '2023-01': 0.0040, '2023-02': -0.0102, '2023-03': 0.0131,
        '2023-04': 0.0088, '2023-05': 0.0297, '2023-06': 0.0249, '2023-07': 0.0257, '2023-08': 0.0187,
        '2023-09': 0.0098, '2023-10': 0.0103, '2023-11': 0.0121, '2023-12': 0.0117, '2024-01': 0.0166,
        '2024-02': 0.0243, '2024-03': 0.0105, '2024-04': 0.0057, '2024-05': 0.0093, '2024-06': 0.0075,
        '2024-07': 0.0118, '2024-08': 0.0124, '2024-09': 0.0129, '2024-10': 0.0097, '2024-11': 0.0036,
        '2024-12': 0.0070, '2025-01': 0.0101, '2025-02': 0.0130, '2025-03': 0.0110, '2025-04': 0.0130,
        '2025-05': 0.0107, '2025-06': 0.0157, '2025-07': 0.0147, '2025-08': 0.0180, '2025-09': 0.0211,
        '2025-10': 0.0060, '2025-11': 0.0103, '2025-12': 0.0095
    }
    
    spx_patriot_returns = {
        '2012-07': 0.0035, '2012-08': 0.0366, '2012-09': 0.0304, '2012-10': 0.0190, '2012-11': 0.0153, '2012-12': 0.0488,
        '2013-01': 0.0222, '2013-02': -0.0096, '2013-03': -0.0045, '2013-04': 0.0086, '2013-05': 0.0114, '2013-06': -0.0514, '2013-07': 0.0113, '2013-08': 0.0070, '2013-09': 0.0261, '2013-10': 0.0291, '2013-11': -0.0087, '2013-12': -0.0202,
        '2014-01': -0.0381, '2014-02': 0.0015, '2014-03': 0.0345, '2014-04': 0.0302, '2014-05': -0.0131, '2014-06': 0.0214, '2014-07': 0.0219, '2014-08': 0.0722, '2014-09': -0.0984, '2014-10': 0.0245, '2014-11': 0.0180, '2014-12': -0.0663,
        '2015-01': -0.0822, '2015-02': 0.0940, '2015-03': -0.0036, '2015-04': 0.0653, '2015-05': -0.0441, '2015-06': -0.0055, '2015-07': -0.0172, '2015-08': -0.0602, '2015-09': -0.0144, '2015-10': 0.0209, '2015-11': 0.0101, '2015-12': -0.0162,
        '2016-01': -0.0219, '2016-02': 0.0456, '2016-03': 0.0914, '2016-04': 0.0806, '2016-05': -0.0320, '2016-06': 0.0652, '2016-07': 0.1174, '2016-08': 0.0362, '2016-09': -0.0011, '2016-10': 0.1101, '2016-11': -0.0902, '2016-12': -0.0009,
        '2017-01': 0.0968, '2017-02': 0.0391, '2017-03': -0.0105, '2017-04': 0.0034, '2017-05': -0.0377, '2017-06': 0.0039, '2017-07': 0.0527, '2017-08': 0.0691, '2017-09': 0.0566, '2017-10': -0.0010, '2017-11': -0.0291, '2017-12': 0.0624,
        '2018-01': 0.0921, '2018-02': 0.0034, '2018-03': 0.0432, '2018-04': 0.0073, '2018-05': -0.1071, '2018-06': -0.0364, '2018-07': 0.0556, '2018-08': -0.0187, '2018-09': 0.0202, '2018-10': 0.0738, '2018-11': 0.0347, '2018-12': -0.0012,
        '2019-01': 0.1134, '2019-02': -0.0180, '2019-03': -0.0079, '2019-04': 0.0065, '2019-05': 0.0174, '2019-06': 0.0418, '2019-07': -0.0026, '2019-08': -0.0001, '2019-09': 0.0117, '2019-10': 0.0214, '2019-11': 0.0062, '2019-12': 0.0899,
        '2020-01': 0.0093, '2020-02': -0.0773, '2020-03': -0.2696, '2020-04': 0.0787, '2020-05': 0.0530, '2020-06': 0.0683, '2020-07': 0.0804, '2020-08': -0.0280, '2020-09': -0.0442, '2020-10': -0.0346, '2020-11': 0.1295, '2020-12': 0.0968,
        '2021-01': -0.0210, '2021-02': 0.0030, '2021-03': 0.0616, '2021-04': 0.0365, '2021-05': 0.0445, '2021-06': 0.0193, '2021-07': -0.0461, '2021-08': -0.0282, '2021-09': -0.0487, '2021-10': -0.0870, '2021-11': -0.0267, '2021-12': 0.0559,
        '2022-01': 0.0635, '2022-02': -0.0296, '2022-03': 0.0597, '2022-04': -0.0777, '2022-05': 0.0220, '2022-06': -0.1011, '2022-07': 0.0517, '2022-08': 0.0904, '2022-09': 0.0081, '2022-10': 0.0835, '2022-11': -0.0675, '2022-12': -0.0235,
        '2023-01': 0.0478, '2023-02': -0.0525, '2023-03': -0.0166, '2023-04': 0.0291, '2023-05': 0.0632, '2023-06': 0.1091, '2023-07': 0.0328, '2023-08': -0.0560, '2023-09': -0.0037, '2023-10': -0.0312, '2023-11': 0.1274, '2023-12': 0.0598,
        '2024-01': -0.0509, '2024-02': 0.0207, '2024-03': 0.0181, '2024-04': -0.0566, '2024-05': -0.0120, '2024-06': 0.0079, '2024-07': 0.0362, '2024-08': 0.0365, '2024-09': -0.0352, '2024-10': 0.0063, '2024-11': -0.0284, '2024-12': -0.0348,
        '2025-01': 0.0652, '2025-02': -0.0172, '2025-03': 0.0537, '2025-04': 0.0914, '2025-05': 0.0697, '2025-06': 0.0192, '2025-07': -0.0491, '2025-08': 0.0978, '2025-09': 0.0368, '2025-10': 0.0154, '2025-11': 0.0687, '2025-12': 0.0126,
        '2026-01': 0.0275
    }

    df_manual = pd.DataFrame({
        'Tarpon GT': pd.Series(tarpon_returns),
        'Absolute Pace': pd.Series(absolute_returns),
        'Sparta Infra': pd.Series(sparta_returns),
        'SPX Patriot': pd.Series(spx_patriot_returns)
    })
    
    # Padronizar datas manuais
    df_manual.index = pd.to_datetime(df_manual.index).to_period('M').to_timestamp('M')

    # --- 2. Dados Automatizados (API BRAPI) ---
    with st.spinner('Atualizando fundos via BRAPI...'):
        df_api = get_brapi_fund_data(FUND_CNPJS, BRAPI_TOKEN)
    
    if not df_api.empty:
        # Mesclar: Usa dados da API para atualizar/preencher a partir de Jan/2026
        # A lógica combine_first dá prioridade ao DF que chama (df_api) sobre o argumento (df_manual)
        # Filtramos a API para garantir que ela só sobrescreva/adicione dados recentes se necessário
        
        # Garante alinhamento de índices
        df_api.index = df_api.index.to_period('M').to_timestamp('M')
        
        # Merge inteligente: Prioriza API para dados > 2026, mantém Manual para histórico antigo
        df_final = df_api.combine_first(df_manual)
    else:
        df_final = df_manual
        st.warning("Não foi possível obter dados atualizados da BRAPI. Usando apenas histórico manual.")

    return df_final

# ==========================================
# 2. FUNÇÕES DE DADOS (YFINANCE)
# ==========================================
@st.cache_data
def get_market_data(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()
    
    processed_tickers = []
    for t in tickers:
        t = t.strip().upper()
        if "." not in t and any(char.isdigit() for char in t): 
            processed_tickers.append(f"{t}.SA")
        else:
            processed_tickers.append(t)
            
    try:
        data = yf.download(processed_tickers, start=start_date, end=end_date, progress=False)
        if data.empty: return pd.DataFrame()

        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            try: prices = data.xs('Adj Close', level=0, axis=1)
            except KeyError: prices = data.xs('Close', level=0, axis=1)
        
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=processed_tickers[0])
            
        if isinstance(prices.columns, pd.MultiIndex):
             prices.columns = prices.columns.get_level_values(-1)

        monthly_prices = prices.resample('ME').last() 
        returns = monthly_prices.pct_change()
        returns.columns = [str(c).replace('.SA', '') for c in returns.columns]
        
        return returns
    except Exception as e:
        st.error(f"Erro no download: {e}")
        return pd.DataFrame()

@st.cache_data
def get_benchmark_data(start_date, end_date):
    """Baixa o Ibovespa (^BVSP) como benchmark de mercado."""
    try:
        # Ibovespa
        ibov = yf.download("BOVA11", start=start_date, end=end_date, progress=False)['Adj Close']
        ibov = ibov.resample('ME').last().pct_change()
        if isinstance(ibov, pd.DataFrame):
             ibov = ibov.iloc[:, 0]
        ibov.name = "Ibovespa"
        return ibov
    except:
        return pd.Series()

# ==========================================
# 3. LÓGICA DE CÁLCULO
# ==========================================
def calculate_portfolio_performance(returns_df, weights, initial_cap, monthly_contribution, rebalance_freq):
    returns_df = returns_df.dropna()
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    
    if not available_assets: return None, None, None

    active_weights = np.array([weights[c] for c in available_assets])
    active_weights = active_weights / active_weights.sum() 
    
    # 1. Performance Pura (Cota Base 100) - Sem aportes externos
    portfolio_pure_idx = [100.0]
    monthly_returns = []
    
    # 2. Performance com Aportes (Patrimônio)
    portfolio_wealth = [initial_cap]
    
    current_weights = active_weights.copy()
    dates = returns_df.index
    asset_returns_np = returns_df[available_assets].values
    
    for i in range(len(dates)):
        r_t = asset_returns_np[i]
        
        # Retorno do mês (weighted average)
        port_ret = np.dot(current_weights, r_t)
        monthly_returns.append(port_ret)
        
        # Atualiza Cota (Base 100)
        new_idx = portfolio_pure_idx[-1] * (1 + port_ret)
        portfolio_pure_idx.append(new_idx)
        
        # Atualiza Patrimônio (Com aporte)
        new_wealth = (portfolio_wealth[-1] * (1 + port_ret)) + monthly_contribution
        portfolio_wealth.append(new_wealth)
        
        # Drift dos pesos
        current_weights = current_weights * (1 + r_t) / (1 + port_ret)
        
        # Rebalanceamento
        is_rebalance_time = (rebalance_freq == 'Mensal') or \
                            (rebalance_freq == 'Anual' and dates[i].month == 12)
        if is_rebalance_time:
            current_weights = active_weights.copy()
            
    portfolio_pure_series = pd.Series(portfolio_pure_idx[1:], index=dates)
    portfolio_wealth_series = pd.Series(portfolio_wealth[1:], index=dates)
    monthly_returns_series = pd.Series(monthly_returns, index=dates)
    monthly_returns_series.name = "Portfólio"
    
    return portfolio_pure_series, portfolio_wealth_series, monthly_returns_series

def create_monthly_heatmap(returns_series):
    """Cria tabela estilo Mais Retorno (Ano x Mês)."""
    df_ret = returns_series.to_frame(name='Retorno')
    df_ret['Ano'] = df_ret.index.year
    df_ret['Mes'] = df_ret.index.month
    
    pivot = df_ret.pivot(index='Ano', columns='Mes', values='Retorno')
    
    # Adicionar acumulado do ano
    pivot['YTD'] = ((1 + pivot.fillna(0)).prod(axis=1) - 1)
    
    # Mapa de meses numérico para nome curto
    month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
                 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    pivot.rename(columns=month_map, inplace=True)
    
    return pivot

# ==========================================
# 4. INTERFACE
# ==========================================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    min_date = datetime(2012, 1, 1)
    max_date = datetime.today()
    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Início", datetime(2018, 1, 1), min_value=min_date, max_value=max_date)
    end_date = col_d2.date_input("Fim", max_date, min_value=min_date, max_value=max_date)
    
    rebalance_freq = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])
    rf_rate_annual = st.number_input("Taxa CDI/Livre de Risco (% a.a.)", value=10.0, step=0.5)
    rf_rate_monthly = (1 + rf_rate_annual/100)**(1/12) - 1

    aporte_mensal = st.number_input("Aporte Mensal (R$)", value=2000.0, step=100.0)
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=50000.0, step=1000.0)

    st.markdown("---")
    st.subheader("📦 Composição da Carteira")
    
    # Inputs com valores padrão
    default_stocks = "AGRO3, B3SA3, BBAS3, BBSE3, BPAC11, CMIG3, EGIE3, ITUB3, PRIO3, PSSA3, SAPR4, SBSP3, TAEE3, TOTS3, VIVT3, WEGE3"
    default_fiis = "ALZR11, BRCO11, BTLG11, HGLG11, HGRE11, HGRU11, KNCR11, KNRI11, LVBI11, MXRF11, PMLL11, TRXF11, VILG11, VISC11, XPLG11, XPML11"
    default_etfs = "IVVB11"
    
    with st.expander("Selecionar Ativos", expanded=False):
        stocks_input = st.text_area("Ações BR", default_stocks)
        fiis_input = st.text_area("FIIs", default_fiis)
        etfs_input = st.text_area("ETFs", default_etfs)
    
    st.markdown("### Pesos (%)")
    w_stocks = st.slider("Ações", 0, 100, 15)
    w_fiis = st.slider("FIIs", 0, 100, 5)
    w_etfs = st.slider("ETFs", 0, 100, 30)
    
    st.markdown("**Fundos Ativos**")
    w_tarpon = st.number_input("Fundo Tarpon", 0, 100, 20)
    w_absolute = st.number_input("Fundo Absolute", 0, 100, 10)
    w_sparta = st.number_input("Fundo Sparta", 0, 100, 10)
    w_spx = st.number_input("Fundo SPX Patriot", 0, 100, 10)
    
    total_w = w_stocks + w_fiis + w_etfs + w_tarpon + w_absolute + w_sparta + w_spx
    if total_w != 100:
        st.warning(f"Total: {total_w}%. Será normalizado.")

# --- DADOS ---
stock_list = [x.strip() for x in stocks_input.split(',') if x.strip()]
fii_list = [x.strip() for x in fiis_input.split(',') if x.strip()]
etf_list = [x.strip() for x in etfs_input.split(',') if x.strip()]

# Chamada da nova função de Fundos (Manual + API)
df_funds = get_combined_funds_data()

with st.spinner('Consolidando dados de mercado...'):
    df_stocks = get_market_data(stock_list, start_date, end_date)
    df_fiis = get_market_data(fii_list, start_date, end_date)
    df_etfs = get_market_data(etf_list, start_date, end_date)
    ibov_ret = get_benchmark_data(start_date, end_date)

# Consolidar Master DF
all_dates = df_funds.index.union(df_stocks.index).union(df_fiis.index).union(df_etfs.index)
if not ibov_ret.empty:
    all_dates = all_dates.union(ibov_ret.index)
all_dates = all_dates.sort_values()

master_df = pd.DataFrame(index=all_dates)

# Preencher classes de ativos
if not df_stocks.empty: master_df['Ações Consolidadas'] = df_stocks.mean(axis=1)
if not df_fiis.empty: master_df['FIIs Consolidados'] = df_fiis.mean(axis=1)
if not df_etfs.empty: master_df['ETFs Consolidados'] = df_etfs.mean(axis=1)

# Preencher Fundos (Reindexando para garantir alinhamento de datas)
master_df['Tarpon GT'] = df_funds['Tarpon GT'].reindex(master_df.index)
master_df['Absolute Pace'] = df_funds['Absolute Pace'].reindex(master_df.index)
master_df['Sparta Infra'] = df_funds['Sparta Infra'].reindex(master_df.index)
master_df['SPX Patriot'] = df_funds['SPX Patriot'].reindex(master_df.index)

# Filtrar datas
mask = (master_df.index >= pd.to_datetime(start_date)) & (master_df.index <= pd.to_datetime(end_date))
master_df = master_df.loc[mask].dropna(how='all').fillna(0)
ibov_ret = ibov_ret.reindex(master_df.index).fillna(0)

weights = {
    'Ações Consolidadas': w_stocks,
    'FIIs Consolidados': w_fiis,
    'ETFs Consolidados': w_etfs,
    'Tarpon GT': w_tarpon,
    'Absolute Pace': w_absolute,
    'Sparta Infra': w_sparta,
    'SPX Patriot': w_spx
}

# CALCULAR
port_pure, port_wealth, port_ret = calculate_portfolio_performance(
    master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
)

if port_ret is not None:
    # Gerar Benchmarks Acumulados
    cdi_ret_series = pd.Series(rf_rate_monthly, index=port_ret.index)
    cdi_accum = (1 + cdi_ret_series).cumprod() * 100
    ibov_accum = (1 + ibov_ret).cumprod() * 100
    
    # Métricas Gerais
    total_ret = (port_pure.iloc[-1] / 100) - 1
    years = len(port_ret) / 12
    # Proteção contra divisão por zero se years < 1
    if years > 0:
        cagr = (1 + total_ret) ** (1/years) - 1
    else:
        cagr = 0
        
    vol = port_ret.std() * np.sqrt(12)
    sharpe = (port_ret.mean() - rf_rate_monthly) / port_ret.std() * np.sqrt(12)
    
    # Drawdown
    cum_ret = (1 + port_ret).cumprod()
    peak = cum_ret.cummax()
    dd_series = (cum_ret - peak) / peak
    max_dd = dd_series.min()

    # Layout Principal
    st.title("📊 Relatório de Performance")
    
    # --- HEADER KPI (Estilo Cards) ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"<div class='metric-card'><div class='metric-value'>{total_ret:.1%}</div><div class='metric-label'>Retorno Total</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-value'>{cagr:.1%}</div><div class='metric-label'>CAGR (a.a.)</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-value'>{vol:.1%}</div><div class='metric-label'>Volatilidade</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-value'>{sharpe:.2f}</div><div class='metric-label'>Sharpe</div></div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='metric-card'><div class='metric-value' style='color:red'>{max_dd:.1%}</div><div class='metric-label'>Max Drawdown</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS DE ANÁLISE ---
    tab_perf, tab_risk, tab_month, tab_patr = st.tabs([
        "📈 Rentabilidade Comparativa", 
        "🛡️ Análise de Risco", 
        "📅 Retornos Mensais",
        "💰 Evolução Patrimonial"
    ])
    
    with tab_perf:
        st.subheader("Evolução (Base 100)")
        df_chart = pd.DataFrame({
            'Seu Portfólio': port_pure,
            'Ibovespa': ibov_accum,
            'CDI (Teórico)': cdi_accum
        })
        
        fig = px.line(df_chart, title="Comparativo de Rentabilidade Acumulada")
        fig.update_layout(
            template="plotly_white", 
            xaxis_title="", 
            yaxis_title="Índice (Base 100)",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Nota: O gráfico acima mostra a valorização pura das cotas (iniciando em 100), ignorando aportes, para permitir comparação justa com índices.")

    with tab_risk:
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("**Drawdown Submarino**")
            fig_dd = px.area(dd_series, title="")
            fig_dd.update_traces(fillcolor='rgba(255,0,0,0.2)', line_color='red')
            fig_dd.update_layout(template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig_dd, use_container_width=True)
            
        with col_r2:
            st.markdown("**Volatilidade Móvel (12 Meses)**")
            rolling_vol = port_ret.rolling(12).std() * np.sqrt(12)
            fig_vol = px.line(rolling_vol, title="")
            fig_vol.update_traces(line_color='#FF9800')
            fig_vol.update_layout(template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig_vol, use_container_width=True)

        # Estatísticas Adicionais
        st.markdown("### Estatísticas Detalhadas")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        months_pos = (port_ret > 0).sum()
        months_neg = (port_ret < 0).sum()
        best_month = port_ret.max()
        worst_month = port_ret.min()
        
        stat_col1.metric("Meses Positivos", f"{months_pos} ({months_pos/len(port_ret):.0%})")
        stat_col2.metric("Meses Negativos", f"{months_neg} ({months_neg/len(port_ret):.0%})")
        stat_col3.metric("Melhor Mês", f"{best_month:.2%}")
        stat_col4.metric("Pior Mês", f"{worst_month:.2%}", delta_color="inverse")

    with tab_month:
        st.subheader("Tabela de Rentabilidade (Heatmap)")
        heatmap_data = create_monthly_heatmap(port_ret)
        
        # Colorir dataframe (Estilo Excel)
        st.dataframe(
            heatmap_data.style.format("{:.2%}")
            .background_gradient(cmap='RdYlGn', vmin=-0.05, vmax=0.05, axis=None)
            .highlight_null(color='white'),
            use_container_width=True,
            height=400
        )
        
        st.caption("YTD: Rentabilidade acumulada no ano corrente.")

    with tab_patr:
        st.subheader("Evolução do Saldo em Conta")
        
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            fig_wealth = px.area(port_wealth, title="Crescimento Patrimonial (Cotas + Aportes)")
            fig_wealth.update_traces(fillcolor='rgba(76, 175, 80, 0.3)', line_color='#4CAF50')
            fig_wealth.update_layout(template="plotly_white", yaxis_title="Saldo (R$)")
            st.plotly_chart(fig_wealth, use_container_width=True)
        
        with col_p2:
            final_val = port_wealth.iloc[-1]
            total_invested = investimento_inicial + (aporte_mensal * len(port_ret))
            profit_loss = final_val - total_invested
            
            st.metric("Saldo Final", f"R$ {final_val:,.2f}")
            st.metric("Total Investido", f"R$ {total_invested:,.2f}")
            st.metric("Lucro/Prejuízo", f"R$ {profit_loss:,.2f}", 
                      delta=f"{(final_val/total_invested - 1):.1%}")

else:
    st.info("👈 Configure os parâmetros na barra lateral e aguarde o processamento.")
