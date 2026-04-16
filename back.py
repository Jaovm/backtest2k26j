import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import brfunds as brf

# ==========================================
# 0. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Asset Allocator Pro - Style Mais Retorno",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DADOS HARDCODED (HISTÓRICO PREMIUM/ANTIGO)
# ==========================================
def get_hardcoded_funds():
    # Mantendo os dados históricos que você já possui
    tarpon_returns = {
        '2018-01': 0.0721, '2018-02': 0.0003, '2018-03': 0.0306, '2018-04': -0.0229, '2018-05': -0.1069, '2018-06': -0.0888, '2018-07': 0.0823, '2018-08': -0.0363, '2018-09': -0.0203, '2018-10': 0.2285, '2018-11': 0.0805, '2018-12': 0.0432,
        '2019-01': 0.0721, '2019-02': 0.0366, '2019-03': -0.0144, '2019-04': 0.0328, '2019-05': 0.0350, '2019-06': 0.0300, '2019-07': 0.0507, '2019-08': 0.0145, '2019-09': 0.0070, '2019-10': -0.0013, '2019-11': 0.0051, '2019-12': 0.1658,
        '2020-01': 0.0140, '2020-02': -0.0663, '2020-03': -0.2902, '2020-04': 0.0864, '2020-05': 0.0858, '2020-06': 0.2421, '2020-07': 0.0899, '2020-08': -0.0260, '2020-09': -0.0425, '2020-10': -0.0183, '2020-11': 0.0902, '2020-12': 0.0696,
        '2021-01': -0.0338, '2021-02': 0.0527, '2021-03': 0.0302, '2021-04': 0.1080, '2021-05': 0.0553, '2021-06': 0.0310, '2021-07': -0.0110, '2021-08': -0.0643, '2021-09': 0.0091, '2021-10': -0.0284, '2021-11': -0.0656, '2021-12': 0.0879,
        '2022-01': 0.0128, '2022-02': 0.0552, '2022-03': 0.0762, '2022-04': -0.0622, '2022-05': 0.0010, '2022-06': -0.1066, '2022-07': 0.0968, '2022-08': 0.1024, '2022-09': 0.0365, '2022-10': 0.1124, '2022-11': -0.1026, '2022-12': -0.0475,
        '2023-01': 0.0232, '2023-02': -0.0252, '2023-03': -0.0407, '2023-04': 0.0451, '2023-05': 0.1327, '2023-06': 0.1013, '2023-07': 0.0592, '2023-08': -0.0212, '2023-09': 0.0768, '2023-10': -0.0605, '2023-11': 0.0806, '2023-12': 0.0986,
        '2024-01': -0.0617, '2024-02': 0.0224, '2024-03': 0.0673, '2024-04': -0.0233, '2024-05': -0.0324, '2024-06': 0.0021, '2024-07': 0.0210, '2024-08': 0.0392, '2024-09': -0.0029, '2024-10': 0.0052, '2024-11': -0.0301, '2024-12': -0.0072,
        '2025-01': 0.0430, '2025-02': 0.0259, '2025-03': 0.0748, '2025-04': 0.0748, '2025-05': 0.0033, '2025-06': 0.0160, '2025-07': -0.0698, '2025-08': 0.0364, '2025-09': 0.0059, '2025-10': 0.0229, '2025-11': 0.1059, '2025-12': -0.0071,
        '2026-01': 0.0596, '2026-02': 0.0156
    }
    absolute_returns = {
        '2018-12': 0.0209,
        '2019-01': 0.1064, '2019-02': 0.0345, '2019-03': 0.0295, '2019-04': 0.0270, '2019-05': 0.0213, '2019-06': 0.0414, '2019-07': 0.0496, '2019-08': 0.0184, '2019-09': 0.0124, '2019-10': 0.0445, '2019-11': 0.0254, '2019-12': 0.1110,
        '2020-01': 0.0176, '2020-02': -0.0754, '2020-03': -0.2341, '2020-04': 0.1210, '2020-05': 0.0542, '2020-06': 0.0837, '2020-07': 0.0820, '2020-08': -0.0123, '2020-09': -0.0692, '2020-10': -0.0049, '2020-11': 0.1372, '2020-12': 0.0549,
        '2021-01': -0.0237, '2021-02': -0.0202, '2021-03': 0.0648, '2021-04': 0.0517, '2021-05': 0.0459, '2021-06': 0.0108, '2021-07': -0.0279, '2021-08': -0.0126, '2021-09': -0.0242, '2021-10': -0.0528, '2021-11': 0.0250, '2021-12': 0.0513,
        '2022-01': 0.0501, '2022-02': -0.0032, '2022-03': 0.0664, '2022-04': -0.0125, '2022-05': 0.0610, '2022-06': -0.0780, '2022-07': 0.0266, '2022-08': 0.0472, '2022-09': -0.0240, '2022-10': 0.0315, '2022-11': -0.0039, '2022-12': -0.0108,
        '2023-01': 0.0027, '2023-02': -0.0244, '2023-03': -0.0051, '2023-04': 0.0224, '2023-05': 0.0488, '2023-06': 0.0968, '2023-07': 0.0495, '2023-08': -0.0263, '2023-09': -0.0063, '2023-10': -0.0395, '2023-11': 0.0958, '2023-12': 0.0395,
        '2024-01': -0.0030, '2024-02': 0.0278, '2024-03': 0.0135, '2024-04': -0.0358, '2024-05': -0.0210, '2024-06': 0.0064, '2024-07': 0.0471, '2024-08': 0.0458, '2024-09': -0.0259, '2024-10': -0.0202, '2024-11': -0.0387, '2024-12': -0.0390,
        '2025-01': 0.0551, '2025-02': -0.0286, '2025-03': 0.0365, '2025-04': 0.0824, '2025-05': 0.0670, '2025-06': 0.0254, '2025-07': -0.0583, '2025-08': 0.0613, '2025-09': 0.0498, '2025-10': 0.0088, '2025-11': 0.0618, '2025-12': -0.0217,
        '2026-01': 0.0517, '2026-02': 0.0163, '2026-03': 0.0092
    }

    df = pd.DataFrame({
        'Tarpon GT': pd.Series(tarpon_returns, dtype=float),
        'Absolute Pace': pd.Series(absolute_returns, dtype=float)
    })
    df.index = pd.to_datetime(df.index).to_period('M').to_timestamp('M')
    return df


# ==========================================
# 2. FUNÇÕES DE DADOS (CVM, YFINANCE E BCB)
# ==========================================

@st.cache_data(show_spinner=False, ttl="24h")
def get_fundos_cvm(cnpj_dict, start_date, end_date):
    """
    Recupera dados de rentabilidade via brfunds usando getFundsEarnings.
    """
    df_returns = pd.DataFrame()
    try:
        cnpjs = list(cnpj_dict.values())
        nomes = list(cnpj_dict.keys())
        
        # O brfunds exige data no formato DD/MM/YY ou DD/MM/YYYY em formato string
        start_str = start_date.strftime('%d/%m/%y')
        end_str = end_date.strftime('%d/%m/%y')
        
        # Chamada correta conforme documentação 0.2.0+
        # Usamos *cnpjs para passar a lista de strings como argumentos posicionais
        df_raw = brf.getFundsEarnings(*cnpjs, start=start_str, end=end_str)
        
        if df_raw is not None and not df_raw.empty:
            df_raw.index = pd.to_datetime(df_raw.index)
            
            # Renomear colunas para os nomes amigáveis
            # O brfunds retorna as colunas com base nos CNPJs ou nomes internos
            # Mapeamos a ordem para garantir consistência
            if len(df_raw.columns) == len(nomes):
                df_raw.columns = nomes
            
            # Como getFundsEarnings retorna a rentabilidade acumulada (base 0 ou 1),
            # precisamos transformar em retornos mensais para o resto do script.
            mensal = df_raw.resample('ME').last()
            
            # Se o brfunds retornar rentabilidade acumulada (ex: 0.91), 
            # calculamos a variação da variação para ter o retorno mensal relativo.
            # Convertemos de acumulado para retorno do período: (1+r_atual)/(1+r_anterior) - 1
            df_returns = mensal.pct_change().dropna()
            df_returns.index = df_returns.index.to_period('M').to_timestamp('M')
                
    except Exception as e:
        st.error(f"⚠️ Erro na integração com brfunds: {e}")
        return pd.DataFrame()
        
    return df_returns


def merge_historical_and_api(old_series, new_series):
    if new_series is None or new_series.empty:
        return old_series
    api_start = new_series.dropna().index.min()
    old_filtered = old_series[old_series.index < api_start]
    combined = pd.concat([old_filtered, new_series.dropna()])
    return combined.sort_index()

@st.cache_data
def get_cdi_data(start_date, end_date):
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json"
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df.set_index('data', inplace=True)
        df['valor'] = df['valor'].astype(float) / 100.0
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        return df.loc[mask, 'valor'].resample('ME').last()
    except:
        return pd.Series(dtype='float64')

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
    try:
        ibov = yf.download("BOVA11.SA", start=start_date, end=end_date, progress=False)
        if ibov.empty:
            return pd.Series(dtype='float64')

        if 'Adj Close' in ibov.columns:
            prices = ibov['Adj Close']
        else:
            prices = ibov.iloc[:, 0]

        returns = prices.resample('ME').last().pct_change().dropna()
        returns.name = "Ibovespa"
        return returns
    except Exception as e:
        st.error(f"Erro no benchmark: {e}")
        return pd.Series(dtype='float64')

# ==========================================
# 3. LÓGICA DE CÁLCULO
# ==========================================
def calculate_portfolio_performance(returns_df, weights, initial_cap, monthly_contribution, rebalance_freq):
    returns_df = returns_df.dropna()
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    
    if not available_assets: return None, None, None

    active_weights = np.array([weights[c] for c in available_assets])
    active_weights = active_weights / active_weights.sum() 
    
    portfolio_pure_idx = [100.0]
    monthly_returns = []
    portfolio_wealth = [initial_cap]
    
    current_weights = active_weights.copy()
    dates = returns_df.index
    asset_returns_np = returns_df[available_assets].values
    
    for i in range(len(dates)):
        r_t = asset_returns_np[i]
        
        port_ret = np.dot(current_weights, r_t)
        monthly_returns.append(port_ret)
        
        new_idx = portfolio_pure_idx[-1] * (1 + port_ret)
        portfolio_pure_idx.append(new_idx)
        
        new_wealth = (portfolio_wealth[-1] * (1 + port_ret)) + monthly_contribution
        portfolio_wealth.append(new_wealth)
        
        current_weights = current_weights * (1 + r_t) / (1 + port_ret)
        
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
    df_ret = returns_series.to_frame(name='Retorno')
    df_ret['Ano'] = df_ret.index.year
    df_ret['Mes'] = df_ret.index.month
    
    pivot = df_ret.pivot(index='Ano', columns='Mes', values='Retorno')
    pivot['YTD'] = ((1 + pivot.fillna(0)).prod(axis=1) - 1)
    
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

    aporte_mensal = st.number_input("Aporte Mensal (R$)", value=1000.0, step=100.0)
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=100000.0, step=1000.0)

    st.markdown("---")
    st.subheader("📦 Composição da Carteira")
    
    with st.expander("Selecionar Ativos", expanded=False):
        default_stocks = "EGIE3, ITUB3, PSSA3, WEGE3, CXSE3, SBSP3, TAEE3, VIVT3, CPFE3, SAPR3, BBAS3, PRIO3, TOTS3, BPAC3, ALUP3, BMOB3"
        default_fiis = "ALZR11, BRCO11, BTLG11, HGLG11, HGRE11, HGRU11, KNCR11, KNRI11, LVBI11, MXRF11, PMLL11, XPLG11, XPML11"
        default_etfs = "IVVB11"
        stocks_input = st.text_area("Ações BR", default_stocks)
        fiis_input = st.text_area("FIIs", default_fiis)
        etfs_input = st.text_area("ETFs", default_etfs)
    
    st.markdown("### Pesos (%)")
    w_stocks = st.slider("Ações", 0, 100, 35)
    w_fiis = st.slider("FIIs", 0, 100, 0)
    w_etfs = st.slider("ETFs", 0, 100, 20)
    
    st.markdown("**Fundos Ativos & Caixa**")
    col_f1, col_f2 = st.columns(2)
    w_tarpon = col_f1.number_input("Tarpon GT", 0, 100, 10)
    w_absolute = col_f2.number_input("Absolute Pace", 0, 100, 25)
    w_cdi = col_f1.number_input("CDI", 0, 100, 0)
    w_spx = col_f2.number_input("SPX Patriot", 0, 100, 0)
    w_real = col_f1.number_input("Real Investor", 0, 100, 0)
    w_organon = col_f2.number_input("Organon FIC", 0, 100, 10)
    
    total_w = w_stocks + w_fiis + w_etfs + w_tarpon + w_absolute + w_cdi + w_spx + w_real + w_organon
    if total_w != 100:
        st.warning(f"Total: {total_w}%. Será normalizado.")

stock_list = [x.strip() for x in stocks_input.split(',') if x.strip()]
fii_list = [x.strip() for x in fiis_input.split(',') if x.strip()]
etf_list = [x.strip() for x in etfs_input.split(',') if x.strip()]

# Mapeamento dos CNPJs dos seus fundos
fund_cnpjs = {
    'Tarpon GT': '22.232.927/0001-90',
    'Absolute Pace': '32.073.525/0001-43',
    'SPX Patriot': '15.334.585/0001-53', # Favor validar este CNPJ
    'Real Investor': '10.500.884/0001-05',
    'Organon FIC FIA': '17.400.251/0001-66'
}

# 1. Carrega os dados hardcoded (histórico antigo/premium)
df_funds_old = get_hardcoded_funds()

with st.spinner('Consolidando dados de mercado, CVM e taxas (BCB)...'):
    # Extração de Mercado (Ações, FIIs, ETFs, IBOV, CDI)
    df_stocks = get_market_data(stock_list, start_date, end_date)
    df_fiis = get_market_data(fii_list, start_date, end_date)
    df_etfs = get_market_data(etf_list, start_date, end_date)
    ibov_ret = get_benchmark_data(start_date, end_date)
    cdi_ret = get_cdi_data(start_date, end_date)
    
    # Extração da API da CVM
    df_funds_api = get_fundos_cvm(fund_cnpjs, start_date, end_date)
    
    # Mesclagem Híbrida (Manual + API)
    df_funds_hybrid = pd.DataFrame()
    for fund_name in fund_cnpjs.keys():
        old_s = df_funds_old.get(fund_name, pd.Series(dtype=float))
        api_s = df_funds_api.get(fund_name, pd.Series(dtype=float))
        df_funds_hybrid[fund_name] = merge_historical_and_api(old_s, api_s)

    # Construção do Índice Global Consolidado
    all_dates = df_funds_hybrid.index.union(df_stocks.index).union(df_fiis.index).union(df_etfs.index).union(cdi_ret.index)
    if not ibov_ret.empty:
        all_dates = all_dates.union(ibov_ret.index)
    all_dates = all_dates.sort_values()

    master_df = pd.DataFrame(index=all_dates)

    # Atribuição das colunas
    if not df_stocks.empty: master_df['Ações Consolidadas'] = df_stocks.mean(axis=1)
    if not df_fiis.empty: master_df['FIIs Consolidados'] = df_fiis.mean(axis=1)
    if not df_etfs.empty: master_df['ETFs Consolidados'] = df_etfs.mean(axis=1)

    master_df['Tarpon GT'] = df_funds_hybrid['Tarpon GT'].reindex(master_df.index)
    master_df['Absolute Pace'] = df_funds_hybrid['Absolute Pace'].reindex(master_df.index)
    master_df['SPX Patriot'] = df_funds_hybrid['SPX Patriot'].reindex(master_df.index)
    master_df['Real Investor'] = df_funds_hybrid['Real Investor'].reindex(master_df.index)
    master_df['Organon FIC FIA'] = df_funds_hybrid['Organon FIC FIA'].reindex(master_df.index)
    master_df['CDI'] = cdi_ret.reindex(master_df.index)

    # Filtragem e limpeza final
    mask = (master_df.index >= pd.to_datetime(start_date)) & (master_df.index <= pd.to_datetime(end_date))
    master_df = master_df.loc[mask].dropna(how='all').fillna(0)
    
    ibov_ret = ibov_ret.reindex(master_df.index).fillna(0)
    cdi_ret = cdi_ret.reindex(master_df.index).fillna(0)

weights = {
    'Ações Consolidadas': w_stocks,
    'FIIs Consolidados': w_fiis,
    'ETFs Consolidados': w_etfs,
    'Tarpon GT': w_tarpon,
    'Absolute Pace': w_absolute,
    'CDI': w_cdi,
    'SPX Patriot': w_spx,
    'Real Investor': w_real,
    'Organon FIC FIA': w_organon
}

port_pure, port_wealth, port_ret = calculate_portfolio_performance(
    master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
)

if port_ret is not None:
    cdi_ret_series = cdi_ret.reindex(port_ret.index).fillna(0)
    cdi_accum = (1 + cdi_ret_series).cumprod() * 100
    ibov_accum = (1 + ibov_ret).cumprod() * 100
    
    total_ret = (port_pure.iloc[-1] / 100) - 1
    years = len(port_ret) / 12
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    vol = port_ret.std() * np.sqrt(12)
    
    excess_returns = port_ret - cdi_ret_series
    sharpe = (excess_returns.mean() / port_ret.std()) * np.sqrt(12) if port_ret.std() > 0 else 0
    
    cum_ret = (1 + port_ret).cumprod()
    peak = cum_ret.cummax()
    dd_series = (cum_ret - peak) / peak
    max_dd = dd_series.min()

    st.title("📊 Relatório de Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"<div class='metric-card'><div class='metric-value'>{total_ret:.1%}</div><div class='metric-label'>Retorno Total</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-value'>{cagr:.1%}</div><div class='metric-label'>CAGR (a.a.)</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-value'>{vol:.1%}</div><div class='metric-label'>Volatilidade</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-value'>{sharpe:.2f}</div><div class='metric-label'>Sharpe (vs. CDI)</div></div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='metric-card'><div class='metric-value' style='color:red'>{max_dd:.1%}</div><div class='metric-label'>Max Drawdown</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    tab_perf, tab_risk, tab_month, tab_patr, tab_proj = st.tabs([
        "📈 Rentabilidade Comparativa",
        "🛡️ Análise de Risco",
        "📅 Retornos Mensais",
        "💰 Evolução Patrimonial",
        "🔮 Projeções (3 Anos)"
    ])
    
    with tab_perf:
        st.subheader("Evolução (Base 100)")
        df_chart = pd.DataFrame({
            'Seu Portfólio': port_pure,
            'Ibovespa': ibov_accum,
            'CDI Real (BCB)': cdi_accum
        })
        
        fig = px.line(df_chart, title="Comparativo de Rentabilidade Acumulada")
        fig.update_layout(
            template="plotly_white", 
            xaxis_title="", 
            yaxis_title="Índice (Base 100)",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")
        st.info("Nota: O gráfico acima mostra a valorização pura das cotas (iniciando em 100), ignorando aportes, para permitir comparação justa com índices.")

    with tab_risk:
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("**Drawdown Submarino**")
            fig_dd = px.area(dd_series, title="")
            fig_dd.update_traces(fillcolor='rgba(255,0,0,0.2)', line_color='red')
            fig_dd.update_layout(template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig_dd, width="stretch")
            
        with col_r2:
            st.markdown("**Volatilidade Móvel (12 Meses)**")
            rolling_vol = port_ret.rolling(12).std() * np.sqrt(12)
            fig_vol = px.line(rolling_vol, title="")
            fig_vol.update_traces(line_color='#FF9800')
            fig_vol.update_layout(template="plotly_white", yaxis_tickformat=".1%", showlegend=False)
            st.plotly_chart(fig_vol, width="stretch")

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
        
        st.dataframe(
            heatmap_data.style.format("{:.2%}")
            .background_gradient(cmap='RdYlGn', vmin=-0.05, vmax=0.05, axis=None)
            .highlight_null(color='white'),
            width="stretch",
            height=400
        )
        st.caption("YTD: Rentabilidade acumulada no ano corrente.")

        st.markdown("---")
        st.subheader("💎 Sharpe Ratio (Janelas Móveis)")
        
        sharpe_periods = {
            "12 Meses": 12,
            "24 Meses": 24,
            "48 Meses": 48,
            "60 Meses": 60,
            "Desde o Início (Completo)": len(port_ret)
        }
        
        sharpe_results = {}
        for label, months in sharpe_periods.items():
            if len(port_ret) >= months:
                subset_port = port_ret.tail(months)
                subset_cdi = cdi_ret_series.tail(months)
                vol_subset = subset_port.std()
                if vol_subset > 0:
                    excess_subset = subset_port - subset_cdi
                    sharpe_val = (excess_subset.mean() / vol_subset) * np.sqrt(12)
                    sharpe_results[label] = sharpe_val
                else:
                    sharpe_results[label] = 0.0
            else:
                sharpe_results[label] = None

        df_sharpe_table = pd.DataFrame([sharpe_results], index=["Índice de Sharpe"])
        
        st.dataframe(
            df_sharpe_table.style.format("{:.2f}", na_rep="-")
            .background_gradient(cmap='Blues', axis=1, vmin=0, vmax=2),
            width="stretch"
        )
        st.caption("ℹ️ O cálculo de excesso de retorno utiliza o histórico real e dinâmico da série do CDI (BCB), anualizando a volatilidade.")

    with tab_patr:
        st.subheader("Evolução do Saldo em Conta")
        
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            fig_wealth = px.area(port_wealth, title="Crescimento Patrimonial (Cotas + Aportes)")
            fig_wealth.update_traces(fillcolor='rgba(76, 175, 80, 0.3)', line_color='#4CAF50')
            fig_wealth.update_layout(template="plotly_white", yaxis_title="Saldo (R$)")
            st.plotly_chart(fig_wealth, width="stretch")
        
        with col_p2:
            final_val = port_wealth.iloc[-1]
            total_invested = investimento_inicial + (aporte_mensal * len(port_ret))
            profit_loss = final_val - total_invested
            
            st.metric("Saldo Final", f"R$ {final_val:,.2f}")
            st.metric("Total Investido", f"R$ {total_invested:,.2f}")
            st.metric("Lucro/Prejuízo", f"R$ {profit_loss:,.2f}", 
                      delta=f"{(final_val/total_invested - 1):.1%}")

    with tab_proj:
        st.subheader("🔮 Projeção de Cenários — Próximos 36 Meses")

        mu    = port_ret.mean()
        sigma = port_ret.std()
        N_MONTHS  = 36
        N_SIM     = 20_000
        saldo_t0  = port_wealth.iloc[-1]
        last_date = port_wealth.index[-1]

        np.random.seed(42)
        paths = np.empty((N_SIM, N_MONTHS + 1))
        paths[:, 0] = saldo_t0

        rand_returns = np.random.normal(mu, sigma, size=(N_SIM, N_MONTHS))
        for t in range(1, N_MONTHS + 1):
            paths[:, t] = paths[:, t - 1] * (1 + rand_returns[:, t - 1]) + aporte_mensal

        p_otimista   = np.percentile(paths, 95, axis=0)
        p_neutro     = np.percentile(paths, 50, axis=0)
        p_pessimista = np.percentile(paths, 5,  axis=0)

        hist_tail    = port_wealth.tail(12)
        future_dates = pd.date_range(start=last_date, periods=N_MONTHS + 1, freq="ME")[1:]
        proj_dates   = [last_date] + list(future_dates)

        fig_proj = go.Figure()

        fig_proj.add_trace(go.Scatter(
            x=list(proj_dates) + list(reversed(proj_dates)),
            y=list(p_otimista) + list(reversed(p_pessimista)),
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.09)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name="Intervalo P5–P95",
            hoverinfo="skip",
        ))

        fig_proj.add_trace(go.Scatter(
            x=hist_tail.index,
            y=hist_tail.values,
            mode="lines",
            name="Histórico Real",
            line=dict(color="#2c3e50", width=3),
            hovertemplate="<b>Histórico</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))

        fig_proj.add_trace(go.Scatter(
            x=proj_dates,
            y=p_otimista,
            mode="lines",
            name="Otimista (P95)",
            line=dict(color="#27ae60", width=2.5, dash="dash"),
            hovertemplate="<b>Otimista</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))

        fig_proj.add_trace(go.Scatter(
            x=proj_dates,
            y=p_neutro,
            mode="lines",
            name="Neutro (P50)",
            line=dict(color="#2980b9", width=2.5, dash="dot"),
            hovertemplate="<b>Neutro</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))

        fig_proj.add_trace(go.Scatter(
            x=proj_dates,
            y=p_pessimista,
            mode="lines",
            name="Pessimista (P5)",
            line=dict(color="#e74c3c", width=2.5, dash="dash"),
            hovertemplate="<b>Pessimista</b><br>%{x|%b/%Y}: R$ %{y:,.0f}<extra></extra>",
        ))

        fig_proj.add_vline(
            x=last_date,
            line_width=1.5,
            line_dash="dot",
            line_color="gray"
        )
        fig_proj.add_annotation(
            x=last_date,
            y=1,
            yref="paper",
            text=" Hoje",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color="gray", size=12)
        )

        fig_proj.update_layout(
            template="plotly_white",
            title=dict(
                text=f"Monte Carlo — {N_SIM:,} simulações | µ={mu:.2%}/mês | σ={sigma:.2%}/mês | Aporte R$ {aporte_mensal:,.0f}/mês",
                font_size=13,
            ),
            yaxis=dict(title="Saldo (R$)", tickformat=",.0f"),
            xaxis_title="",
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
            hovermode="x unified",
            margin=dict(t=80),
        )

        st.plotly_chart(fig_proj, width="stretch")

        st.markdown("### 📊 Saldo Final Projetado em 36 Meses")

        saldo_otimista   = p_otimista[-1]
        saldo_neutro     = p_neutro[-1]
        saldo_pessimista = p_pessimista[-1]

        col_p1, col_p2, col_p3 = st.columns(3)

        col_p1.metric(
            label="🟢 Cenário Otimista (P95)",
            value=f"R$ {saldo_otimista:,.2f}",
            delta=f"+R$ {saldo_otimista - saldo_t0:,.0f} vs. hoje",
        )
        col_p2.metric(
            label="🔵 Cenário Neutro (P50)",
            value=f"R$ {saldo_neutro:,.2f}",
            delta=f"+R$ {saldo_neutro - saldo_t0:,.0f} vs. hoje",
        )
        col_p3.metric(
            label="🔴 Cenário Pessimista (P5)",
            value=f"R$ {saldo_pessimista:,.2f}",
            delta=f"R$ {saldo_pessimista - saldo_t0:,.0f} vs. hoje",
            delta_color="inverse",
        )

        st.markdown("---")
        col_inf1, col_inf2, col_inf3, col_inf4 = st.columns(4)
        col_inf1.info(f"**Drift µ:** {mu:.3%}/mês")
        col_inf2.info(f"**Volatilidade σ:** {sigma:.3%}/mês")
        col_inf3.info(f"**CAGR implícito:** {(1 + mu)**12 - 1:.1%}/ano")
        col_inf4.info(f"**Saldo atual:** R$ {saldo_t0:,.2f}")

        st.caption(
            f"⚠️ Projeções geradas por {N_SIM:,} simulações de Monte Carlo com retornos distribuídos normalmente "
            f"(µ = {mu:.3%}, σ = {sigma:.3%}), incluindo aporte mensal de R$ {aporte_mensal:,.2f}. "
            "Rentabilidade passada não é garantia de retorno futuro."
        )

else:
    st.info("👈 Configure os parâmetros na barra lateral e aguarde o processamento.")
