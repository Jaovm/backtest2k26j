import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import io

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
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #7f8c8d; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. BUSCA DE DADOS DE FUNDOS (CVM API)
# ==========================================
@st.cache_data(ttl=86400)  # Cache de 24 horas
def get_fund_data_cvm(start_date, end_date):
    """
    Busca dados de cotas dos fundos diretamente do Portal de Dados Abertos da CVM.
    """
    fund_cnpjs = {
        'Tarpon GT': '22.232.927/0001-90',
        'Absolute Pace': '32.073.525/0001-43',
        'Inter Hedge': '30.877.528/0001-04',
        'SPX Patriot': '15.334.585/0001-53'
    }
    
    all_data = []
    years = range(start_date.year, end_date.year + 1)
    
    progress_bar = st.sidebar.progress(0)
    for i, year in enumerate(years):
        # A CVM disponibiliza arquivos anuais e mensais para o ano corrente
        url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{year}.zip"
        try:
            # Para fins de exemplo e performance, focamos nos anos necessários
            # Nota: No ambiente Streamlit Cloud, baixar ZIPs grandes pode ser lento.
            # Alternativa: Usar apenas dados mensais se o range for curto.
            df_year = pd.read_csv(url, sep=';', compression='zip', usecols=['CNPJ_FUNDO', 'DT_COMPTC', 'VL_QUOTA'])
            df_year = df_year[df_year['CNPJ_FUNDO'].isin(fund_cnpjs.values())]
            all_data.append(df_year)
        except:
            # Tenta baixar o mensal caso o anual ainda não exista (ex: 2026)
            for month in range(1, 13):
                if year == datetime.now().year and month > datetime.now().month: break
                m_str = f"{month:02d}"
                url_m = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{year}{m_str}.csv"
                try:
                    df_m = pd.read_csv(url_m, sep=';', usecols=['CNPJ_FUNDO', 'DT_COMPTC', 'VL_QUOTA'])
                    df_m = df_m[df_m['CNPJ_FUNDO'].isin(fund_cnpjs.values())]
                    all_data.append(df_m)
                except: continue
        progress_bar.progress((i + 1) / len(years))
    progress_bar.empty()

    if not all_data: return pd.DataFrame()

    df = pd.concat(all_data)
    df['DT_COMPTC'] = pd.to_datetime(df['DT_COMPTC'])
    
    # Pivotar e calcular retorno mensal
    pivot_df = df.pivot(index='DT_COMPTC', columns='CNPJ_FUNDO', values='VL_QUOTA')
    
    # Inverter o mapeamento para renomear colunas
    inv_map = {v: k for k, v in fund_cnpjs.items()}
    pivot_df.rename(columns=inv_map, inplace=True)
    
    # Resample mensal (última cota do mês) e variação percentual
    monthly_returns = pivot_df.resample('ME').last().pct_change()
    return monthly_returns

# ==========================================
# 2. FUNÇÕES DE DADOS (YFINANCE)
# ==========================================
@st.cache_data
def get_market_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    processed_tickers = [f"{t.strip().upper()}.SA" if "." not in t and any(c.isdigit() for c in t) else t.strip().upper() for t in tickers]
    try:
        data = yf.download(processed_tickers, start=start_date, end=end_date, progress=False)
        if data.empty: return pd.DataFrame()
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        if isinstance(prices, pd.Series): prices = prices.to_frame(name=processed_tickers[0])
        if isinstance(prices.columns, pd.MultiIndex): prices.columns = prices.columns.get_level_values(-1)
        returns = prices.resample('ME').last().pct_change()
        returns.columns = [str(c).replace('.SA', '') for c in returns.columns]
        return returns
    except: return pd.DataFrame()

@st.cache_data
def get_benchmark_data(start_date, end_date):
    try:
        ibov = yf.download("BOVA11", start=start_date, end=end_date, progress=False)['Adj Close']
        return ibov.resample('ME').last().pct_change()
    except: return pd.Series()

# ==========================================
# 3. LÓGICA DE PERFORMANCE (Mantida)
# ==========================================
def calculate_portfolio_performance(returns_df, weights, initial_cap, monthly_contribution, rebalance_freq):
    returns_df = returns_df.dropna()
    available_assets = [c for c in returns_df.columns if c in weights and weights[c] > 0]
    if not available_assets: return None, None, None
    
    active_weights = np.array([weights[c] for c in available_assets])
    active_weights = active_weights / active_weights.sum() 
    
    portfolio_pure_idx = [100.0]; portfolio_wealth = [initial_cap]; monthly_returns = []
    current_weights = active_weights.copy()
    dates = returns_df.index
    asset_returns_np = returns_df[available_assets].values
    
    for i in range(len(dates)):
        port_ret = np.dot(current_weights, asset_returns_np[i])
        monthly_returns.append(port_ret)
        portfolio_pure_idx.append(portfolio_pure_idx[-1] * (1 + port_ret))
        portfolio_wealth.append((portfolio_wealth[-1] * (1 + port_ret)) + monthly_contribution)
        current_weights = current_weights * (1 + asset_returns_np[i]) / (1 + port_ret)
        if rebalance_freq == 'Mensal' or (rebalance_freq == 'Anual' and dates[i].month == 12):
            current_weights = active_weights.copy()
            
    return pd.Series(portfolio_pure_idx[1:], index=dates), pd.Series(portfolio_wealth[1:], index=dates), pd.Series(monthly_returns, index=dates)

# ==========================================
# 4. INTERFACE
# ==========================================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    start_date = st.date_input("Início", datetime(2018, 1, 1))
    end_date = st.date_input("Fim", datetime.today())
    
    rebalance_freq = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])
    rf_rate_annual = st.number_input("Taxa CDI (% a.a.)", value=10.0)
    rf_rate_monthly = (1 + rf_rate_annual/100)**(1/12) - 1
    aporte_mensal = st.number_input("Aporte Mensal (R$)", value=2000.0)
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=50000.0)

    st.markdown("### Pesos (%)")
    w_stocks = st.slider("Ações BR", 0, 100, 10)
    w_etfs = st.slider("ETFs Int.", 0, 100, 20)
    w_tarpon = st.number_input("Tarpon GT", 0, 100, 20)
    w_absolute = st.number_input("Absolute Pace", 0, 100, 20)
    w_inter = st.number_input("Inter Hedge Inc.", 0, 100, 15)
    w_spx = st.number_input("SPX Patriot", 0, 100, 15)
    
    total_w = w_stocks + w_etfs + w_tarpon + w_absolute + w_inter + w_spx
    if total_w != 100: st.warning(f"Total: {total_w}% (Normalizado)")

# --- PROCESSAMENTO ---
with st.spinner('Baixando dados oficiais da CVM e Mercado...'):
    df_funds_api = get_fund_data_cvm(start_date, end_date)
    df_market = get_market_data(["WEGE3", "ITUB3", "IVVB11"], start_date, end_date) # Exemplo reduzido
    ibov_ret = get_benchmark_data(start_date, end_date)

# Consolidação
master_df = pd.DataFrame(index=df_funds_api.index)
if not df_market.empty:
    master_df['Ações Consolidadas'] = df_market.filter(regex='WEGE3|ITUB3').mean(axis=1) # Exemplo
    master_df['ETFs Consolidados'] = df_market.filter(regex='IVVB11').mean(axis=1)

master_df = master_df.join(df_funds_api, how='outer').fillna(0)

weights = {
    'Ações Consolidadas': w_stocks, 'ETFs Consolidados': w_etfs,
    'Tarpon GT': w_tarpon, 'Absolute Pace': w_absolute,
    'Inter Hedge': w_inter, 'SPX Patriot': w_spx
}

port_pure, port_wealth, port_ret = calculate_portfolio_performance(
    master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
)

# --- DASHBOARD ---
if port_ret is not None:
    st.title("📊 Relatório de Performance Atualizado")
    
    # KPIs
    total_ret = (port_pure.iloc[-1] / 100) - 1
    vol = port_ret.std() * np.sqrt(12)
    sharpe = (port_ret.mean() - rf_rate_monthly) / port_ret.std() * np.sqrt(12)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno Total", f"{total_ret:.1%}")
    c2.metric("Volatilidade", f"{vol:.1%}")
    c3.metric("Sharpe", f"{sharpe:.2f}")
    c4.metric("Saldo Final", f"R$ {port_wealth.iloc[-1]:,.2f}")

    tab1, tab2 = st.tabs(["Rentabilidade", "Evolução"])
    with tab1:
        df_chart = pd.DataFrame({'Portfólio': port_pure, 'Ibovespa': (1 + ibov_ret).cumprod() * 100}).fillna(method='ffill')
        st.plotly_chart(px.line(df_chart, title="Cota 100 vs Benchmark"), use_container_width=True)
    
    with tab2:
        st.plotly_chart(px.area(port_wealth, title="Patrimônio com Aportes"), use_container_width=True)
else:
    st.info("Aguardando dados...")
