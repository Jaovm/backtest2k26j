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
# 1. BUSCA AUTOMÁTICA DE DADOS (CVM API/OPEN DATA)
# ==========================================
@st.cache_data(ttl=86400) # Cache de 24h para não sobrecarregar o portal CVM
def get_fund_data_cvm(start_date, end_date):
    """Busca cotas históricas no Portal Brasileiro de Dados Abertos (CVM)."""
    fund_cnpjs = {
        '22.232.927/0001-90': 'Tarpon GT',
        '32.073.525/0001-43': 'Absolute Pace',
        '30.877.528/0001-04': 'Inter Hedge',
        '15.334.585/0001-53': 'SPX Patriot'
    }
    
    all_data = []
    # Itera pelos anos necessários
    for year in range(start_date.year, end_date.year + 1):
        # A CVM organiza arquivos anuais e mensais (para o ano corrente)
        if year < datetime.now().year:
            url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{year}.zip"
            try:
                df = pd.read_csv(url, sep=';', compression='zip', usecols=['CNPJ_FUNDO', 'DT_COMPTC', 'VL_QUOTA'])
                df = df[df['CNPJ_FUNDO'].isin(fund_cnpjs.keys())]
                all_data.append(df)
            except: continue
        else:
            # Para o ano atual, tenta pegar os meses individualmente
            for month in range(1, datetime.now().month + 1):
                m_str = f"{month:02d}"
                url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{year}{m_str}.csv"
                try:
                    df = pd.read_csv(url, sep=';', usecols=['CNPJ_FUNDO', 'DT_COMPTC', 'VL_QUOTA'])
                    df = df[df['CNPJ_FUNDO'].isin(fund_cnpjs.keys())]
                    all_data.append(df)
                except: continue

    if not all_data: return pd.DataFrame()

    full_df = pd.concat(all_data)
    full_df['DT_COMPTC'] = pd.to_datetime(full_df['DT_COMPTC'])
    full_df['Nome'] = full_df['CNPJ_FUNDO'].map(fund_cnpjs)
    
    # Pivotar para ter datas como index e fundos como colunas
    pivot_df = full_df.pivot_table(index='DT_COMPTC', columns='Nome', values='VL_QUOTA')
    
    # Calcular retornos mensais
    monthly_returns = pivot_df.resample('ME').last().pct_change()
    return monthly_returns

# ==========================================
# 2. DADOS DE MERCADO (YFINANCE)
# ==========================================
@st.cache_data
def get_market_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    processed = [f"{t.strip().upper()}.SA" if t.strip().upper().isalnum() else t.strip() for t in tickers]
    try:
        data = yf.download(processed, start=start_date, end=end_date, progress=False)['Adj Close']
        if isinstance(data, pd.Series): data = data.to_frame()
        returns = data.resample('ME').last().pct_change()
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
# 3. LÓGICA DE CÁLCULO (Adaptada)
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

def create_monthly_heatmap(returns_series):
    df_ret = returns_series.to_frame(name='Retorno')
    df_ret['Ano'] = df_ret.index.year
    df_ret['Mes'] = df_ret.index.month
    pivot = df_ret.pivot(index='Ano', columns='Mes', values='Retorno')
    pivot['YTD'] = ((1 + pivot.fillna(0)).prod(axis=1) - 1)
    month_map = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
    pivot.rename(columns=month_map, inplace=True)
    return pivot

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    start_date = st.date_input("Início", datetime(2018, 1, 1))
    end_date = st.date_input("Fim", datetime.today())
    
    rebalance_freq = st.selectbox("Rebalanceamento", ["Mensal", "Anual"])
    rf_rate_annual = st.number_input("Taxa CDI/Selic (% a.a.)", value=10.5)
    aporte_mensal = st.number_input("Aporte Mensal (R$)", value=2000.0)
    investimento_inicial = st.number_input("Investimento Inicial (R$)", value=50000.0)

    st.markdown("### Pesos da Carteira (%)")
    w_stocks = st.slider("Ações Consolidadas", 0, 100, 10)
    w_etfs = st.slider("ETFs Internacionais", 0, 100, 20)
    w_tarpon = st.number_input("Tarpon GT", 0, 100, 20)
    w_absolute = st.number_input("Absolute Pace", 0, 100, 20)
    w_inter = st.number_input("Inter Hedge Inc.", 0, 100, 15)
    w_spx = st.number_input("SPX Patriot", 0, 100, 15)
    
    total_w = w_stocks + w_etfs + w_tarpon + w_absolute + w_inter + w_spx
    if total_w != 100: st.warning(f"Total: {total_w}% (O script normalizará para 100%)")

# Processamento
with st.spinner('🔥 Buscando dados oficiais da CVM e Mercado...'):
    df_funds_api = get_fund_data_cvm(start_date, end_date)
    df_market = get_market_data(["WEGE3", "ITUB3", "IVVB11"], start_date, end_date) # Exemplo
    ibov_ret = get_benchmark_data(start_date, end_date)

# Consolidação do Master DF
master_df = pd.DataFrame(index=df_funds_api.index)
if not df_market.empty:
    master_df['Ações Consolidadas'] = df_market.filter(regex='WEGE3|ITUB3').mean(axis=1)
    master_df['ETFs Consolidados'] = df_market['IVVB11'] if 'IVVB11' in df_market.columns else 0

# Merge com dados da CVM
master_df = master_df.join(df_funds_api, how='outer').fillna(0)

weights = {
    'Ações Consolidadas': w_stocks, 'ETFs Consolidados': w_etfs,
    'Tarpon GT': w_tarpon, 'Absolute Pace': w_absolute,
    'Inter Hedge': w_inter, 'SPX Patriot': w_spx
}

port_pure, port_wealth, port_ret = calculate_portfolio_performance(
    master_df, weights, investimento_inicial, aporte_mensal, rebalance_freq
)

# Dashboard Visual
if port_ret is not None:
    st.title("📊 Relatório de Alocação Atualizado")
    
    # KPIs principais
    total_ret = (port_pure.iloc[-1] / 100) - 1
    vol = port_ret.std() * np.sqrt(12)
    sharpe = (port_ret.mean() - ((1 + rf_rate_annual/100)**(1/12)-1)) / port_ret.std() * np.sqrt(12)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno Total", f"{total_ret:.1%}")
    c2.metric("Volatilidade (a.a.)", f"{vol:.1%}")
    c3.metric("Sharpe", f"{sharpe:.2f}")
    c4.metric("Saldo Final", f"R$ {port_wealth.iloc[-1]:,.2f}")

    tab1, tab2, tab3 = st.tabs(["Performance", "Heatmap Mensal", "Evolução Patrimonial"])
    
    with tab1:
        ibov_accum = (1 + ibov_ret.reindex(port_pure.index).fillna(0)).cumprod() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_pure.index, y=port_pure, name="Sua Carteira", line=dict(color='green', width=3)))
        fig.add_trace(go.Scatter(x=ibov_accum.index, y=ibov_accum, name="Ibovespa", line=dict(color='gray', dash='dash')))
        fig.update_layout(title="Comparativo Base 100", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        heatmap = create_monthly_heatmap(port_ret)
        st.dataframe(heatmap.style.format("{:.2%}").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

    with tab3:
        st.plotly_chart(px.area(port_wealth, title="Patrimônio Total com Aportes"), use_container_width=True)
else:
    st.info("Aguardando carregamento de dados...")
